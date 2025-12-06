import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from scipy import stats

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# For loading SafeRLHF reward/cost models
try:
    from safe_rlhf.models import AutoModelForScore
    SAFE_RLHF_AVAILABLE = True
except ImportError:
    print("Warning: safe_rlhf not installed. Install with: pip install safe-rlhf")
    SAFE_RLHF_AVAILABLE = False

from peft import PeftModel

# ============================================================
# Model Loading
# ============================================================

def load_policy_model(
    base_model_name: str,
    checkpoint_path: str = None,
    device: str = "cuda"
):
    """
    Load a policy model (baseline or DPO-trained) with LLaMA3 rope_scaling fix.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base or checkpoint model
    if checkpoint_path is None:
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
    else:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(config_path):
            # LoRA checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(model, checkpoint_path)
            model = model.merge_and_unload()
        else:
            # Full model checkpoint
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float32,
                device_map="auto"
            )

    model.eval()
    return model, tokenizer


def load_saferl_evaluators(
    reward_model_name: str = "PKU-Alignment/beaver-7b-v3.0-reward",
    cost_model_name: str = "PKU-Alignment/beaver-7b-v3.0-cost",
    device: str = "cuda"
):
    if not SAFE_RLHF_AVAILABLE:
        raise ImportError("safe_rlhf not installed. pip install safe-rlhf")

    print(f"Loading SafeRLHF Reward Model: {reward_model_name}")
    reward_model = AutoModelForScore.from_pretrained(
        reward_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    reward_model.eval()

    print(f"Loading SafeRLHF Cost Model: {cost_model_name}")
    cost_model = AutoModelForScore.from_pretrained(
        cost_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    cost_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    return reward_model, cost_model, tokenizer

# ============================================================
# Generation
# ============================================================

def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: str = "cuda"
) -> List[str]:

    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = [
            f"### Question:\n{p}\n\n### Answer:\n"
            for p in prompts[i:i + batch_size]
        ]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            prompt_length = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            )
            responses.append(response.strip())

    return responses

# ============================================================
# SafeRLHF Evaluation
# ============================================================

def evaluate_with_saferl_models(
    reward_model,
    cost_model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    batch_size: int = 8
) -> Dict[str, List[float]]:
    rewards = []
    costs = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating with SafeRLHF models"):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]

        conversations = [
            f"BEGINNING OF CONVERSATION: USER: {p} ASSISTANT:{r}"
            for p, r in zip(batch_prompts, batch_responses)
        ]

        inputs = tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(reward_model.device)

        with torch.no_grad():
            reward_scores = reward_model(**inputs).end_scores.squeeze(-1)
            rewards.extend(reward_scores.cpu().tolist())

        with torch.no_grad():
            cost_scores = cost_model(**inputs).end_scores.squeeze(-1)
            costs.extend(cost_scores.cpu().tolist())

    return {'rewards': rewards, 'costs': costs}

# ============================================================
# Metrics Computation
# ============================================================

def compute_improvement_metrics(
    baseline_rewards: List[float],
    dpo_rewards: List[float],
    baseline_costs: List[float],
    dpo_costs: List[float],
) -> Dict[str, float]:

    baseline_rewards = np.array(baseline_rewards)
    dpo_rewards = np.array(dpo_rewards)
    baseline_costs = np.array(baseline_costs)
    dpo_costs = np.array(dpo_costs)

    metrics = {
        "baseline_mean_reward": float(np.mean(baseline_rewards)),
        "dpo_mean_reward": float(np.mean(dpo_rewards)),
        "reward_improvement": float(np.mean(dpo_rewards) - np.mean(baseline_rewards)),
        "reward_improvement_pct": float((np.mean(dpo_rewards) - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards)) * 100),

        "baseline_mean_cost": float(np.mean(baseline_costs)),
        "dpo_mean_cost": float(np.mean(dpo_costs)),
        "cost_reduction": float(np.mean(baseline_costs) - np.mean(dpo_costs)),
        "cost_reduction_pct": float((np.mean(baseline_costs) - np.mean(dpo_costs)) / abs(np.mean(baseline_costs)) * 100),

        "dpo_more_helpful_pct": float(np.mean(dpo_rewards > baseline_rewards) * 100),
        "dpo_safer_pct": float(np.mean(dpo_costs < baseline_costs) * 100),
        "dpo_better_both_pct": float(np.mean((dpo_rewards > baseline_rewards) & (dpo_costs < baseline_costs)) * 100),

        "reward_improvement_pvalue": float(stats.wilcoxon(dpo_rewards, baseline_rewards, alternative='greater')[1]) if len(dpo_rewards) > 1 else 1.0,
        "cost_reduction_pvalue": float(stats.wilcoxon(baseline_costs, dpo_costs, alternative='greater')[1]) if len(dpo_costs) > 1 else 1.0,

        "num_samples": len(baseline_rewards),
        "baseline_reward_std": float(np.std(baseline_rewards)),
        "dpo_reward_std": float(np.std(dpo_rewards)),
        "baseline_cost_std": float(np.std(baseline_costs)),
        "dpo_cost_std": float(np.std(dpo_costs)),
    }
    return metrics

# ============================================================
# Load Prompts
# ============================================================

def load_evaluation_prompts(
    dataset_name: str = "PKU-Alignment/PKU-SafeRLHF",
    split: str = "test",
    num_samples: int = None
) -> List[str]:
    print(f"Loading evaluation prompts from {dataset_name} ({split} split)...")
    ds = load_dataset(dataset_name)
    df = ds[split].to_pandas()
    prompts = df["prompt"].unique().tolist()

    if num_samples is not None and num_samples < len(prompts):
        np.random.seed(42)
        indices = np.random.choice(len(prompts), num_samples, replace=False)
        prompts = [prompts[i] for i in indices]

    print(f"Loaded {len(prompts)} evaluation prompts from {split} split")
    return prompts

# ============================================================
# Main Evaluation
# ============================================================

def evaluate_dpo_with_saferl(
    base_model_name: str,
    dpo_checkpoint_path: str,
    reward_model_name: str = "PKU-Alignment/beaver-7b-v3.0-reward",
    cost_model_name: str = "PKU-Alignment/beaver-7b-v3.0-cost",
    output_dir: str = "evaluation_results",
    num_eval_prompts: int = 500,
    generation_batch_size: int = 8,
    eval_batch_size: int = 16,
):

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*60)
    print("STEP 1: Loading Evaluation Prompts")
    print("="*60)
    prompts = load_evaluation_prompts(num_samples=num_eval_prompts)

    print("\n" + "="*60)
    print("STEP 2: Loading SafeRLHF Evaluators")
    print("="*60)
    reward_model, cost_model, eval_tokenizer = load_saferl_evaluators(
        reward_model_name, cost_model_name, device=device
    )

    print("\n" + "="*60)
    print("STEP 3: Generating Baseline Responses")
    print("="*60)
    baseline_model, baseline_tokenizer = load_policy_model(base_model_name, checkpoint_path=None, device=device)
    baseline_responses = generate_responses(baseline_model, baseline_tokenizer, prompts, batch_size=generation_batch_size)
    del baseline_model
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("STEP 4: Generating DPO Responses")
    print("="*60)
    dpo_model, dpo_tokenizer = load_policy_model(base_model_name, checkpoint_path=dpo_checkpoint_path, device=device)
    dpo_responses = generate_responses(dpo_model, dpo_tokenizer, prompts, batch_size=generation_batch_size)
    del dpo_model
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("STEP 5: Evaluating Baseline")
    print("="*60)
    baseline_results = evaluate_with_saferl_models(reward_model, cost_model, eval_tokenizer, prompts, baseline_responses, batch_size=eval_batch_size)
    baseline_rewards = baseline_results['rewards']
    baseline_costs = baseline_results['costs']

    print("\n" + "="*60)
    print("STEP 6: Evaluating DPO")
    print("="*60)
    dpo_results = evaluate_with_saferl_models(reward_model, cost_model, eval_tokenizer, prompts, dpo_responses, batch_size=eval_batch_size)
    dpo_rewards = dpo_results['rewards']
    dpo_costs = dpo_results['costs']

    print("\n" + "="*60)
    print("STEP 7: Computing Metrics")
    print("="*60)
    metrics = compute_improvement_metrics(baseline_rewards, dpo_rewards, baseline_costs, dpo_costs)

    # ============================================================
    # Save Results
    # ============================================================
    results_df = pd.DataFrame({
        "prompt": prompts,
        "baseline_response": baseline_responses,
        "baseline_reward": baseline_rewards,
        "baseline_cost": baseline_costs,
        "dpo_response": dpo_responses,
        "dpo_reward": dpo_rewards,
        "dpo_cost": dpo_costs,
        "reward_delta": np.array(dpo_rewards) - np.array(baseline_rewards),
        "cost_delta": np.array(baseline_costs) - np.array(dpo_costs),
        "dpo_better_reward": np.array(dpo_rewards) > np.array(baseline_rewards),
        "dpo_better_cost": np.array(dpo_costs) < np.array(baseline_costs),
        "dpo_better_both": (np.array(dpo_rewards) > np.array(baseline_rewards)) & (np.array(dpo_costs) < np.array(baseline_costs)),
    })
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

    # ============================================================
    # Show Example Improvements
    # ============================================================
    results_df['improvement_score'] = results_df['reward_delta'] + results_df['cost_delta']
    results_df_sorted = results_df.sort_values('improvement_score', ascending=False)
    for i in range(min(5, len(results_df_sorted))):
        row = results_df_sorted.iloc[i]
        print(f"\nExample {i+1}:")
        print(f"Prompt: {row['prompt'][:150]}...")
        print(f"Baseline: {row['baseline_response'][:150]}... | Reward: {row['baseline_reward']:.4f} | Cost: {row['baseline_cost']:.4f}")
        print(f"DPO:      {row['dpo_response'][:150]}... | Reward: {row['dpo_reward']:.4f} | Cost: {row['dpo_cost']:.4f}")
        print(f"ΔReward: {row['reward_delta']:+.4f} {'✓' if row['dpo_better_reward'] else '✗'} | ΔCost: {row['cost_delta']:+.4f} {'✓' if row['dpo_better_cost'] else '✗'}")
        print("-"*60)

    print(f"\nAll results saved to {output_dir}/")
    return metrics, results_df

# ============================================================
# Main
# ============================================================

def main():
    metrics, df = evaluate_dpo_with_saferl(
        base_model_name="meta-llama/Llama-3.1-8B",
        dpo_checkpoint_path="dpo_models/8b/dpo_results_df_noise_5/checkpoint-9000",
        reward_model_name="PKU-Alignment/beaver-7b-v3.0-reward",
        cost_model_name="PKU-Alignment/beaver-7b-v3.0-cost",
        output_dir="evaluation_results/dpo",
        num_eval_prompts=500,
        generation_batch_size=8,
        eval_batch_size=16,
    )
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
