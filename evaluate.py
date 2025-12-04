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
    ppo_checkpoint_path: str = None,
    device: str = "cuda"
):
    """
    Load a policy model.
    If ppo_checkpoint_path is None, loads the base model.
    Otherwise, loads the PPO-trained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # IMPORTANT: Set left padding for generation
    tokenizer.padding_side = "left"
    
    if ppo_checkpoint_path is None:
        # Load base model (untrained)
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
        )
    else:
        # Load PPO-trained model
        print(f"Loading PPO-trained model from {ppo_checkpoint_path}...")
        
        # Check if this is a full model or LoRA checkpoint
        config_path = os.path.join(ppo_checkpoint_path, "adapter_config.json")
        
        if os.path.exists(config_path):
            # It's a LoRA checkpoint
            print("  Detected LoRA checkpoint")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
            )
            model = PeftModel.from_pretrained(model, ppo_checkpoint_path)
            model = model.merge_and_unload()
        else:
            # It's a full model checkpoint from PPOTrainer
            print("  Detected full model checkpoint")
            try:
                # Try loading directly as a full model
                model = AutoModelForCausalLM.from_pretrained(
                    ppo_checkpoint_path,
                    torch_dtype=torch.float32,
                )
            except:
                # If that fails, load base model and manually load weights
                print("  Loading base model and applying checkpoint weights...")
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                )
                
                # Load the pretrained_model subdirectory if it exists
                pretrained_path = os.path.join(ppo_checkpoint_path, "pretrained_model")
                if os.path.exists(pretrained_path):
                    print(f"  Loading from {pretrained_path}")
                    checkpoint_model = AutoModelForCausalLM.from_pretrained(
                        pretrained_path,
                        torch_dtype=torch.float32,
                    )
                    model.load_state_dict(checkpoint_model.state_dict())
                    del checkpoint_model
                else:
                    raise ValueError(
                        f"Cannot load model from {ppo_checkpoint_path}. "
                        f"Expected either adapter_config.json or pretrained_model/ subdirectory"
                    )
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def load_saferl_evaluators(
    reward_model_name: str = "PKU-Alignment/beaver-7b-v3.0-reward",
    cost_model_name: str = "PKU-Alignment/beaver-7b-v3.0-cost",
    device: str = "cuda"
):
    """
    Load official SafeRLHF reward and cost models.
    
    Reward Model: Measures helpfulness (higher = more helpful)
    Cost Model: Measures harmfulness (lower = safer)
    
    Returns:
        reward_model, cost_model, tokenizer
    """
    if not SAFE_RLHF_AVAILABLE:
        raise ImportError(
            "safe_rlhf package not found. Install with:\n"
            "pip install safe-rlhf"
        )
    
    print(f"Loading SafeRLHF Reward Model: {reward_model_name}")
    reward_model = AutoModelForScore.from_pretrained(
        reward_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    reward_model.eval()
    
    print(f"Loading SafeRLHF Cost Model: {cost_model_name}")
    cost_model = AutoModelForScore.from_pretrained(
        cost_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
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
        batch_raw_prompts = prompts[i:i + batch_size]

        # Format prompts for Llama (same as training)
        batch_prompts = [
            f"### Question:\n{p}\n\n### Answer:\n"
            for p in batch_raw_prompts
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
    """
    Evaluate responses using SafeRLHF reward and cost models.
    
    Returns:
        Dictionary with 'rewards' (helpfulness) and 'costs' (harmfulness)
    """
    rewards = []
    costs = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating with SafeRLHF models"):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]
        
        # Format according to SafeRLHF convention
        # "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}"
        conversations = [
            f"BEGINNING OF CONVERSATION: USER: {p} ASSISTANT:{r}"
            for p, r in zip(batch_prompts, batch_responses)
        ]
        
        # Tokenize
        inputs = tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(reward_model.device)
        
        # Get reward scores (helpfulness)
        with torch.no_grad():
            reward_scores = reward_model(**inputs).end_scores.squeeze(-1)
            rewards.extend(reward_scores.cpu().tolist())
        
        # Get cost scores (harmfulness - lower is better)
        with torch.no_grad():
            cost_scores = cost_model(**inputs).end_scores.squeeze(-1)
            costs.extend(cost_scores.cpu().tolist())
    
    return {
        'rewards': rewards,  # Higher = more helpful
        'costs': costs,      # Lower = safer
    }


# ============================================================
# Metrics Computation
# ============================================================

def compute_improvement_metrics(
    baseline_rewards: List[float],
    ppo_rewards: List[float],
    baseline_costs: List[float],
    ppo_costs: List[float],
) -> Dict[str, float]:
    """
    Compute metrics showing improvement from baseline to PPO.
    
    Rewards: Higher is better (helpfulness)
    Costs: Lower is better (safety)
    """
    baseline_rewards = np.array(baseline_rewards)
    ppo_rewards = np.array(ppo_rewards)
    baseline_costs = np.array(baseline_costs)
    ppo_costs = np.array(ppo_costs)
    
    metrics = {
        # Helpfulness (Reward) - higher is better
        "baseline_mean_reward": float(np.mean(baseline_rewards)),
        "ppo_mean_reward": float(np.mean(ppo_rewards)),
        "reward_improvement": float(np.mean(ppo_rewards) - np.mean(baseline_rewards)),
        "reward_improvement_pct": float((np.mean(ppo_rewards) - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards)) * 100),
        
        # Safety (Cost) - lower is better
        "baseline_mean_cost": float(np.mean(baseline_costs)),
        "ppo_mean_cost": float(np.mean(ppo_costs)),
        "cost_reduction": float(np.mean(baseline_costs) - np.mean(ppo_costs)),  # Positive = safer
        "cost_reduction_pct": float((np.mean(baseline_costs) - np.mean(ppo_costs)) / abs(np.mean(baseline_costs)) * 100),
        
        # Win rates
        "ppo_more_helpful_pct": float(np.mean(ppo_rewards > baseline_rewards) * 100),
        "ppo_safer_pct": float(np.mean(ppo_costs < baseline_costs) * 100),
        
        # Combined: safer AND more helpful
        "ppo_better_both_pct": float(np.mean((ppo_rewards > baseline_rewards) & (ppo_costs < baseline_costs)) * 100),
        
        # Statistical tests
        "reward_improvement_pvalue": float(
            stats.wilcoxon(ppo_rewards, baseline_rewards, alternative='greater')[1]
        ) if len(ppo_rewards) > 1 else 1.0,
        
        "cost_reduction_pvalue": float(
            stats.wilcoxon(baseline_costs, ppo_costs, alternative='greater')[1]
        ) if len(ppo_costs) > 1 else 1.0,
        
        # Sample stats
        "num_samples": len(baseline_rewards),
        "baseline_reward_std": float(np.std(baseline_rewards)),
        "ppo_reward_std": float(np.std(ppo_rewards)),
        "baseline_cost_std": float(np.std(baseline_costs)),
        "ppo_cost_std": float(np.std(ppo_costs)),
    }
    
    return metrics


def load_evaluation_prompts(
    dataset_name: str = "PKU-Alignment/PKU-SafeRLHF",
    split: str = "test",
    num_samples: int = None
) -> List[str]:
    """
    Load evaluation prompts from PKU-SafeRLHF test split.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Which split to use ('test' or 'train')
        num_samples: Maximum number of prompts to use (None = use all)
    
    Returns:
        List of prompt strings
    """
    print(f"Loading evaluation prompts from {dataset_name} ({split} split)...")
    ds = load_dataset(dataset_name)
    df = ds[split].to_pandas()
    
    # Extract unique prompts
    prompts = df["prompt"].unique().tolist()
    
    if num_samples is not None and num_samples < len(prompts):
        # Random sample
        np.random.seed(42)
        indices = np.random.choice(len(prompts), num_samples, replace=False)
        prompts = [prompts[i] for i in indices]
    
    print(f"Loaded {len(prompts)} evaluation prompts from {split} split")
    
    return prompts


# ============================================================
# Main Evaluation
# ============================================================

def evaluate_ppo_with_saferl(
    base_model_name: str = "EleutherAI/pythia-70m",
    ppo_checkpoint_path: str = "ppo_outputs/noise_5/final_model",
    reward_model_name: str = "PKU-Alignment/beaver-7b-v3.0-reward",
    cost_model_name: str = "PKU-Alignment/beaver-7b-v3.0-cost",
    output_dir: str = "evaluation_results",
    num_eval_prompts: int = 1000,
    generation_batch_size: int = 8,
    eval_batch_size: int = 16,
):
    """
    Evaluate PPO-trained model using official SafeRLHF evaluators.
    
    This evaluates:
    1. Helpfulness (via Reward Model) - higher is better
    2. Safety (via Cost Model) - lower is better
    
    Following the evaluation methodology from the SafeRLHF paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============================================================
    # 1. Load Evaluation Prompts (from SafeRLHF TEST split)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: Loading Evaluation Prompts from TEST Split")
    print("="*60)
    
    prompts = load_evaluation_prompts(
        dataset_name="PKU-Alignment/PKU-SafeRLHF",
        split="test",
        num_samples=num_eval_prompts
    )
    
    # ============================================================
    # 2. Load SafeRLHF Evaluators
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Loading SafeRLHF Reward & Cost Models")
    print("="*60)
    
    reward_model, cost_model, eval_tokenizer = load_saferl_evaluators(
        reward_model_name,
        cost_model_name,
        device=device
    )
    
    # ============================================================
    # 3. Generate Baseline Responses
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Generating Responses from BASELINE Model")
    print("="*60)
    
    baseline_model, baseline_tokenizer = load_policy_model(
        base_model_name,
        ppo_checkpoint_path=None,
        device=device
    )
    
    baseline_responses = generate_responses(
        baseline_model,
        baseline_tokenizer,
        prompts,
        batch_size=generation_batch_size,
        device=device
    )
    
    del baseline_model
    torch.cuda.empty_cache()
    
    # ============================================================
    # 4. Generate PPO Responses
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Generating Responses from PPO-TRAINED Model")
    print("="*60)
    
    ppo_model, ppo_tokenizer = load_policy_model(
        base_model_name,
        ppo_checkpoint_path=ppo_checkpoint_path,
        device=device
    )
    
    ppo_responses = generate_responses(
        ppo_model,
        ppo_tokenizer,
        prompts,
        batch_size=generation_batch_size,
        device=device
    )
    
    del ppo_model
    torch.cuda.empty_cache()
    
    # ============================================================
    # 5. Evaluate Baseline with SafeRLHF Models
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: Evaluating BASELINE with SafeRLHF Models")
    print("="*60)
    
    baseline_results = evaluate_with_saferl_models(
        reward_model,
        cost_model,
        eval_tokenizer,
        prompts,
        baseline_responses,
        batch_size=eval_batch_size
    )
    
    baseline_rewards = baseline_results['rewards']
    baseline_costs = baseline_results['costs']
    
    print(f"\nBaseline Results:")
    print(f"  Mean Reward (Helpfulness): {np.mean(baseline_rewards):.4f}")
    print(f"  Mean Cost (Harmfulness):   {np.mean(baseline_costs):.4f}")
    
    # ============================================================
    # 6. Evaluate PPO with SafeRLHF Models
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: Evaluating PPO with SafeRLHF Models")
    print("="*60)
    
    ppo_results = evaluate_with_saferl_models(
        reward_model,
        cost_model,
        eval_tokenizer,
        prompts,
        ppo_responses,
        batch_size=eval_batch_size
    )
    
    ppo_rewards = ppo_results['rewards']
    ppo_costs = ppo_results['costs']
    
    print(f"\nPPO Results:")
    print(f"  Mean Reward (Helpfulness): {np.mean(ppo_rewards):.4f}")
    print(f"  Mean Cost (Harmfulness):   {np.mean(ppo_costs):.4f}")
    
    # ============================================================
    # 7. Compute Metrics
    # ============================================================
    print("\n" + "="*60)
    print("STEP 7: Computing Improvement Metrics")
    print("="*60)
    
    metrics = compute_improvement_metrics(
        baseline_rewards,
        ppo_rewards,
        baseline_costs,
        ppo_costs
    )
    
    # ============================================================
    # 8. Print Results
    # ============================================================
    print("\n" + "="*60)
    print("EVALUATION RESULTS (SafeRLHF Official Models)")
    print("="*60)
    
    print(f"\n{'='*25} HELPFULNESS (Reward) {'='*25}")
    print(f"Baseline Mean Reward: {metrics['baseline_mean_reward']:.4f} ± {metrics['baseline_reward_std']:.4f}")
    print(f"PPO Mean Reward:      {metrics['ppo_mean_reward']:.4f} ± {metrics['ppo_reward_std']:.4f}")
    print(f"Improvement:          {metrics['reward_improvement']:+.4f} ({metrics['reward_improvement_pct']:+.2f}%)")
    print(f"  └─ PPO more helpful on {metrics['ppo_more_helpful_pct']:.1f}% of samples")
    print(f"Statistical Significance: p = {metrics['reward_improvement_pvalue']:.6f}")
    
    print(f"\n{'='*25} SAFETY (Cost) {'='*25}")
    print(f"Baseline Mean Cost: {metrics['baseline_mean_cost']:.4f} ± {metrics['baseline_cost_std']:.4f}")
    print(f"PPO Mean Cost:      {metrics['ppo_mean_cost']:.4f} ± {metrics['ppo_cost_std']:.4f}")
    print(f"Reduction:          {metrics['cost_reduction']:+.4f} ({metrics['cost_reduction_pct']:+.2f}%)")
    print(f"  └─ PPO safer on {metrics['ppo_safer_pct']:.1f}% of samples")
    print(f"Statistical Significance: p = {metrics['cost_reduction_pvalue']:.6f}")
    
    print(f"\n{'='*25} COMBINED METRICS {'='*25}")
    print(f"PPO better on BOTH metrics: {metrics['ppo_better_both_pct']:.1f}% of samples")
    
    print(f"\n{'='*25} SUMMARY {'='*25}")
    
    # Helpfulness summary
    if metrics['reward_improvement'] > 0 and metrics['reward_improvement_pvalue'] < 0.05:
        print("✓ PPO significantly improved helpfulness (p < 0.05)")
    elif metrics['reward_improvement'] > 0:
        print("~ PPO improved helpfulness but not significantly")
    else:
        print("✗ PPO did not improve helpfulness")
    
    # Safety summary
    if metrics['cost_reduction'] > 0 and metrics['cost_reduction_pvalue'] < 0.05:
        print("✓ PPO significantly improved safety (p < 0.05)")
    elif metrics['cost_reduction'] > 0:
        print("~ PPO improved safety but not significantly")
    else:
        print("✗ PPO did not improve safety")
    
    # Overall
    if (metrics['reward_improvement'] > 0 and metrics['cost_reduction'] > 0 and
        metrics['reward_improvement_pvalue'] < 0.05 and metrics['cost_reduction_pvalue'] < 0.05):
        print("\n🎉 SUCCESS: PPO significantly improved both helpfulness AND safety!")
    
    # ============================================================
    # 9. Save Results
    # ============================================================
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    # Detailed results
    results_df = pd.DataFrame({
        "prompt": prompts,
        "baseline_response": baseline_responses,
        "baseline_reward": baseline_rewards,
        "baseline_cost": baseline_costs,
        "ppo_response": ppo_responses,
        "ppo_reward": ppo_rewards,
        "ppo_cost": ppo_costs,
        "reward_delta": np.array(ppo_rewards) - np.array(baseline_rewards),
        "cost_delta": np.array(baseline_costs) - np.array(ppo_costs),  # Positive = safer
        "ppo_better_reward": np.array(ppo_rewards) > np.array(baseline_rewards),
        "ppo_better_cost": np.array(ppo_costs) < np.array(baseline_costs),
        "ppo_better_both": (np.array(ppo_rewards) > np.array(baseline_rewards)) & 
                          (np.array(ppo_costs) < np.array(baseline_costs)),
    })
    
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    # Metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    
    # Save baseline performance separately
    baseline_performance = {
        "mean_reward": metrics['baseline_mean_reward'],
        "mean_cost": metrics['baseline_mean_cost'],
        "std_reward": metrics['baseline_reward_std'],
        "std_cost": metrics['baseline_cost_std'],
        "num_samples": metrics['num_samples'],
    }
    baseline_df = pd.DataFrame([baseline_performance])
    baseline_df.to_csv(os.path.join(output_dir, "baseline_performance.csv"), index=False)
    
    # Save PPO performance separately
    ppo_performance = {
        "mean_reward": metrics['ppo_mean_reward'],
        "mean_cost": metrics['ppo_mean_cost'],
        "std_reward": metrics['ppo_reward_std'],
        "std_cost": metrics['ppo_cost_std'],
        "num_samples": metrics['num_samples'],
    }
    ppo_df = pd.DataFrame([ppo_performance])
    ppo_df.to_csv(os.path.join(output_dir, "ppo_performance.csv"), index=False)
    
    # Save improvement metrics
    improvement_metrics = {
        "reward_improvement": metrics['reward_improvement'],
        "reward_improvement_pct": metrics['reward_improvement_pct'],
        "reward_improvement_pvalue": metrics['reward_improvement_pvalue'],
        "cost_reduction": metrics['cost_reduction'],
        "cost_reduction_pct": metrics['cost_reduction_pct'],
        "cost_reduction_pvalue": metrics['cost_reduction_pvalue'],
        "ppo_more_helpful_pct": metrics['ppo_more_helpful_pct'],
        "ppo_safer_pct": metrics['ppo_safer_pct'],
        "ppo_better_both_pct": metrics['ppo_better_both_pct'],
    }
    improvement_df = pd.DataFrame([improvement_metrics])
    improvement_df.to_csv(os.path.join(output_dir, "improvement_metrics.csv"), index=False)
    
    print(f"\nSaved files:")
    print(f"  - detailed_results.csv (all prompts and responses)")
    print(f"  - metrics_summary.csv (all metrics)")
    print(f"  - baseline_performance.csv (baseline model stats)")
    print(f"  - ppo_performance.csv (PPO model stats)")
    print(f"  - improvement_metrics.csv (delta/improvement stats)")
    
    # ============================================================
    # 10. Show Examples
    # ============================================================
    print("\n" + "="*60)
    print("EXAMPLE COMPARISONS (Most Improved)")
    print("="*60)
    
    # Sort by combined improvement
    results_df['improvement_score'] = results_df['reward_delta'] + results_df['cost_delta']
    results_df_sorted = results_df.sort_values('improvement_score', ascending=False)
    
    for i in range(min(5, len(results_df))):
        row = results_df_sorted.iloc[i]
        print(f"\nExample {i+1}:")
        print(f"Prompt: {row['prompt'][:150]}...")
        
        print(f"\nBaseline Response: {row['baseline_response'][:150]}...")
        print(f"  Reward (Helpfulness): {row['baseline_reward']:.4f}")
        print(f"  Cost (Harmfulness):   {row['baseline_cost']:.4f}")
        
        print(f"\nPPO Response: {row['ppo_response'][:150]}...")
        print(f"  Reward (Helpfulness): {row['ppo_reward']:.4f}")
        print(f"  Cost (Harmfulness):   {row['ppo_cost']:.4f}")
        
        print(f"\nImprovement:")
        print(f"  Reward Δ: {row['reward_delta']:+.4f} {'✓' if row['ppo_better_reward'] else '✗'}")
        print(f"  Cost Δ:   {row['cost_delta']:+.4f} {'✓' if row['ppo_better_cost'] else '✗'}")
        print("-" * 60)
    
    print(f"\nAll results saved to {output_dir}/")
    
    return metrics, results_df


def evaluate_single_checkpoint_with_saferl(
    base_model_name: str,
    ppo_checkpoint_path: str,
    reward_model_name: str = "PKU-Alignment/beaver-7b-v3.0-reward",
    cost_model_name: str = "PKU-Alignment/beaver-7b-v3.0-cost",
    num_eval_prompts: int = 500,
    generation_batch_size: int = 8,
    eval_batch_size: int = 16,
):
    """
    Evaluate ONLY the PPO checkpoint.
    Store all outputs inside the SAME directory as the checkpoint.
    No baseline, no comparisons.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directory = SAME directory as checkpoint
    output_dir = os.path.join(ppo_checkpoint_path, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 1: Loading Evaluation Prompts")
    print("="*60)

    prompts = load_evaluation_prompts(
        dataset_name="PKU-Alignment/PKU-SafeRLHF",
        split="test",
        num_samples=num_eval_prompts
    )

    print("\n" + "="*60)
    print("STEP 2: Loading SafeRLHF Reward & Cost Models")
    print("="*60)

    reward_model, cost_model, eval_tokenizer = load_saferl_evaluators(
        reward_model_name,
        cost_model_name,
        device=device
    )

    print("\n" + "="*60)
    print("STEP 3: Loading PPO Model")
    print("="*60)

    ppo_model, ppo_tokenizer = load_policy_model(
        base_model_name,
        ppo_checkpoint_path=ppo_checkpoint_path,
        device=device
    )

    print("\n" + "="*60)
    print("STEP 4: Generating Responses")
    print("="*60)

    responses = generate_responses(
        ppo_model,
        ppo_tokenizer,
        prompts,
        batch_size=generation_batch_size,
        device=device
    )

    del ppo_model
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("STEP 5: Evaluating Responses")
    print("="*60)

    results = evaluate_with_saferl_models(
        reward_model,
        cost_model,
        eval_tokenizer,
        prompts,
        responses,
        batch_size=eval_batch_size
    )

    rewards = np.array(results["rewards"])
    costs = np.array(results["costs"])

    print(f"\nMean Reward: {rewards.mean():.4f}")
    print(f"Mean Cost:   {costs.mean():.4f}")

    print("\n" + "="*60)
    print("STEP 6: Saving Results")
    print("="*60)

    df = pd.DataFrame({
        "prompt": prompts,
        "response": responses,
        "reward": rewards,
        "cost": costs
    })

    df.to_csv(os.path.join(output_dir, "responses_with_scores.csv"), index=False)

    metrics = {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std()),
        "num_samples": len(rewards),
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_dir, "metrics_summary.csv"),
        index=False
    )

    print(f"\nSaved:")
    print(f"  {output_dir}/responses_with_scores.csv")
    print(f"  {output_dir}/metrics_summary.csv")

    return metrics, df


# ============================================================
# Main
# ============================================================

# def main():
#     metrics, results_df = evaluate_ppo_with_saferl(
#         base_model_name="meta-llama/Llama-3.1-8B",
#         ppo_checkpoint_path="ppo_outputs/llama7b_noise_30/checkpoint-100",
#         reward_model_name="PKU-Alignment/beaver-7b-v3.0-reward",
#         cost_model_name="PKU-Alignment/beaver-7b-v3.0-cost",
#         output_dir="evaluation_results/saferl_official",
#         num_eval_prompts=1000,  # Use 1000 prompts from test set
#         generation_batch_size=8,
#         eval_batch_size=16,
#     )
    
#     print("\n" + "="*60)
#     print("Evaluation complete!")
#     print(f"Results show performance on PKU-SafeRLHF TEST split")
#     print("="*60)

def main():
    metrics, df = evaluate_single_checkpoint_with_saferl(
        base_model_name="meta-llama/Llama-3.1-8B",
        ppo_checkpoint_path="ppo_outputs/llama3.18b_noise20_redo_paper_setup/best_checkpoint",
        reward_model_name="PKU-Alignment/beaver-7b-v3.0-reward",
        cost_model_name="PKU-Alignment/beaver-7b-v3.0-cost",
        num_eval_prompts=500,
        generation_batch_size=8,
        eval_batch_size=16,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
