import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle as pkl

# ============================================================
# 0. Utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def moving_average(x, w=200):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode="valid")

# ============================================================
# 1. Manipulations on TRAIN ONLY
# ============================================================

def make_splits(train_df, seed=0):
    np.random.seed(seed)

    df_33 = train_df.sample(frac=0.33, random_state=seed).reset_index(drop=True)
    df_66 = train_df.sample(frac=0.66, random_state=seed).reset_index(drop=True)
    df_full = train_df.copy().reset_index(drop=True)

    def inject_noise(df_in, noise_frac):
        df_noisy = df_in.copy()
        n = int(len(df_noisy) * noise_frac)
        if n == 0:
            return df_noisy
        idx = np.random.choice(len(df_noisy), size=n, replace=False)
        df_noisy.loc[idx, "better_response_id"] = 1 - df_noisy.loc[idx, "better_response_id"]
        df_noisy.loc[idx, "safer_response_id"] = 1 - df_noisy.loc[idx, "safer_response_id"]
        return df_noisy.reset_index(drop=True)

    result = {
        "size_33": df_33,
        "size_66": df_66,
        "full": df_full,
        "noise_5": inject_noise(df_full, 0.05),
        "noise_10": inject_noise(df_full, 0.10),
        "noise_20": inject_noise(df_full, 0.20),
        "noise_30": inject_noise(df_full, 0.30),
        "noise_40": inject_noise(df_full, 0.40),
    }

    pkl.dump(result, open("splits.pkl", "wb"))
    return result


# ============================================================
# 2. Dataset
# ============================================================

class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # First use safety preference if unsafe
        if not row["is_response_0_safe"] or not row["is_response_1_safe"]:
            if row["safer_response_id"] == 0:
                pos = row["response_0"]
                neg = row["response_1"]
            else:
                pos = row["response_1"]
                neg = row["response_0"]
        else:
            # Otherwise use "better"
            if row["better_response_id"] == 0:
                pos = row["response_0"]
                neg = row["response_1"]
            else:
                pos = row["response_1"]
                neg = row["response_0"]

        return {
            "prompt": row["prompt"],
            "pos": pos,
            "neg": neg,
        }


# ============================================================
# 3. Reward Model — Llama-3.2-1B
# ============================================================

BASE_MODEL = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # *** REQUIRED for Llama ***


class RewardModel(nn.Module):
    def __init__(self, lora_adapter_path=None):
        super().__init__()

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map=None
        )

        if lora_adapter_path is not None:
            self.model = PeftModel.from_pretrained(base, lora_adapter_path)
        else:
            cfg = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj", "k_proj", "v_proj",
                    "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(base, cfg)

        hidden = self.model.config.hidden_size
        self.head = nn.Linear(hidden, 1, dtype=torch.bfloat16)


    def encode(self, prompts, replies):
        texts = [
            "PROMPT: " + p + tokenizer.eos_token + "RESPONSE: " + r
            for p, r in zip(prompts, replies)
        ]
        return tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        hs = out.hidden_states[-1]                     # bf16
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = hs[torch.arange(hs.size(0)), last_idx]

        # Cast pooled to the same dtype as the head
        pooled = pooled.to(self.head.weight.dtype)

        return self.head(pooled).squeeze(-1)


# ============================================================
# 4. Loss
# ============================================================

def contrastive_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos - neg)).mean()


# ============================================================
# 5. Accuracy
# ============================================================

@torch.no_grad()
def compute_accuracy(model, df, batch_size=32):
    model.eval()
    ds = ContrastiveDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    for batch in loader:
        enc_pos = model.encode(batch["prompt"], batch["pos"])
        enc_neg = model.encode(batch["prompt"], batch["neg"])

        enc_pos = {k: v.cuda() for k, v in enc_pos.items()}
        enc_neg = {k: v.cuda() for k, v in enc_neg.items()}

        pos_s = model(**enc_pos)
        neg_s = model(**enc_neg)

        pred = (pos_s > neg_s).long()
        gt = torch.ones_like(pred)

        correct += (pred == gt).sum().item()
        total += pred.numel()

    model.train()
    return correct / total


# ============================================================
# 6. Training loop
# ============================================================

def train_split(
    train_df,
    val_df,
    outdir,
    epochs=100,
    batch_size=8,
    lr=2e-5,
    patience=10,
    eval_every=200,
    max_batches_per_epoch=None
):
    train_loader = DataLoader(
        ContrastiveDataset(train_df),
        batch_size=batch_size,
        shuffle=True
    )

    model = RewardModel().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- REMOVED SCALER ---
    # scaler = torch.cuda.amp.GradScaler()

    best_val = -1
    best_epoch = 0
    wait = 0

    losses = []
    val_accs = []

    global_step = 0

    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep+1}")

        batch_idx = 0

        for batch in pbar:
            batch_idx += 1
            global_step += 1

            if max_batches_per_epoch is not None and batch_idx > max_batches_per_epoch:
                break

            enc_pos = model.encode(batch["prompt"], batch["pos"])
            enc_neg = model.encode(batch["prompt"], batch["neg"])

            enc_pos = {k: v.cuda() for k, v in enc_pos.items()}
            enc_neg = {k: v.cuda() for k, v in enc_neg.items()}

            # -----------------------------
            # BF16 forward pass (safe)
            # -----------------------------
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pos_s = model(**enc_pos)
                neg_s = model(**enc_neg)
                loss = contrastive_loss(pos_s, neg_s)

            # -----------------------------
            # NO SCALER — pure FP32 backward
            # -----------------------------
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

            del pos_s, neg_s, enc_pos, enc_neg
            torch.cuda.empty_cache()

            # ======================================
            # MID-EPOCH EVALUATION
            # ======================================
            if eval_every is not None and global_step % eval_every == 0:
                if len(val_df) > 100:
                    sample_size = max(1, int(0.1 * len(val_df)))
                    val_sample = val_df.sample(n=sample_size, random_state=global_step)
                else:
                    val_sample = val_df   # if small, just use all
                mid_val_acc = compute_accuracy(model, val_sample)

                print(f"[mid-epoch] step={global_step}  acc={mid_val_acc:.4f}")

                if mid_val_acc > best_val:
                    best_val = mid_val_acc
                    wait = 0
                    model.model.save_pretrained(os.path.join(outdir, "lora_adapter"))
                    torch.save(model.head.state_dict(), os.path.join(outdir, "head.pt"))
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at global step {global_step}")
                        return os.path.join(outdir, "lora_adapter"), os.path.join(outdir, "head.pt")

        # ======================================
        # END-OF-EPOCH EVAL
        # ======================================
        val_acc = compute_accuracy(model, val_df)
        val_accs.append(val_acc)
        print(f"[end-epoch] acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = ep
            wait = 0
            model.model.save_pretrained(os.path.join(outdir, "lora_adapter"))
            torch.save(model.head.state_dict(), os.path.join(outdir, "head.pt"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    pd.DataFrame({"loss": losses}).to_csv(os.path.join(outdir, "train_loss.csv"), index=False)
    pd.DataFrame({"val_acc": val_accs}).to_csv(os.path.join(outdir, "val_acc.csv"), index=False)

    return os.path.join(outdir, "lora_adapter"), os.path.join(outdir, "head.pt")


# ============================================================
# 7. Model Loading
# ============================================================

def load_trained_model(lora_path, head_path):
    model = RewardModel(lora_adapter_path=lora_path).cuda()
    model.head.load_state_dict(torch.load(head_path))
    return model


# ============================================================
# 8. Main
# ============================================================

def main():
    print("Loading PKU dataset…")
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")

    full_train_df = ds["train"].to_pandas()
    test_df       = ds["test"].to_pandas()

    ensure_dir("outputs")

    splits = make_splits(full_train_df)

    for name, manipulated_train_df in splits.items():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(f"\n=== Training on split: {name} ===")

        outdir = os.path.join("outputs", f"split_{name}")
        ensure_dir(outdir)

        train_df, val_df = train_test_split(
            manipulated_train_df,
            test_size=0.1,
            random_state=42
        )

        # print("Computing initial accuracy...")
        # temp = RewardModel().cuda()
        # init_acc = compute_accuracy(temp, test_df)
        # with open(os.path.join(outdir, "initial_acc.txt"), "w") as f:
        #     f.write(f"{init_acc:.6f}\n")
        # print(f"Initial acc: {init_acc:.4f}")

        # del temp
        # torch.cuda.empty_cache()

        # lora_path, head_path = train_split(train_df, val_df, outdir)
        lora_path, head_path = train_split(train_df, val_df, outdir, eval_every=1000)


        print("Loading best model for final evaluation...")
        model = load_trained_model(lora_path, head_path)

        final_acc = compute_accuracy(model, test_df)
        with open(os.path.join(outdir, "final_acc.txt"), "w") as f:
            f.write(f"{final_acc:.6f}\n")

        print(f"Final acc: {final_acc:.4f}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
