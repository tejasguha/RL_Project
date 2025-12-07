import os
os.environ["ACCELERATE_DISABLE_DEEPSPEED"] = "1"

import torch
print(torch.__version__)

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model
import numpy as np
import pandas as pd
from trl import DPOTrainer, DPOConfig

import pickle

# ds = load_dataset("PKU-Alignment/PKU-SafeRLHF")
# df = ds['train'].to_pandas()
# df.head()
# def make_splits_and_noise(df, seed=42):
#     np.random.seed(seed)
#     # -------- SIZE SPLITS --------
#     df_33 = df.sample(frac=0.33, random_state=seed).reset_index(drop=True)
#     df_66 = df.sample(frac=0.66, random_state=seed).reset_index(drop=True)
#     df_full = df.copy().reset_index(drop=True)
#     # -------- NOISE INJECTION --------
#     def inject_noise(df_in, noise_frac):
#         df_noisy = df_in.copy()
#         n_noise = int(len(df_noisy) * noise_frac)
#         if n_noise == 0:
#             return df_noisy
#         idx = np.random.choice(len(df_noisy), size=n_noise, replace=False)
#         # Flip 0 :left_right_arrow: 1
#         df_noisy.loc[idx, "better_response_id"] = 1 - df_noisy.loc[idx, "better_response_id"]
#         return df_noisy.reset_index(drop=True)
#     df_noise_5  = inject_noise(df_full, 0.05)
#     df_noise_10 = inject_noise(df_full, 0.10)
#     df_noise_20 = inject_noise(df_full, 0.20)
#     return {
#         "size_33": df_33,
#         "size_66": df_66,
#         "full": df_full,
#         "noise_5": df_noise_5,
#         "noise_10": df_noise_10,
#         "noise_20": df_noise_20,
#     }
# # Usage
# splits = make_splits_and_noise(df)

# df_33 = splits["size_33"]
# df_66 = splits["size_33"]
# df_full = splits["full"]
# df_noise_5 = splits["noise_5"]
# df_noise_10 = splits["noise_10"]
# df_noise_20 = splits["noise_20"]

# def convert(row):
#     if row["better_response_id"] == 0:
#         chosen = row["response_0"]
#         rejected = row["response_1"]
#     else:
#         chosen = row["response_1"]
#         rejected = row["response_0"]
#     return pd.Series({"prompt": row["prompt"], "chosen": chosen, "rejected": rejected})

# dpo_df_33 = df_33.apply(convert, axis=1)
# dpo_df_66 = df_66.apply(convert, axis=1)
# dpo_df_full = df_full.apply(convert, axis=1)
# dpo_df_noise_5 = df_noise_5.apply(convert, axis=1)
# dpo_df_noise_10 = df_noise_10.apply(convert, axis=1)
# dpo_df_noise_20 = df_noise_20.apply(convert, axis=1)

cache_dir = "./llama"
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", cache_dir=cache_dir)

lora_r = 16
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_r,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "gate_proj",
        "v_proj",
        "o_proj",
        "k_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

path = "splits.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

import os
train_dataset_map = {
    "df_33": Dataset.from_pandas(data["df_33"]),
    "df_66": Dataset.from_pandas(data["df_66"]),
    "df_full": Dataset.from_pandas(data["df_full"]),
    "df_noise_5": Dataset.from_pandas(data["noise_5"]),
    "df_noise_10": Dataset.from_pandas(data["noise_10"]),
    "df_noise_20": Dataset.from_pandas(data["noise_20"]),
    "df_noise_30": Dataset.from_pandas(data["noise_30"]),
    "df_noise_40": Dataset.from_pandas(data["noise_40"])
}
for ds in ["df_noise_30", "df_noise_40"]:
    train_datset = train_dataset_map[ds]
    os.makedirs(f"./dpo_results_llama_{ds}", exist_ok=True)
    training_args = DPOConfig(
        output_dir=f"./dpo_results_llama_{ds}",
        save_steps=4500,
        deepspeed=None
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_datset
    )
    trainer.train()
