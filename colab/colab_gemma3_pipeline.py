#!/usr/bin/env python3
"""
RustMentor — Gemma3-1B-IT: Train, Convert to .litertlm, Upload to HuggingFace.

Full pipeline optimized for Google AI Edge Gallery on Pixel 8 Pro.
Gemma3 gets full GPU delegation on Tensor G3 (Mali-G715).

Usage:
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    import os; from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python colab/colab_gemma3_pipeline.py
"""

import os
import sys
import subprocess
import glob
import time

# ── CONFIG ──────────────────────────────────────────────
HF_USERNAME = "sylvester-francis"
REPO_NAME = "rust-mentor-1b-mobile"
BASE_MODEL = "unsloth/gemma-3-1b-it"
FULL_PRECISION_MODEL = "google/gemma-3-1b-it"
LITERT_MODEL_SIZE = "1b"

# Training
LORA_R = 16
LORA_ALPHA = 16
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4
MAX_SEQ_LENGTH = 2048

# Export
LITERT_QUANT = "dynamic_int8"
KV_CACHE_LEN = 2048
STOP_TOKEN_IDS = [1, 106]  # <eos>, <end_of_turn>
# ────────────────────────────────────────────────────────


def run(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def run_py(name, script):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    if result.returncode != 0:
        print(f"\n  ❌ FAILED: {name}")
        sys.exit(1)
    print(f"\n  ✅ DONE: {name}")


def main():
    total_start = time.time()
    print("🦀 RustMentor — Gemma3-1B Mobile Pipeline")
    print(f"   Model: {BASE_MODEL}")
    print(f"   Repo: {HF_USERNAME}/{REPO_NAME}")
    print()

    # ══════════════════════════════════════════════
    #  PHASE 1: TRAIN (needs Unsloth + torch 2.10)
    # ══════════════════════════════════════════════

    print("📦 Installing training dependencies...")
    run("pip install -q unsloth trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer")
    run("pip install -q 'transformers>=4.51.3,<=5.2.0'")

    # Step 1: Generate + preprocess data
    run_py("Generate & Preprocess Data", """
import sys, os
sys.path.insert(0, os.getcwd())
from scripts.data_collection import generate_rust_dataset
from scripts.data_preprocessing import preprocess_and_merge

os.makedirs("data/processed", exist_ok=True)

SYSTEM_PROMPT = "You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools.\\n\\nYour teaching style:\\n- Draw parallels to Go/Python/TypeScript concepts they already know\\n- Explain ownership, borrowing, and lifetimes with practical examples\\n- When reviewing code, explain what the borrow checker is doing and why\\n- Keep explanations concise with code snippets\\n- Guide them to write the code themselves rather than giving full solutions"

count = generate_rust_dataset(
    output_path="data/processed/rust_tutor_synthetic.jsonl",
    num_samples=500,
    system_prompt=SYSTEM_PROMPT,
)
print(f"Generated {count} synthetic samples")

count = preprocess_and_merge(
    synthetic_path="data/processed/rust_tutor_synthetic.jsonl",
    strandset_samples=3000,
    output_path="data/processed/train.jsonl",
    max_seq_length=2048,
)
print(f"Merged dataset: {count} samples")
""")

    # Step 2: Train
    run_py("Train Gemma3-1B-IT", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from scripts.training import train_model

train_model(
    base_model="{BASE_MODEL}",
    data_path="data/processed/train.jsonl",
    output_dir="models/{REPO_NAME}",
    lora_r={LORA_R},
    lora_alpha={LORA_ALPHA},
    batch_size={BATCH_SIZE},
    grad_accum={GRAD_ACCUM},
    epochs={EPOCHS},
    lr={LR},
    max_seq_length={MAX_SEQ_LENGTH},
)
""")

    # Step 3: Upload adapter to HF
    run_py("Upload adapter to HuggingFace", f"""
import os
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN", "")
api = HfApi(token=token)
username = api.whoami()["name"]
repo_id = f"{{username}}/{REPO_NAME}"

print(f"Uploading adapter → {{repo_id}}")
create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path="models/{REPO_NAME}",
    repo_id=repo_id,
    token=token,
)
print(f"✅ https://huggingface.co/{{repo_id}}")
""")

    # ══════════════════════════════════════════════
    #  PHASE 2: MERGE (still torch 2.10)
    # ══════════════════════════════════════════════

    run_py("Merge adapter into base model", f"""
import os, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_dir = "models/{REPO_NAME}"
merged_dir = "models/{REPO_NAME}-litert/merged"
os.makedirs(merged_dir, exist_ok=True)

print("Loading base model: {FULL_PRECISION_MODEL} (fp16)...")
model = AutoModelForCausalLM.from_pretrained(
    "{FULL_PRECISION_MODEL}",
    torch_dtype=torch.float16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

print("Merging adapter...")
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()

# Untie lm_head for LiteRT converter
model.config.tie_word_embeddings = False
if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.clone())

print(f"Saving → {{merged_dir}}")
model.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print("✅ Merged")
""")

    # ══════════════════════════════════════════════
    #  PHASE 3: CONVERT TO .litertlm (needs litert-torch)
    # ══════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print(f"  Install litert-torch")
    print(f"{'=' * 60}\n")
    run("pip uninstall -y tensorflow tensorflow-cpu keras -q 2>/dev/null || true")
    run("pip install -q litert-torch 'protobuf>=5.26,<7.0'")
    run("pip install -q 'torchao==0.11.0' --force-reinstall --no-deps")

    # Convert to .tflite using Gemma3 converter
    merged_dir = f"models/{REPO_NAME}-litert/merged"
    output_dir = f"models/{REPO_NAME}-litert"

    print(f"\n{'=' * 60}")
    print(f"  Convert → .tflite (Gemma3 converter)")
    print(f"{'=' * 60}\n")

    run(
        f"python -m litert_torch.generative.examples.gemma3.convert_gemma3_to_tflite"
        f" --model_size={LITERT_MODEL_SIZE}"
        f" --checkpoint_path={merged_dir}"
        f" --output_path={output_dir}"
        f" --output_name_prefix={REPO_NAME}"
        f" --quantize={LITERT_QUANT}"
        f" --kv_cache_max_len={KV_CACHE_LEN}"
        f" --gpu_dynamic_shapes=true"
    )

    # Bundle .tflite + tokenizer → .litertlm
    print(f"\n{'=' * 60}")
    print(f"  Bundle → .litertlm")
    print(f"{'=' * 60}\n")

    # Write a quick bundler call with Gemma3 stop tokens
    run_py("Bundle .litertlm", f"""
import os, glob, shutil
from litert_torch.generative.utilities.litertlm_builder import build_litertlm

output_dir = "{output_dir}"
merged_dir = "{merged_dir}"
workdir = os.path.join(output_dir, "_bundle_tmp")
os.makedirs(workdir, exist_ok=True)

# Find .tflite
tflites = glob.glob(os.path.join(output_dir, "*.tflite"))
if not tflites:
    print("No .tflite found!")
    exit(1)
tflite_path = tflites[0]

# Find tokenizer
tokenizer_path = os.path.join(merged_dir, "tokenizer.json")
if not os.path.exists(tokenizer_path):
    print("No tokenizer.json found!")
    exit(1)

print(f"Bundling: {{tflite_path}}")
print(f"Tokenizer: {{tokenizer_path}}")

build_litertlm(
    tflite_model_path=tflite_path,
    workdir=workdir,
    output_path=output_dir,
    context_length={KV_CACHE_LEN},
    hf_tokenizer_model_path=tokenizer_path,
    llm_model_type="gemma3",
    stop_token_ids={STOP_TOKEN_IDS},
)

# Clean up
if os.path.exists(workdir):
    shutil.rmtree(workdir)

for f in os.listdir(output_dir):
    if f.endswith(".litertlm"):
        size_mb = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
        print(f"✅ {{f}} ({{size_mb:.0f}} MB)")
""")

    # Clean up merged dir
    import shutil
    if os.path.exists(merged_dir):
        print(f"  Cleaning up {merged_dir}...")
        shutil.rmtree(merged_dir)

    # ══════════════════════════════════════════════
    #  PHASE 4: UPLOAD .litertlm TO HUGGINGFACE
    # ══════════════════════════════════════════════

    run_py("Upload .litertlm to HuggingFace", f"""
import os, glob
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN", "")
api = HfApi(token=token)
username = api.whoami()["name"]
repo_id = f"{{username}}/{REPO_NAME}-LiteRT"

print(f"Uploading to {{repo_id}}")
create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

output_dir = "models/{REPO_NAME}-litert"
for ext in ["*.litertlm", "*.tflite"]:
    for f in glob.glob(os.path.join(output_dir, ext)):
        fname = os.path.basename(f)
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  Uploading {{fname}} ({{size_mb:.0f}} MB)")
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=fname,
            repo_id=repo_id,
            token=token,
        )
print(f"✅ https://huggingface.co/{{repo_id}}")
""")

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"🦀 Done! ({total:.0f}s / {total/60:.1f}min)")
    print(f"   Adapter: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
    print(f"   LiteRT:  https://huggingface.co/{HF_USERNAME}/{REPO_NAME}-LiteRT")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
