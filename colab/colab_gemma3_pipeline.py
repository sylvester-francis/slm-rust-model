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
        print(f"\n  FAILED: {name}")
        sys.exit(1)
    print(f"\n  DONE: {name}")


def main():
    total_start = time.time()
    print("RustMentor — Gemma3-1B Mobile Pipeline")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Repo: {HF_USERNAME}/{REPO_NAME}")
    print()

    # ══════════════════════════════════════════════
    #  PHASE 1: TRAIN (needs Unsloth + torch)
    # ══════════════════════════════════════════════

    print("Installing training dependencies...")
    run("pip install -q unsloth trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer")
    run("pip install -q 'transformers>=4.51.3,<=5.2.0'")

    # Step 1: Generate + preprocess data
    run_py("Generate & Preprocess Data", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from rustmentor.data import generate_rust_dataset, preprocess_and_merge
os.makedirs("data/processed", exist_ok=True)

count = generate_rust_dataset(
    output_path="data/processed/rust_tutor_synthetic.jsonl",
    num_samples=500,
)
print(f"Generated {{count}} synthetic samples")

count = preprocess_and_merge(
    synthetic_path="data/processed/rust_tutor_synthetic.jsonl",
    strandset_samples=3000,
    output_path="data/processed/train.jsonl",
    max_seq_length={MAX_SEQ_LENGTH},
)
print(f"Merged dataset: {{count}} samples")
""")

    # Step 2: Train
    run_py("Train Gemma3-1B-IT", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from rustmentor.training import train_model

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

print(f"Uploading adapter to {{repo_id}}")
create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path="models/{REPO_NAME}",
    repo_id=repo_id,
    token=token,
)
print(f"https://huggingface.co/{{repo_id}}")
""")

    # ══════════════════════════════════════════════
    #  PHASE 2: MERGE + CONVERT (needs litert-torch)
    # ══════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print(f"  Install litert-torch")
    print(f"{'=' * 60}\n")
    run("pip uninstall -y tensorflow tensorflow-cpu keras -q 2>/dev/null || true")
    run("pip install -q litert-torch 'protobuf>=5.26,<7.0'")
    run("pip install -q 'torchao==0.11.0' --force-reinstall --no-deps")

    # Merge + Convert + Bundle using rustmentor
    run_py("Merge, Convert, and Bundle", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from rustmentor.export import convert_gemma3_to_litert

output = convert_gemma3_to_litert(
    model_dir="models/{REPO_NAME}",
    output_dir="models/{REPO_NAME}-litert",
    model_size="{LITERT_MODEL_SIZE}",
    output_name_prefix="{REPO_NAME}",
    quantization="{LITERT_QUANT}",
    kv_cache_max_len={KV_CACHE_LEN},
    base_model="{FULL_PRECISION_MODEL}",
)
if output:
    print(f"LiteRT output: {{output}}")
else:
    print("LiteRT conversion failed")
    exit(1)
""")

    # ══════════════════════════════════════════════
    #  PHASE 3: UPLOAD .litertlm TO HUGGINGFACE
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
print(f"https://huggingface.co/{{repo_id}}")
""")

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done! ({total:.0f}s / {total/60:.1f}min)")
    print(f"  Adapter: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
    print(f"  LiteRT:  https://huggingface.co/{HF_USERNAME}/{REPO_NAME}-LiteRT")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
