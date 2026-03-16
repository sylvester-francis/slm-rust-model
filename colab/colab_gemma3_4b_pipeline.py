#!/usr/bin/env python3
"""
RustMentor 4B — Gemma 3 4B-IT: Fine-Tune + GGUF Deployment Pipeline

Trains a Rust programming tutor that teaches by comparing to Python, Go,
and TypeScript. Includes refusal examples and prompt injection hardening.
Exports to GGUF (Q4_K_M) via llama.cpp for PocketPal AI / llama.cpp.

NOTE: The LiteRT Gemma3 converter only supports 1B/270M, so this pipeline
exports to GGUF instead.

Usage (Colab A100):
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    import os; from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python colab/colab_gemma3_4b_pipeline.py
"""

import os
import sys
import subprocess
import time
import json

# ── CONFIG ──────────────────────────────────────────────
TRAIN_MODEL = "unsloth/gemma-3-4b-it"
FULL_PRECISION_MODEL = "google/gemma-3-4b-it"
MODEL_NAME = "rust-mentor-4b"
HF_USERNAME = "sylvester-francis"
REPO_NAME = f"{MODEL_NAME}-mobile"

LORA_R = 16
LORA_ALPHA = 16
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LR = 2e-4
MAX_SEQ_LENGTH = 2048
GGUF_QUANT = "Q4_K_M"

ADAPTER_DIR = f"models/{MODEL_NAME}"
MERGED_DIR = f"models/{MODEL_NAME}-litert/merged"
OUTPUT_DIR = f"models/{MODEL_NAME}-litert"

SKIP_TRAINING = os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))
# ────────────────────────────────────────────────────────


def run(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def run_py(name, script, skip=False):
    if skip:
        print(f"\n  SKIPPED: {name}")
        return
    print(f"\n{'=' * 64}")
    print(f"  {name}")
    print(f"{'=' * 64}\n")
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

    print("=" * 64)
    print("  RustMentor 4B — Gemma 3 4B-IT Pipeline")
    print(f"  Train: {TRAIN_MODEL}")
    print(f"  Merge: {FULL_PRECISION_MODEL}")
    print(f"  Quant: {GGUF_QUANT}")
    print("=" * 64)
    print()

    # ══════════════════════════════════════════════
    #  PHASE 1A: Install dependencies
    # ══════════════════════════════════════════════

    if SKIP_TRAINING:
        print("Adapter found — skipping training")
        run("pip install -q peft transformers accelerate safetensors huggingface_hub torch torchvision")
    else:
        print("Installing training dependencies...")
        run("pip install -q unsloth trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer")
        run("pip install -q 'transformers>=4.51.3,<=5.2.0'")

    # ══════════════════════════════════════════════
    #  PHASE 1B: Build dataset + Train
    # ══════════════════════════════════════════════

    run_py("Build Dataset + Train", skip=SKIP_TRAINING, script=f"""
import sys, os, json
sys.path.insert(0, os.getcwd())

# --- Build dataset using rustmentor + local seed data ---
from rustmentor.data import generate_rust_dataset, preprocess_and_merge
from rustmentor.config import SYSTEM_PROMPT

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

# --- Train ---
from rustmentor.training import train_model

train_model(
    base_model="{TRAIN_MODEL}",
    data_path="data/processed/train.jsonl",
    output_dir="{ADAPTER_DIR}",
    lora_r={LORA_R},
    lora_alpha={LORA_ALPHA},
    batch_size={BATCH_SIZE},
    grad_accum={GRAD_ACCUM},
    epochs={EPOCHS},
    lr={LR},
    max_seq_length={MAX_SEQ_LENGTH},
)
""")

    # ══════════════════════════════════════════════
    #  PHASE 1C: Merge adapter (with Gemma3 lm_head untying)
    # ══════════════════════════════════════════════

    run_py("Merge Adapter into Base Model", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from rustmentor.export import merge_adapter

merged = merge_adapter(
    adapter_dir="{ADAPTER_DIR}",
    output_dir="{OUTPUT_DIR}",
    base_model="{FULL_PRECISION_MODEL}",
    untie_lm_head=True,
)
print(f"Merged to: {{merged}}")
""")

    # ══════════════════════════════════════════════
    #  PHASE 2: GGUF conversion via llama.cpp
    # ══════════════════════════════════════════════

    print(f"\n{'=' * 64}")
    print("  Building llama.cpp for GGUF conversion")
    print(f"{'=' * 64}\n")
    run("pip install -q gguf sentencepiece protobuf numpy")
    if not os.path.exists("llama.cpp/build/bin/llama-quantize"):
        run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git llama.cpp")
        run("cmake -B llama.cpp/build -S llama.cpp -DCMAKE_BUILD_TYPE=Release")
        run("cmake --build llama.cpp/build --config Release -j$(nproc) --target llama-quantize")
    else:
        print("  llama.cpp already built")

    run_py("Convert to GGUF", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from rustmentor.export import convert_to_gguf_llamacpp

output = convert_to_gguf_llamacpp(
    merged_dir="{MERGED_DIR}",
    output_dir="{OUTPUT_DIR}",
    quantization="{GGUF_QUANT}",
    model_name="{MODEL_NAME}",
    base_model="{FULL_PRECISION_MODEL}",
    project_root=os.getcwd(),
)
if output:
    print(f"GGUF output: {{output}}")
else:
    print("GGUF conversion failed")
    exit(1)
""")

    # ══════════════════════════════════════════════
    #  PHASE 3: Upload to HuggingFace
    # ══════════════════════════════════════════════

    run_py("Upload to HuggingFace", f"""
import os, glob
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("No HF_TOKEN set, skipping upload")
    exit(0)

api = HfApi(token=token)
username = api.whoami()["name"]
repo_id = f"{{username}}/{MODEL_NAME}-GGUF"

print(f"Uploading to {{repo_id}}")
create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

for f in glob.glob(os.path.join("{OUTPUT_DIR}", "*.gguf")):
    fname = os.path.basename(f)
    size_mb = os.path.getsize(f) / 1024**2
    print(f"  Uploading {{fname}} ({{size_mb:.0f}} MB)")
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=fname,
        repo_id=repo_id,
        token=token,
    )
print(f"https://huggingface.co/{{repo_id}}")
""")

    # ══════════════════════════════════════════════
    #  CLEANUP
    # ══════════════════════════════════════════════

    import shutil
    if os.path.exists(MERGED_DIR):
        print(f"\nCleaning up {MERGED_DIR}...")
        shutil.rmtree(MERGED_DIR)

    total = time.time() - total_start
    print(f"\n{'=' * 64}")
    print(f"  Pipeline complete! ({total:.0f}s / {total / 60:.1f}min)")
    print(f"{'=' * 64}")
    print(f"  Adapter: {ADAPTER_DIR}/")
    print(f"  GGUF:    {OUTPUT_DIR}/")

    if os.path.isdir(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / 1024**2
                print(f"    {f} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
