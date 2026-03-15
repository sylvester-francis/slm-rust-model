#!/usr/bin/env python3
"""
RustMentor — Download from HuggingFace, convert to LiteRT, upload back.

No training. No transformers dependency hell.
Just: download adapter → merge → convert → upload.

Each step runs in its own subprocess for clean isolation.

Usage in Colab:
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    import os; from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python colab/colab_litert_convert.py
"""

import os
import sys
import subprocess
import time

# ── CONFIG ──────────────────────────────────────────────
HF_USERNAME = "sylvester-francis"
VARIANTS = ["0.6b", "1.7b"]
LITERT_QUANT = "dynamic_int8"     # dynamic_int8, dynamic_int4, fp16
KV_CACHE_LEN = 2048
# ────────────────────────────────────────────────────────

# Qwen3 full-precision base models (for merging)
BASE_MODELS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
}


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
    print("🦀 RustMentor — LiteRT Conversion Pipeline")
    print(f"   Variants: {VARIANTS}")
    print(f"   Source: HuggingFace ({HF_USERNAME})")
    print()

    # ── STEP 1: Install merge deps (peft + transformers, needs torch 2.10) ──
    print("📦 Installing merge dependencies...")
    run("pip install -q peft transformers accelerate safetensors huggingface_hub")

    # ── STEP 2: Download adapters from HF + merge into full-precision models ──
    for variant in VARIANTS:
        adapter_repo = f"{HF_USERNAME}/rust-mentor-{variant}"
        adapter_dir = f"models/rust-mentor-{variant}"
        merged_dir = f"models/rust-mentor-{variant}-litert/merged"
        base_model = BASE_MODELS[variant]

        run_py(f"Download & merge {variant}", f"""
import os, torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_dir = "{adapter_dir}"
merged_dir = "{merged_dir}"

# Skip if already merged
if os.path.exists(os.path.join(merged_dir, "config.json")):
    print(f"  Already merged: {{merged_dir}}")
    exit(0)

# Download adapter from HuggingFace
print(f"  Downloading {adapter_repo}...")
snapshot_download(
    repo_id="{adapter_repo}",
    local_dir=adapter_dir,
    token=os.environ.get("HF_TOKEN", ""),
)

# Load base model in full precision
print(f"  Loading base model: {base_model} (fp16)...")
model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype=torch.float16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

# Load and merge adapter
print(f"  Merging adapter...")
model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()

# Save
os.makedirs(merged_dir, exist_ok=True)
print(f"  Saving merged model → {{merged_dir}}")
model.save_pretrained(merged_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_dir)
print(f"  ✅ Merged")
""")

    # ── STEP 3: Install litert-torch (replaces torch + tensorflow) ──
    print(f"\n{'=' * 60}")
    print(f"  Install litert-torch")
    print(f"{'=' * 60}\n")
    run("pip uninstall -y tensorflow tensorflow-cpu keras -q 2>/dev/null || true")
    run("pip install -q litert-torch 'protobuf>=5.26,<7.0'")

    # ── STEP 4: Convert to LiteRT using Google's official CLI ──
    for variant in VARIANTS:
        merged_dir = f"models/rust-mentor-{variant}-litert/merged"
        output_dir = f"models/rust-mentor-{variant}-litert"
        prefix = f"rust_mentor_{variant.replace('.', '_')}"

        print(f"\n{'=' * 60}")
        print(f"  Convert {variant} to LiteRT")
        print(f"{'=' * 60}\n")

        run(
            f"python -m litert_torch.generative.examples.qwen.convert_v3_to_tflite"
            f" --model_size={variant}"
            f" --checkpoint_path={merged_dir}"
            f" --output_path={output_dir}"
            f" --output_name_prefix={prefix}"
            f" --quantize={LITERT_QUANT}"
            f" --kv_cache_max_len={KV_CACHE_LEN}"
        )

    # ── STEP 5: Upload LiteRT to HuggingFace ──
    upload_parts = []
    for variant in VARIANTS:
        upload_parts.append(f"""
litert_dir = "models/rust-mentor-{variant}-litert"
repo_id = f"{{username}}/rust-mentor-{variant}-LiteRT"
print(f"\\nUploading {{litert_dir}} → {{repo_id}}")
create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
# Upload only .tflite files and tokenizer, skip the huge merged/ dir
import glob
for f in glob.glob(os.path.join(litert_dir, "*.tflite")):
    print(f"  Uploading {{os.path.basename(f)}}")
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=repo_id, token=token)
print(f"✅ https://huggingface.co/{{repo_id}}")
""")

    run_py("Upload to HuggingFace", f"""
import os, glob
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("No HF_TOKEN, skipping upload")
    exit(0)

api = HfApi(token=token)
username = api.whoami()["name"]
print(f"HF Username: {{username}}")
{"".join(upload_parts)}
""")

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"🦀 Done! ({total:.0f}s / {total/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
