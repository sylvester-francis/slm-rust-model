#!/usr/bin/env python3
"""
RustMentor SLM — Staged Colab Pipeline

Each stage runs in its own subprocess to isolate dependency conflicts.
No more torch/torchvision/protobuf hell.

Usage in Colab:
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    # Set HF_TOKEN in Colab Secrets, then:
    import os; from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python colab/colab_stages.py
"""

import os
import sys
import subprocess
import time

# ── CONFIG ──────────────────────────────────────────────
VARIANTS = ["1.7b", "0.6b"]       # Which variants to train
EXPORT = "litert"                  # "litert", "gguf", or "both"
LITERT_QUANT = "dynamic_int8"      # dynamic_int8, dynamic_int4, fp16
GGUF_QUANT = "q4_k_m"
# ────────────────────────────────────────────────────────

VARIANT_CONFIGS = {
    "8b": {
        "base_model": "unsloth/Qwen3-8B",
        "lora_r": 32, "lora_alpha": 32,
        "batch_size": 2, "grad_accum": 4,
    },
    "4b": {
        "base_model": "unsloth/Qwen3-4B",
        "lora_r": 16, "lora_alpha": 16,
        "batch_size": 1, "grad_accum": 8,
    },
    "1.7b": {
        "base_model": "unsloth/Qwen3-1.7B",
        "lora_r": 16, "lora_alpha": 16,
        "batch_size": 2, "grad_accum": 4,
    },
    "0.6b": {
        "base_model": "unsloth/Qwen3-0.6B",
        "lora_r": 8, "lora_alpha": 8,
        "batch_size": 4, "grad_accum": 2,
    },
}


def run_stage(name, script):
    """Run a Python script as a subprocess. Exits on failure."""
    print(f"\n{'=' * 60}")
    print(f"  STAGE: {name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    if result.returncode != 0:
        print(f"\n  ❌ STAGE FAILED: {name}")
        sys.exit(1)
    print(f"\n  ✅ STAGE COMPLETE: {name}")


def run_shell(cmd):
    """Run a shell command."""
    subprocess.run(cmd, shell=True, check=True)


def main():
    total_start = time.time()

    print("🦀 RustMentor SLM — Staged Pipeline")
    print(f"   Variants: {VARIANTS}")
    print(f"   Export: {EXPORT}")

    # ── STAGE 0: Install training deps ──
    run_shell("pip install -q unsloth trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer")
    run_shell("pip install -q 'transformers>=4.51.3,<=5.2.0'")

    # ── STAGE 1: Generate + preprocess data ──
    run_stage("Generate & Preprocess Data", """
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

    # ── STAGE 2: Train each variant ──
    for variant in VARIANTS:
        cfg = VARIANT_CONFIGS[variant]
        run_stage(f"Train {variant}", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from scripts.training import train_model

train_model(
    base_model="{cfg['base_model']}",
    data_path="data/processed/train.jsonl",
    output_dir="models/rust-mentor-{variant}",
    lora_r={cfg['lora_r']},
    lora_alpha={cfg['lora_alpha']},
    batch_size={cfg['batch_size']},
    grad_accum={cfg['grad_accum']},
    epochs=3,
    lr=2e-4,
    max_seq_length=2048,
)
""")

    # ── STAGE 3: Merge adapters (still torch 2.10, transformers works) ──
    if EXPORT in ("litert", "both"):
        for variant in VARIANTS:
            run_stage(f"Merge adapter {variant}", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from scripts.convert_litert import merge_adapter

merge_adapter(
    adapter_dir="models/rust-mentor-{variant}",
    output_dir="models/rust-mentor-{variant}-litert",
)
""")

    # ── STAGE 4: Install litert-torch (torch downgrade + TF replacement happens here) ──
    if EXPORT in ("litert", "both"):
        print(f"\n{'=' * 60}")
        print(f"  STAGE: Install litert-torch")
        print(f"{'=' * 60}\n")
        # Colab's tensorflow 2.19 has ABI mismatch with litert-torch's ai-edge-tensorflow.
        # Remove it so litert-torch can install its own compatible version.
        run_shell("pip uninstall -y tensorflow tensorflow-cpu keras -q 2>/dev/null || true")
        run_shell("pip install -q litert-torch 'protobuf>=5.26,<7.0'")
        print("  ✅ litert-torch installed")

    # ── STAGE 5: Convert to LiteRT (torch 2.9 is fine, no transformers needed) ──
    if EXPORT in ("litert", "both"):
        for variant in VARIANTS:
            run_stage(f"Convert {variant} to LiteRT", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from scripts.convert_litert import convert_to_litert

convert_to_litert(
    model_dir="models/rust-mentor-{variant}",
    variant="{variant}",
    quantization="{LITERT_QUANT}",
    kv_cache_max_len=2048,
)
""")

    # ── STAGE 6: Convert to GGUF ──
    if EXPORT in ("gguf", "both"):
        # Reinstall torch 2.10 for Unsloth GGUF export
        run_shell("pip install -q unsloth")
        for variant in VARIANTS:
            run_stage(f"Convert {variant} to GGUF", f"""
import sys, os
sys.path.insert(0, os.getcwd())
from scripts.convert_gguf import convert_to_gguf

convert_to_gguf(
    model_dir="models/rust-mentor-{variant}",
    quantization="{GGUF_QUANT}",
)
""")

    # ── STAGE 7: Upload to HuggingFace ──
    upload_script_parts = []
    for variant in VARIANTS:
        parts = f"""
# Upload adapter
repo_id = f"{{username}}/rust-mentor-{variant}"
print(f"Uploading adapter → {{repo_id}}")
create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
api.upload_folder(folder_path="models/rust-mentor-{variant}", repo_id=repo_id, token=token)
print(f"✅ Adapter: https://huggingface.co/{{repo_id}}")
"""
        if EXPORT in ("litert", "both"):
            parts += f"""
# Upload LiteRT
litert_dir = "models/rust-mentor-{variant}-litert"
if os.path.exists(litert_dir):
    repo_id = f"{{username}}/rust-mentor-{variant}-LiteRT"
    print(f"Uploading LiteRT → {{repo_id}}")
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=litert_dir, repo_id=repo_id, token=token)
    print(f"✅ LiteRT: https://huggingface.co/{{repo_id}}")
"""
        if EXPORT in ("gguf", "both"):
            parts += f"""
# Upload GGUF
gguf_dir = "models/rust-mentor-{variant}-GGUF"
if os.path.exists(gguf_dir):
    repo_id = f"{{username}}/rust-mentor-{variant}-GGUF"
    print(f"Uploading GGUF → {{repo_id}}")
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=gguf_dir, repo_id=repo_id, token=token)
    print(f"✅ GGUF: https://huggingface.co/{{repo_id}}")
"""
        upload_script_parts.append(parts)

    upload_body = "\n".join(upload_script_parts)

    run_stage("Upload to HuggingFace", f"""
import os
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN", "")
if not token:
    print("⏭️  No HF_TOKEN, skipping upload")
    exit(0)

api = HfApi(token=token)
username = api.whoami()["name"]
print(f"HF Username: {{username}}")

{upload_body}
""")

    # ── DONE ──
    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"🦀 Pipeline complete! ({total:.0f}s / {total/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
