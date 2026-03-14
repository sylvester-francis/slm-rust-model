#!/usr/bin/env python3
"""
RustMentor SLM - Google Colab Training Pipeline

Automated pipeline: Generate data → Train → Evaluate → Export GGUF → Upload to HF

Requirements:
- Google Colab Pro with A100 GPU (40GB VRAM)
- HF_TOKEN in Colab Secrets (🔑 icon in left sidebar)

Usage in Colab:
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    !python colab/colab_train_and_upload.py
"""

import os
import sys
import time
import json

# ──────────────────────────────────────────────
# CONFIGURATION — Edit these to customize
# ──────────────────────────────────────────────

# Model
BASE_MODEL = "unsloth/Qwen3-8B"
MAX_SEQ_LENGTH = 2048
MODEL_VARIANT = "standard"  # "standard" or "reasoning"

# LoRA
LORA_R = 32
LORA_ALPHA = 32

# Training
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4

# Dataset
SYNTHETIC_SAMPLES = 500
STRANDSET_SAMPLES = 3000

# Export
GGUF_QUANT = "q4_k_m"  # q4_k_m for Pixel 8 Pro, q5_k_m for higher quality

# HuggingFace
HF_USERNAME = ""  # Set below from secrets or manually

# ──────────────────────────────────────────────


def setup_environment():
    """Install dependencies and verify GPU."""
    print("\n" + "=" * 60)
    print("🦀 RustMentor SLM — Colab Training Pipeline")
    print("=" * 60)

    # Install deps
    print("\n📦 Installing dependencies...")
    os.system("pip install -q unsloth trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer")
    os.system("pip install -q 'transformers>=4.51.3,<=5.2.0'")

    # Verify GPU
    import torch
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Go to Runtime → Change runtime type → A100 GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU: {gpu_name} ({vram:.1f} GB)")

    if vram < 35:
        print("⚠️  A100 (40GB) recommended for 8B model. T4 may OOM.")
        print("   Consider using Qwen3-4B instead (set BASE_MODEL above).")

    # Load HF token
    global HF_USERNAME
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        try:
            from google.colab import userdata
            hf_token = userdata.get("HF_TOKEN")
            os.environ["HF_TOKEN"] = hf_token
        except Exception:
            pass

    if hf_token:
        print("✅ HF_TOKEN loaded")
        # Get username from token
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            HF_USERNAME = api.whoami()["name"]
            print(f"✅ HF Username: {HF_USERNAME}")
        except Exception:
            print("⚠️  Could not determine HF username. Set HF_USERNAME manually.")
    else:
        print("⚠️  HF_TOKEN not found. Upload step will be skipped.")
        print("   Add it in Colab: 🔑 Secrets → HF_TOKEN → your token")


def step_generate_data():
    """Step 1: Generate synthetic Rust tutor data."""
    print("\n" + "─" * 60)
    print("  Step 1/5: Generating Synthetic Rust Tutor Data")
    print("─" * 60)

    # Add project root to path
    sys.path.insert(0, os.getcwd())
    from scripts.data_collection import generate_rust_dataset

    os.makedirs("data/processed", exist_ok=True)
    count = generate_rust_dataset(
        output_path="data/processed/rust_tutor_synthetic.jsonl",
        num_samples=SYNTHETIC_SAMPLES,
        system_prompt="""You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools.

Your teaching style:
- Draw parallels to Go/Python/TypeScript concepts they already know
- Explain ownership, borrowing, and lifetimes with practical examples
- When reviewing code, explain what the borrow checker is doing and why
- Keep explanations concise with code snippets
- Guide them to write the code themselves rather than giving full solutions
""",
    )
    print(f"  ✅ Generated {count} synthetic samples")


def step_preprocess():
    """Step 2: Preprocess and merge datasets."""
    print("\n" + "─" * 60)
    print("  Step 2/5: Preprocessing & Merging Datasets")
    print("─" * 60)

    from scripts.data_preprocessing import preprocess_and_merge

    count = preprocess_and_merge(
        synthetic_path="data/processed/rust_tutor_synthetic.jsonl",
        strandset_samples=STRANDSET_SAMPLES,
        output_path="data/processed/train.jsonl",
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print(f"  ✅ Merged dataset: {count} samples")


def step_train():
    """Step 3: Fine-tune with QLoRA."""
    print("\n" + "─" * 60)
    print("  Step 3/5: Training with QLoRA + Unsloth")
    print("─" * 60)

    from scripts.training import train_model

    stats = train_model(
        base_model=BASE_MODEL,
        data_path="data/processed/train.jsonl",
        output_dir="models/rust-mentor-8b",
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print(f"  ✅ Training complete (loss: {stats.metrics['train_loss']:.4f})")


def step_convert():
    """Step 4: Export to GGUF."""
    print("\n" + "─" * 60)
    print("  Step 4/5: Converting to GGUF")
    print("─" * 60)

    from scripts.convert_gguf import convert_to_gguf

    output = convert_to_gguf(
        model_dir="models/rust-mentor-8b",
        quantization=GGUF_QUANT,
    )
    print(f"  ✅ GGUF exported: {output}")


def step_upload():
    """Step 5: Upload to HuggingFace."""
    print("\n" + "─" * 60)
    print("  Step 5/5: Uploading to HuggingFace")
    print("─" * 60)

    if not HF_USERNAME:
        print("  ⏭️  Skipping upload (no HF credentials)")
        return

    from scripts.upload_to_hf import upload_model

    # Upload LoRA adapter
    repo_id = f"{HF_USERNAME}/rust-mentor-8b"
    print(f"  Uploading adapter → {repo_id}")
    upload_model(
        model_dir="models/rust-mentor-8b",
        repo_id=repo_id,
    )

    # Upload GGUF
    gguf_repo = f"{HF_USERNAME}/rust-mentor-8b-GGUF"
    print(f"  Uploading GGUF → {gguf_repo}")
    upload_model(
        model_dir="models/rust-mentor-8b",
        repo_id=gguf_repo,
        gguf=True,
    )

    print(f"\n  🔗 Model: https://huggingface.co/{repo_id}")
    print(f"  🔗 GGUF:  https://huggingface.co/{gguf_repo}")


def main():
    """Run complete pipeline."""
    total_start = time.time()

    setup_environment()

    steps = [
        ("Generate data", step_generate_data),
        ("Preprocess", step_preprocess),
        ("Train", step_train),
        ("Convert GGUF", step_convert),
        ("Upload", step_upload),
    ]

    timings = {}
    for name, func in steps:
        start = time.time()
        try:
            func()
            timings[name] = time.time() - start
        except Exception as e:
            print(f"\n  ❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            timings[name] = f"FAILED: {e}"

    # Summary
    total = time.time() - total_start
    print("\n" + "=" * 60)
    print("🦀 Pipeline Complete!")
    print("=" * 60)
    for name, duration in timings.items():
        if isinstance(duration, float):
            print(f"  {name}: {duration:.1f}s")
        else:
            print(f"  {name}: {duration}")
    print(f"\n  Total: {total:.1f}s ({total/60:.1f} min)")

    if HF_USERNAME:
        print(f"\n  📱 Next: Download the GGUF from HuggingFace to your Pixel 8 Pro")
        print(f"     → PocketPal AI → Add from Hugging Face")
        print(f"     → Search: {HF_USERNAME}/rust-mentor-8b-GGUF")

    print()


if __name__ == "__main__":
    main()
