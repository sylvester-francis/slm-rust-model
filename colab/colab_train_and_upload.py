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

    # Load HF token — must be set as env var before running the script:
    #   In a Colab cell: import os; from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    #   Then:            !python colab/colab_train_and_upload.py
    global HF_USERNAME
    hf_token = os.environ.get("HF_TOKEN", "")

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


MODEL_CARD = """---
license: apache-2.0
language:
- en
tags:
- rust
- programming
- tutor
- code-generation
- qlora
- unsloth
base_model: Qwen/Qwen3-8B
datasets:
- Fortytwo-Network/Strandset-Rust-v1
pipeline_tag: text-generation
---

# RustMentor-8B{suffix}

Fine-tuned Qwen3-8B specialized in **Rust programming education and code review**.

Designed for experienced Go/Python/TypeScript developers learning Rust. {deployment_note}

## Capabilities

- Rust ownership, borrowing, and lifetime explanations
- Error handling patterns (Result, Option, ?)
- Code review with borrow checker explanations
- Async/await and Tokio patterns
- Smart pointers (Box, Rc, Arc, RefCell)
- Pattern matching and enum design
- Trait-based architecture guidance
- Type conversions (From, Into, AsRef, Deref)
- Serde & serialization
- CLI tooling with clap
- Cargo project structure and workspaces
- Comparisons to Go/Python/TypeScript equivalents

## Training Details

- **Base model**: Qwen3-8B
- **Method**: QLoRA (r=32) with Unsloth optimization
- **Dataset**: Strandset-Rust-v1 (3K samples) + 46 unique synthetic Rust tutor conversations across 28 topics
- **Hardware**: A100 40GB (Google Colab)

## Source

- **Repository**: [github.com/sylvester-francis/slm-rust-model](https://github.com/sylvester-francis/slm-rust-model)

## Citation

```bibtex
@software{{rust_mentor_2026,
  author = {{Francis, Sylvester}},
  title = {{RustMentor-8B: Fine-tuned Rust Programming Tutor}},
  year = {{2026}},
  url = {{https://github.com/sylvester-francis/slm-rust-model}}
}}
```
"""


def upload_model_card(repo_id, token, suffix="", deployment_note=""):
    """Upload a model card README.md to a HuggingFace repo."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    card = MODEL_CARD.format(suffix=suffix, deployment_note=deployment_note)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )


def step_upload():
    """Step 4: Upload adapter + push GGUF directly to HuggingFace (no local save)."""
    print("\n" + "─" * 60)
    print("  Step 4/4: Uploading to HuggingFace")
    print("─" * 60)

    if not HF_USERNAME:
        print("  ⏭️  Skipping upload (no HF credentials)")
        return

    from unsloth import FastLanguageModel
    from huggingface_hub import create_repo

    token = os.environ.get("HF_TOKEN", "")

    # Load the trained model
    print("  Loading trained model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="models/rust-mentor-8b",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Upload LoRA adapter (small, ~100MB)
    repo_id = f"{HF_USERNAME}/rust-mentor-8b"
    print(f"  Uploading adapter → {repo_id}")
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    except Exception:
        pass
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)
    upload_model_card(repo_id, token,
        deployment_note="Load with Hugging Face Transformers or deploy via Ollama.")
    print(f"  ✅ Adapter + model card uploaded")

    # Push GGUF directly to HF — no local disk needed!
    gguf_repo = f"{HF_USERNAME}/rust-mentor-8b-GGUF"
    print(f"  Pushing GGUF directly → {gguf_repo} (skips local save)")
    try:
        create_repo(gguf_repo, token=token, exist_ok=True, repo_type="model")
    except Exception:
        pass
    model.push_to_hub_gguf(
        gguf_repo,
        tokenizer,
        quantization_method=GGUF_QUANT,
        token=token,
    )
    upload_model_card(gguf_repo, token,
        suffix=" (GGUF)",
        deployment_note="Runs offline on Android (Pixel 8 Pro tested) via PocketPal AI. Q4_K_M quantization (~4.5GB).")
    print(f"  ✅ GGUF + model card pushed")

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
        ("Upload + GGUF", step_upload),
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
