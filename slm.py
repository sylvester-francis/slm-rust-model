#!/usr/bin/env python3
"""
RustMentor SLM - Unified CLI for Rust Language Model Fine-Tuning

Fine-tuned small language models specialized in Rust programming education
and code generation. Built on Qwen3-8B with Unsloth QLoRA optimization.

Usage:
    python slm.py pipeline          # Complete training pipeline
    python slm.py collect           # Generate synthetic training data
    python slm.py train             # Train model with QLoRA
    python slm.py evaluate          # Evaluate model quality
    python slm.py convert           # Export to GGUF for mobile
    python slm.py upload            # Upload to HuggingFace
    python slm.py deploy            # Deploy to Ollama
    python slm.py info              # Show system information
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B"
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4
DEFAULT_SEQ_LENGTH = 2048
DEFAULT_GGUF_QUANT = "q4_k_m"

SYSTEM_PROMPT = """You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools.

Your teaching style:
- Draw parallels to Go/Python/TypeScript concepts they already know
- Explain ownership, borrowing, and lifetimes with practical examples
- When reviewing code, explain what the borrow checker is doing and why
- Keep explanations concise with code snippets
- Guide them to write the code themselves rather than giving full solutions
- Use the Socratic method when appropriate
"""


def cmd_info(args):
    """Show system information."""
    print("\n🦀 RustMentor SLM - System Information")
    print("=" * 50)

    # Python
    print(f"\nPython: {sys.version.split()[0]}")

    # PyTorch & CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"VRAM: {mem:.1f} GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Device: Apple Silicon (MPS)")
        else:
            print("Device: CPU only")
    except ImportError:
        print("PyTorch: Not installed")

    # Key packages
    for pkg in ["transformers", "peft", "trl", "unsloth", "datasets", "accelerate"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"{pkg}: {ver}")
        except ImportError:
            print(f"{pkg}: Not installed")

    # Project files
    print(f"\nProject root: {PROJECT_ROOT}")
    datasets = list(PROCESSED_DIR.glob("*.jsonl")) if PROCESSED_DIR.exists() else []
    print(f"Datasets: {len(datasets)} files")
    for ds in datasets:
        lines = sum(1 for _ in open(ds))
        print(f"  {ds.name}: {lines:,} samples")

    models = list(MODELS_DIR.glob("*")) if MODELS_DIR.exists() else []
    print(f"Models: {len(models)} saved")
    print()


def cmd_collect(args):
    """Generate synthetic Rust tutor training data."""
    from scripts.data_collection import generate_rust_dataset

    output = args.output or str(PROCESSED_DIR / "rust_tutor_synthetic.jsonl")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"\n🦀 Generating Rust tutor dataset...")
    print(f"   Samples: {args.samples}")
    print(f"   Output: {output}")

    count = generate_rust_dataset(
        output_path=output,
        num_samples=args.samples,
        system_prompt=SYSTEM_PROMPT,
    )
    print(f"\n✅ Generated {count} training samples → {output}")


def cmd_preprocess(args):
    """Preprocess and merge datasets."""
    from scripts.data_preprocessing import preprocess_and_merge

    print(f"\n🦀 Preprocessing datasets...")

    output = args.output or str(PROCESSED_DIR / "train.jsonl")
    count = preprocess_and_merge(
        synthetic_path=str(PROCESSED_DIR / "rust_tutor_synthetic.jsonl"),
        strandset_samples=args.strandset_samples,
        output_path=output,
        max_seq_length=args.max_length,
    )
    print(f"\n✅ Merged dataset: {count} samples → {output}")


def cmd_train(args):
    """Train model with QLoRA."""
    from scripts.training import train_model

    data_path = args.data or str(PROCESSED_DIR / "train.jsonl")
    output_dir = args.output or str(MODELS_DIR / "rust-mentor-8b")

    print(f"\n🦀 Training RustMentor model")
    print(f"   Base model: {args.model}")
    print(f"   Dataset: {data_path}")
    print(f"   LoRA rank: {args.lora_r}")
    print(f"   Batch size: {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print()

    train_model(
        base_model=args.model,
        data_path=data_path,
        output_dir=output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_length=args.max_length,
    )
    print(f"\n✅ Model saved → {output_dir}")


def cmd_evaluate(args):
    """Evaluate model quality."""
    from scripts.evaluation import evaluate_model

    model_dir = args.model_dir or str(MODELS_DIR / "rust-mentor-8b")
    print(f"\n🦀 Evaluating model: {model_dir}")

    results = evaluate_model(model_dir=model_dir)

    print("\n📊 Evaluation Results:")
    print("-" * 40)
    for metric, score in results.items():
        print(f"  {metric}: {score}")
    print()


def cmd_convert(args):
    """Convert model to GGUF format."""
    from scripts.convert_gguf import convert_to_gguf

    model_dir = args.model_dir or str(MODELS_DIR / "rust-mentor-8b")
    quant = args.quant or DEFAULT_GGUF_QUANT

    print(f"\n🦀 Converting to GGUF ({quant})")
    print(f"   Model: {model_dir}")

    output = convert_to_gguf(
        model_dir=model_dir,
        quantization=quant,
    )
    print(f"\n✅ GGUF exported → {output}")


def cmd_upload(args):
    """Upload model to HuggingFace Hub."""
    from scripts.upload_to_hf import upload_model

    model_dir = args.model_dir or str(MODELS_DIR / "rust-mentor-8b")
    repo = f"{args.username}/rust-mentor-8b"

    if args.gguf:
        repo += "-GGUF"

    print(f"\n🦀 Uploading to HuggingFace: {repo}")
    upload_model(
        model_dir=model_dir,
        repo_id=repo,
        gguf=args.gguf,
    )
    print(f"\n✅ Uploaded → https://huggingface.co/{repo}")


def cmd_deploy(args):
    """Deploy model to Ollama."""
    from scripts.deploy_ollama import deploy_to_ollama

    model_name = args.name or "rust-mentor-8b"
    print(f"\n🦀 Deploying to Ollama: {model_name}")

    deploy_to_ollama(model_name=model_name)
    print(f"\n✅ Model available: ollama run {model_name}")


def cmd_pipeline(args):
    """Run complete pipeline: collect → preprocess → train → evaluate → convert → upload."""
    print("\n🦀 RustMentor SLM - Full Pipeline")
    print("=" * 50)

    steps = [
        ("1/6", "Generating synthetic data", lambda: cmd_collect(args)),
        ("2/6", "Preprocessing & merging", lambda: cmd_preprocess(args)),
        ("3/6", "Training with QLoRA", lambda: cmd_train(args)),
        ("4/6", "Evaluating model", lambda: cmd_evaluate(args)),
        ("5/6", "Converting to GGUF", lambda: cmd_convert(args)),
    ]

    if args.username:
        steps.append(("6/6", "Uploading to HuggingFace", lambda: cmd_upload(args)))

    for step_num, desc, func in steps:
        print(f"\n{'─' * 50}")
        print(f"  Step {step_num}: {desc}")
        print(f"{'─' * 50}")
        start = time.time()
        func()
        elapsed = time.time() - start
        print(f"  ⏱️  {elapsed:.1f}s")

    print("\n" + "=" * 50)
    print("✅ Pipeline complete!")
    print("=" * 50)


# ──────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="🦀 RustMentor SLM - Fine-tuned Rust Programming Tutor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slm.py info                          # System information
  python slm.py pipeline --username myuser    # Full pipeline
  python slm.py train --epochs 5 --lora-r 64  # Custom training
  python slm.py convert --quant q5_k_m        # Higher quality GGUF
  python slm.py upload --username myuser --gguf  # Upload GGUF
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # info
    subparsers.add_parser("info", help="Show system information")

    # collect
    p = subparsers.add_parser("collect", help="Generate synthetic training data")
    p.add_argument("--samples", type=int, default=500, help="Number of samples")
    p.add_argument("--output", type=str, help="Output JSONL path")

    # preprocess
    p = subparsers.add_parser("preprocess", help="Preprocess and merge datasets")
    p.add_argument("--strandset-samples", type=int, default=3000, help="Samples from Strandset")
    p.add_argument("--output", type=str, help="Output JSONL path")
    p.add_argument("--max-length", type=int, default=DEFAULT_SEQ_LENGTH)

    # train
    p = subparsers.add_parser("train", help="Train model with QLoRA")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base model")
    p.add_argument("--data", type=str, help="Training data JSONL")
    p.add_argument("--output", type=str, help="Output directory")
    p.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--max-length", type=int, default=DEFAULT_SEQ_LENGTH)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate trained model")
    p.add_argument("--model-dir", type=str, help="Model directory")

    # convert
    p = subparsers.add_parser("convert", help="Convert to GGUF format")
    p.add_argument("--model-dir", type=str, help="Model directory")
    p.add_argument("--quant", default=DEFAULT_GGUF_QUANT, help="Quantization type")

    # upload
    p = subparsers.add_parser("upload", help="Upload to HuggingFace")
    p.add_argument("--username", required=True, help="HF username")
    p.add_argument("--model-dir", type=str, help="Model directory")
    p.add_argument("--gguf", action="store_true", help="Upload GGUF version")

    # deploy
    p = subparsers.add_parser("deploy", help="Deploy to Ollama")
    p.add_argument("--name", type=str, help="Ollama model name")

    # pipeline
    p = subparsers.add_parser("pipeline", help="Run complete pipeline")
    p.add_argument("--username", type=str, help="HF username (enables upload)")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--strandset-samples", type=int, default=3000)
    p.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--max-length", type=int, default=DEFAULT_SEQ_LENGTH)
    p.add_argument("--quant", default=DEFAULT_GGUF_QUANT)
    p.add_argument("--model-dir", type=str)
    p.add_argument("--data", type=str)
    p.add_argument("--output", type=str)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "info": cmd_info,
        "collect": cmd_collect,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "convert": cmd_convert,
        "upload": cmd_upload,
        "deploy": cmd_deploy,
        "pipeline": cmd_pipeline,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
