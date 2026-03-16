#!/usr/bin/env python3
"""
RustMentor SLM - Train Your Own Rust Programming Tutor

A step-by-step pipeline for fine-tuning small language models specialized
in Rust programming education. Built on Qwen3/Gemma3 with QLoRA via Unsloth.

Steps:
    1. python slm.py collect          # Generate synthetic training data
    2. python slm.py preprocess       # Merge & format datasets
    3. python slm.py train            # Fine-tune with QLoRA
    4. python slm.py evaluate         # Test model quality
    5. python slm.py convert          # Export to GGUF (mobile/edge)
    6. python slm.py convert-litert   # Export to LiteRT (Android GPU/NPU)
    7. python slm.py upload           # Upload to HuggingFace
    8. python slm.py deploy           # Deploy to Ollama

Quick start:
    python slm.py pipeline --variant 0.6b --username your-hf-user
"""

import argparse
import sys
import os
import time

from rustmentor.config import (
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    VARIANT_CONFIGS,
    DEFAULT_BASE_MODEL,
    DEFAULT_LORA_R,
    DEFAULT_LORA_ALPHA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_SEQ_LENGTH,
    DEFAULT_GGUF_QUANT,
)


def _apply_variant(args):
    """If --variant is set, override args with preset values (unless explicitly provided)."""
    variant = getattr(args, "variant", None)
    if not variant:
        return
    cfg = VARIANT_CONFIGS.get(variant)
    if not cfg:
        print(f"  Unknown variant '{variant}'. Available: {', '.join(VARIANT_CONFIGS)}")
        sys.exit(1)
    if args.model == DEFAULT_BASE_MODEL:
        args.model = cfg["model"]
    if args.lora_r == DEFAULT_LORA_R:
        args.lora_r = cfg["lora_r"]
    if args.lora_alpha == DEFAULT_LORA_ALPHA:
        args.lora_alpha = cfg["lora_alpha"]
    if args.batch_size == DEFAULT_BATCH_SIZE:
        args.batch_size = cfg["batch_size"]
    if args.grad_accum == DEFAULT_GRAD_ACCUM:
        args.grad_accum = cfg["grad_accum"]
    if not args.output:
        args.output = str(MODELS_DIR / cfg["output_dir"].split("/")[-1])
    if not getattr(args, "model_dir", None):
        args.model_dir = str(MODELS_DIR / cfg["output_dir"].split("/")[-1])
    if not getattr(args, "name", None):
        args.name = cfg["deploy_name"]


def cmd_info(args):
    """Show system information."""
    print("\n  RustMentor SLM - System Information")
    print("=" * 50)

    print(f"\nPython: {sys.version.split()[0]}")

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

    for pkg in ["transformers", "peft", "trl", "unsloth", "datasets", "accelerate"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"{pkg}: {ver}")
        except ImportError:
            print(f"{pkg}: Not installed")

    print(f"\nProject root: {PROJECT_ROOT}")
    datasets = list(PROCESSED_DIR.glob("*.jsonl")) if PROCESSED_DIR.exists() else []
    print(f"Datasets: {len(datasets)} files")
    for ds in datasets:
        lines = sum(1 for _ in open(ds))
        print(f"  {ds.name}: {lines:,} samples")

    models = list(MODELS_DIR.glob("*")) if MODELS_DIR.exists() else []
    print(f"Models: {len(models)} saved")

    print("\nAvailable variants:")
    for name, cfg in VARIANT_CONFIGS.items():
        print(f"  {name:12s} {cfg['description']}")
    print()


def cmd_collect(args):
    """Step 1: Generate synthetic Rust tutor training data."""
    from rustmentor.data import generate_rust_dataset

    output = args.output or str(PROCESSED_DIR / "rust_tutor_synthetic.jsonl")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"\n  Step 1: Generating Rust tutor dataset...")
    print(f"    Samples: {args.samples}")
    print(f"    Output: {output}")

    count = generate_rust_dataset(output_path=output, num_samples=args.samples)
    print(f"\n  Generated {count} training samples -> {output}")


def cmd_preprocess(args):
    """Step 2: Preprocess and merge datasets."""
    from rustmentor.data import preprocess_and_merge

    print(f"\n  Step 2: Preprocessing datasets...")

    output = args.output or str(PROCESSED_DIR / "train.jsonl")
    count = preprocess_and_merge(
        synthetic_path=str(PROCESSED_DIR / "rust_tutor_synthetic.jsonl"),
        strandset_samples=args.strandset_samples,
        output_path=output,
        max_seq_length=args.max_length,
    )
    print(f"\n  Merged dataset: {count} samples -> {output}")


def cmd_train(args):
    """Step 3: Train model with QLoRA."""
    from rustmentor.training import train_model

    _apply_variant(args)
    data_path = args.data or str(PROCESSED_DIR / "train.jsonl")
    output_dir = args.output or str(MODELS_DIR / "rust-mentor-8b")

    print(f"\n  Step 3: Training RustMentor model")
    print(f"    Base model: {args.model}")
    print(f"    Dataset: {data_path}")
    print(f"    LoRA rank: {args.lora_r}")
    print(f"    Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"    Epochs: {args.epochs}")
    print(f"    Learning rate: {args.lr}")
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
    print(f"\n  Model saved -> {output_dir}")


def cmd_evaluate(args):
    """Step 4: Evaluate model quality."""
    from rustmentor.training import evaluate_model

    _apply_variant(args)
    model_dir = args.model_dir or str(MODELS_DIR / "rust-mentor-8b")
    print(f"\n  Step 4: Evaluating model: {model_dir}")

    results = evaluate_model(model_dir=model_dir)

    if results:
        print("\n  Evaluation Results:")
        print("  " + "-" * 40)
        for metric, score in results.items():
            if metric != "responses":
                print(f"    {metric}: {score}")
    print()


def cmd_convert(args):
    """Step 5a: Convert model to GGUF format."""
    from rustmentor.export import convert_to_gguf

    _apply_variant(args)
    model_dir = args.model_dir or str(MODELS_DIR / "rust-mentor-8b")
    quant = args.quant or DEFAULT_GGUF_QUANT

    print(f"\n  Step 5a: Converting to GGUF ({quant})")
    print(f"    Model: {model_dir}")

    output = convert_to_gguf(model_dir=model_dir, quantization=quant)
    if output:
        print(f"\n  GGUF exported -> {output}")


def cmd_convert_litert(args):
    """Step 5b: Convert model to LiteRT (.tflite) for Android."""
    from rustmentor.export import convert_to_litert

    _apply_variant(args)
    model_dir = args.model_dir or str(MODELS_DIR / "rust-mentor-8b")
    quant = getattr(args, "litert_quant", "dynamic_int8") or "dynamic_int8"
    variant = getattr(args, "variant", None)

    print(f"\n  Step 5b: Converting to LiteRT ({quant})")
    print(f"    Model: {model_dir}")

    output = convert_to_litert(
        model_dir=model_dir,
        variant=variant,
        quantization=quant,
        kv_cache_max_len=getattr(args, "max_length", DEFAULT_SEQ_LENGTH) or DEFAULT_SEQ_LENGTH,
    )
    if output:
        print(f"\n  LiteRT exported -> {output}")
    else:
        print(f"\n  LiteRT conversion failed")


def cmd_upload(args):
    """Step 6: Upload model to HuggingFace Hub."""
    from rustmentor.deploy import upload_model

    _apply_variant(args)
    model_dir = args.model_dir or str(MODELS_DIR / "rust-mentor-8b")
    variant = getattr(args, "variant", None) or "8b"
    repo = f"{args.username}/rust-mentor-{variant}"

    if args.litert:
        repo += "-LiteRT"
        litert_dir = model_dir + "-litert"
        if os.path.exists(litert_dir):
            model_dir = litert_dir
        else:
            print(f"  LiteRT directory not found: {litert_dir}")
            print(f"  Run `python slm.py convert-litert --variant {variant}` first.")
            return
    elif args.gguf:
        repo += "-GGUF"

    print(f"\n  Step 6: Uploading to HuggingFace: {repo}")
    upload_model(model_dir=model_dir, repo_id=repo, gguf=args.gguf)
    print(f"\n  Uploaded -> https://huggingface.co/{repo}")


def cmd_deploy(args):
    """Step 7: Deploy model to Ollama."""
    from rustmentor.deploy import deploy_to_ollama

    _apply_variant(args)
    model_name = args.name or "rust-mentor-8b"
    print(f"\n  Step 7: Deploying to Ollama: {model_name}")

    deploy_to_ollama(model_name=model_name)
    print(f"\n  Model available: ollama run {model_name}")


def cmd_pipeline(args):
    """Run complete pipeline: collect -> preprocess -> train -> evaluate -> convert -> upload."""
    _apply_variant(args)
    print("\n  RustMentor SLM - Full Pipeline")
    print("=" * 50)

    steps = [
        ("1/7", "Generating synthetic data", lambda: cmd_collect(args)),
        ("2/7", "Preprocessing & merging", lambda: cmd_preprocess(args)),
        ("3/7", "Training with QLoRA", lambda: cmd_train(args)),
        ("4/7", "Evaluating model", lambda: cmd_evaluate(args)),
        ("5/7", "Converting to GGUF", lambda: cmd_convert(args)),
        ("6/7", "Converting to LiteRT", lambda: cmd_convert_litert(args)),
    ]

    if args.username:
        steps.append(("7/7", "Uploading to HuggingFace", lambda: cmd_upload(args)))

    for step_num, desc, func in steps:
        print(f"\n{'─' * 50}")
        print(f"  Step {step_num}: {desc}")
        print(f"{'─' * 50}")
        start = time.time()
        func()
        elapsed = time.time() - start
        print(f"  {elapsed:.1f}s")

    print("\n" + "=" * 50)
    print("  Pipeline complete!")
    print("=" * 50)


# ── Argument Parser ────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="RustMentor SLM - Fine-tuned Rust Programming Tutor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slm.py info                                  # System information
  python slm.py pipeline --variant 0.6b --username u  # Full pipeline for 0.6B
  python slm.py train --variant 1.7b                  # Train 1.7B mobile variant
  python slm.py train --variant 0.6b                  # Train ultra-light 0.6B
  python slm.py train --epochs 5 --lora-r 64          # Custom training
  python slm.py convert --variant 0.6b                # Export 0.6B to GGUF
  python slm.py convert-litert --variant 0.6b         # Export 0.6B to LiteRT
  python slm.py upload --variant 1.7b --username u --gguf  # Upload GGUF

Available variants:
  8b          Qwen3-8B   — highest quality, A100 required
  4b          Qwen3-4B   — lighter, T4 compatible
  1.7b        Qwen3-1.7B — fast on-device chat
  0.6b        Qwen3-0.6B — ultra-light, any GPU
  gemma3-1b   Gemma3-1B  — Google AI Edge Gallery (.litertlm)
  gemma3-4b   Gemma3-4B  — PocketPal AI (GGUF)
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # info
    subparsers.add_parser("info", help="Show system information")

    # collect
    p = subparsers.add_parser("collect", help="Step 1: Generate synthetic training data")
    p.add_argument("--samples", type=int, default=500, help="Number of samples")
    p.add_argument("--output", type=str, help="Output JSONL path")

    # preprocess
    p = subparsers.add_parser("preprocess", help="Step 2: Preprocess and merge datasets")
    p.add_argument("--strandset-samples", type=int, default=3000, help="Samples from Strandset")
    p.add_argument("--output", type=str, help="Output JSONL path")
    p.add_argument("--max-length", type=int, default=DEFAULT_SEQ_LENGTH)

    # train
    p = subparsers.add_parser("train", help="Step 3: Train model with QLoRA")
    p.add_argument("--variant", choices=list(VARIANT_CONFIGS), help="Model variant preset")
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
    p = subparsers.add_parser("evaluate", help="Step 4: Evaluate trained model")
    p.add_argument("--variant", choices=list(VARIANT_CONFIGS), help="Model variant preset")
    p.add_argument("--model-dir", type=str, help="Model directory")

    # convert
    p = subparsers.add_parser("convert", help="Step 5a: Convert to GGUF format")
    p.add_argument("--variant", choices=list(VARIANT_CONFIGS), help="Model variant preset")
    p.add_argument("--model-dir", type=str, help="Model directory")
    p.add_argument("--quant", default=DEFAULT_GGUF_QUANT, help="Quantization type")

    # convert-litert
    p = subparsers.add_parser("convert-litert", help="Step 5b: Convert to LiteRT (.tflite)")
    p.add_argument("--variant", choices=list(VARIANT_CONFIGS), help="Model variant preset")
    p.add_argument("--model-dir", type=str, help="Model directory")
    p.add_argument("--litert-quant", default="dynamic_int8",
                    choices=["dynamic_int8", "dynamic_int4", "fp16", "none"],
                    help="LiteRT quantization (default: dynamic_int8)")
    p.add_argument("--max-length", type=int, default=DEFAULT_SEQ_LENGTH, help="KV cache length")

    # upload
    p = subparsers.add_parser("upload", help="Step 6: Upload to HuggingFace")
    p.add_argument("--variant", choices=list(VARIANT_CONFIGS), help="Model variant preset")
    p.add_argument("--username", required=True, help="HF username")
    p.add_argument("--model-dir", type=str, help="Model directory")
    p.add_argument("--gguf", action="store_true", help="Upload GGUF version")
    p.add_argument("--litert", action="store_true", help="Upload LiteRT version")

    # deploy
    p = subparsers.add_parser("deploy", help="Step 7: Deploy to Ollama")
    p.add_argument("--variant", choices=list(VARIANT_CONFIGS), help="Model variant preset")
    p.add_argument("--name", type=str, help="Ollama model name")

    # pipeline
    p = subparsers.add_parser("pipeline", help="Run complete pipeline (Steps 1-7)")
    p.add_argument("--variant", choices=list(VARIANT_CONFIGS), help="Model variant preset")
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
        "convert-litert": cmd_convert_litert,
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
