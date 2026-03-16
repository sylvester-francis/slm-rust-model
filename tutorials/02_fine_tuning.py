#!/usr/bin/env python3
"""
Tutorial 2: Fine-Tuning with QLoRA
====================================

This tutorial trains a Rust programming tutor using QLoRA (Quantized
Low-Rank Adaptation) with Unsloth for 2x faster training.

Prerequisites:
  - Run Tutorial 1 first (data preparation)
  - CUDA GPU required (T4, L4, A100, etc.)
  - Install: pip install unsloth trl peft accelerate bitsandbytes datasets torch

Recommended variants by hardware:
  0.6B  -> Any GPU (even free Colab T4), ~0.4GB GGUF
  1.7B  -> T4/L4/A100, ~1.1GB GGUF
  4B    -> T4/L4/A100, ~2.5GB GGUF
  8B    -> A100 40GB required, ~4.5GB GGUF

Run:
    python tutorials/02_fine_tuning.py              # Default: 0.6B (fastest)
    python tutorials/02_fine_tuning.py --variant 1.7b  # 1.7B mobile variant

Training time: ~15-60 minutes depending on model size and GPU.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rustmentor.training import train_model
from rustmentor.config import VARIANT_CONFIGS, PROCESSED_DIR, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Tutorial 2: Fine-Tuning")
    parser.add_argument(
        "--variant",
        default="0.6b",
        choices=list(VARIANT_CONFIGS.keys()),
        help="Model variant to train (default: 0.6b)",
    )
    args = parser.parse_args()

    cfg = VARIANT_CONFIGS[args.variant]

    print("=" * 60)
    print("  Tutorial 2: Fine-Tuning with QLoRA")
    print("=" * 60)
    print(f"\n  Variant: {args.variant} ({cfg['description']})")
    print(f"  Base model: {cfg['model']}")
    print(f"  LoRA rank: {cfg['lora_r']}")
    print(f"  Batch size: {cfg['batch_size']} x {cfg['grad_accum']} = {cfg['batch_size'] * cfg['grad_accum']}")

    # Check data exists
    data_path = str(PROCESSED_DIR / "train.jsonl")
    if not os.path.exists(data_path):
        print(f"\n  Error: Training data not found: {data_path}")
        print("  Run Tutorial 1 first: python tutorials/01_data_preparation.py")
        return

    output_dir = str(MODELS_DIR / cfg["output_dir"].split("/")[-1])

    print(f"\n  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print()

    # ── Train ──
    print("--- Training ---\n")
    result = train_model(
        base_model=cfg["model"],
        data_path=data_path,
        output_dir=output_dir,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        batch_size=cfg["batch_size"],
        grad_accum=cfg["grad_accum"],
        epochs=3,
        lr=2e-4,
        max_seq_length=2048,
    )

    if result is None:
        print("\n  Training could not start. Check GPU and dependencies.")
        return

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Tutorial 2 Complete!")
    print("=" * 60)
    print(f"\n  Model saved to: {output_dir}")
    print(f"\n  Next steps:")
    print(f"    python tutorials/03_evaluation.py --variant {args.variant}")
    print(f"    python tutorials/04_export.py --variant {args.variant}")
    print()


if __name__ == "__main__":
    main()
