#!/usr/bin/env python3
"""
Tutorial 1: Data Preparation
=============================

This tutorial walks through generating and preparing training data
for a Rust programming tutor model.

Two data sources are combined:
  1. Synthetic Q&A: 46 hand-written Rust tutoring conversations covering
     ownership, borrowing, lifetimes, error handling, traits, async, etc.
  2. Strandset-Rust-v1: ~3000 Rust code task examples from HuggingFace.

The merged dataset is saved as JSONL for training in Tutorial 2.

Run:
    python tutorials/01_data_preparation.py

No GPU required. Takes ~1-2 minutes (mostly downloading Strandset).
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rustmentor.data import generate_rust_dataset, preprocess_and_merge
from rustmentor.config import PROCESSED_DIR


def main():
    print("=" * 60)
    print("  Tutorial 1: Data Preparation")
    print("=" * 60)

    # ── Step 1: Generate synthetic Rust tutor conversations ──
    print("\n--- Step 1: Generating synthetic training data ---\n")
    print("  We have 46 hand-written Rust tutoring conversations covering")
    print("  28 topics (ownership, borrowing, traits, async, etc.).")
    print("  These are duplicated to reach the target sample count.\n")

    synthetic_path = str(PROCESSED_DIR / "rust_tutor_synthetic.jsonl")
    os.makedirs(str(PROCESSED_DIR), exist_ok=True)

    count = generate_rust_dataset(
        output_path=synthetic_path,
        num_samples=500,
    )
    print(f"\n  Result: {count} synthetic samples -> {synthetic_path}")

    # ── Step 2: Download and merge Strandset-Rust-v1 ──
    print("\n--- Step 2: Merging with Strandset-Rust-v1 ---\n")
    print("  Downloading Rust code tasks from HuggingFace...")
    print("  (This adds real-world Rust code to improve generation quality)\n")

    output_path = str(PROCESSED_DIR / "train.jsonl")
    total = preprocess_and_merge(
        synthetic_path=synthetic_path,
        strandset_samples=3000,
        output_path=output_path,
    )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Tutorial 1 Complete!")
    print("=" * 60)
    print(f"\n  Total training samples: {total}")
    print(f"  Output files:")
    if PROCESSED_DIR.exists():
        for f in sorted(PROCESSED_DIR.glob("*.jsonl")):
            lines = sum(1 for _ in open(f))
            print(f"    {f.name}: {lines:,} samples")

    print(f"\n  Next: python tutorials/02_fine_tuning.py")
    print()


if __name__ == "__main__":
    main()
