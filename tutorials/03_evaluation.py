#!/usr/bin/env python3
"""
Tutorial 3: Model Evaluation
==============================

Tests the fine-tuned model on 5 Rust tutoring prompts and scores
keyword coverage to measure domain knowledge retention.

Evaluation prompts cover:
  1. Ownership explanation (for Go developers)
  2. Error handling patterns (Result, Option, ?)
  3. Code review with borrow checker
  4. Traits vs Go interfaces
  5. CLI tooling with clap

Prerequisites:
  - Run Tutorial 2 first (training)
  - GPU required for inference

Run:
    python tutorials/03_evaluation.py                  # Default: 0.6B
    python tutorials/03_evaluation.py --variant 1.7b   # Evaluate 1.7B
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rustmentor.training import evaluate_model
from rustmentor.config import VARIANT_CONFIGS, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Tutorial 3: Evaluation")
    parser.add_argument(
        "--variant",
        default="0.6b",
        choices=list(VARIANT_CONFIGS.keys()),
        help="Model variant to evaluate (default: 0.6b)",
    )
    args = parser.parse_args()

    cfg = VARIANT_CONFIGS[args.variant]
    model_dir = str(MODELS_DIR / cfg["output_dir"].split("/")[-1])

    print("=" * 60)
    print("  Tutorial 3: Model Evaluation")
    print("=" * 60)
    print(f"\n  Variant: {args.variant}")
    print(f"  Model: {model_dir}")

    if not os.path.exists(model_dir):
        print(f"\n  Error: Model not found: {model_dir}")
        print(f"  Run Tutorial 2 first: python tutorials/02_fine_tuning.py --variant {args.variant}")
        return

    # ── Evaluate ──
    print("\n--- Evaluating on 5 Rust tutoring prompts ---\n")
    results = evaluate_model(model_dir=model_dir)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Tutorial 3 Complete!")
    print("=" * 60)

    if results:
        print(f"\n  Keyword accuracy: {results.get('keyword_accuracy', 'N/A')}")
        print(f"  Avg response length: {results.get('avg_response_length', 'N/A')} chars")
        print(f"\n  Detailed results saved to: {model_dir}/eval_results.json")

    print(f"\n  Next: python tutorials/04_export.py --variant {args.variant}")
    print()


if __name__ == "__main__":
    main()
