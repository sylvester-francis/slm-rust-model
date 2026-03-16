#!/usr/bin/env python3
"""
Tutorial 4: Model Export
=========================

Export your fine-tuned model for deployment:

  GGUF format (default):
    - Compatible with llama.cpp, Ollama, PocketPal AI
    - Best for: CPU inference, Android via PocketPal
    - Sizes: q4_k_m (~0.4-4.5GB depending on variant)

  LiteRT format (optional):
    - .tflite for Android GPU/NPU acceleration
    - Best for: Pixel 8 Pro, Tensor G3 devices
    - 2-3x faster than GGUF on supported hardware

Prerequisites:
  - Run Tutorial 2 first (training)
  - For GGUF: pip install unsloth
  - For LiteRT: pip install litert-torch peft transformers

Run:
    python tutorials/04_export.py                        # GGUF (default)
    python tutorials/04_export.py --format litert        # LiteRT
    python tutorials/04_export.py --variant 1.7b         # Export 1.7B
    python tutorials/04_export.py --quant q8_0           # Higher quality GGUF
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rustmentor.config import VARIANT_CONFIGS, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Tutorial 4: Export")
    parser.add_argument(
        "--variant",
        default="0.6b",
        choices=list(VARIANT_CONFIGS.keys()),
        help="Model variant to export (default: 0.6b)",
    )
    parser.add_argument(
        "--format",
        default="gguf",
        choices=["gguf", "litert", "both"],
        help="Export format (default: gguf)",
    )
    parser.add_argument(
        "--quant",
        default="q4_k_m",
        help="GGUF quantization (default: q4_k_m)",
    )
    args = parser.parse_args()

    cfg = VARIANT_CONFIGS[args.variant]
    model_dir = str(MODELS_DIR / cfg["output_dir"].split("/")[-1])

    print("=" * 60)
    print("  Tutorial 4: Model Export")
    print("=" * 60)
    print(f"\n  Variant: {args.variant}")
    print(f"  Format: {args.format}")
    print(f"  Model: {model_dir}")

    if not os.path.exists(model_dir):
        print(f"\n  Error: Model not found: {model_dir}")
        print(f"  Run Tutorial 2 first: python tutorials/02_fine_tuning.py --variant {args.variant}")
        return

    # ── GGUF Export ──
    if args.format in ("gguf", "both"):
        print(f"\n--- Exporting to GGUF ({args.quant}) ---\n")
        from rustmentor.export import convert_to_gguf

        output = convert_to_gguf(
            model_dir=model_dir,
            quantization=args.quant,
        )
        if output:
            print(f"  GGUF exported: {output}")

    # ── LiteRT Export ──
    if args.format in ("litert", "both"):
        print(f"\n--- Exporting to LiteRT (dynamic_int8) ---\n")
        from rustmentor.export import convert_to_litert

        output = convert_to_litert(
            model_dir=model_dir,
            variant=args.variant,
            quantization="dynamic_int8",
        )
        if output:
            print(f"  LiteRT exported: {output}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Tutorial 4 Complete!")
    print("=" * 60)
    print(f"\n  Next: python tutorials/05_deploy.py --variant {args.variant}")
    print()


if __name__ == "__main__":
    main()
