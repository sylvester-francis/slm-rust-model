#!/usr/bin/env python3
"""
Tutorial 5: Deployment
=======================

Deploy your fine-tuned Rust tutor model:

  Ollama (local):
    - Run locally with `ollama run rust-mentor-0.6b`
    - Requires: Ollama installed (https://ollama.ai)

  HuggingFace (share):
    - Upload to HuggingFace Hub for others to use
    - Requires: HF_TOKEN environment variable

  Android (mobile):
    - GGUF: Load in PocketPal AI from HF Hub
    - LiteRT: Load in Google AI Edge Gallery

Prerequisites:
  - Run Tutorial 4 first (export)
  - For Ollama: install from https://ollama.ai
  - For HuggingFace: export HF_TOKEN=your_write_token

Run:
    python tutorials/05_deploy.py --target ollama        # Local Ollama
    python tutorials/05_deploy.py --target huggingface --username your-hf-user
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rustmentor.config import VARIANT_CONFIGS, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Tutorial 5: Deployment")
    parser.add_argument(
        "--variant",
        default="0.6b",
        choices=list(VARIANT_CONFIGS.keys()),
        help="Model variant to deploy (default: 0.6b)",
    )
    parser.add_argument(
        "--target",
        default="ollama",
        choices=["ollama", "huggingface"],
        help="Deployment target (default: ollama)",
    )
    parser.add_argument("--username", type=str, help="HuggingFace username")
    args = parser.parse_args()

    cfg = VARIANT_CONFIGS[args.variant]
    model_dir = str(MODELS_DIR / cfg["output_dir"].split("/")[-1])

    print("=" * 60)
    print("  Tutorial 5: Deployment")
    print("=" * 60)
    print(f"\n  Variant: {args.variant}")
    print(f"  Target: {args.target}")

    # ── Ollama ──
    if args.target == "ollama":
        print(f"\n--- Deploying to Ollama ---\n")
        from rustmentor.deploy import deploy_to_ollama

        deploy_to_ollama(
            model_name=cfg["deploy_name"],
            model_dir=model_dir,
        )

    # ── HuggingFace ──
    elif args.target == "huggingface":
        if not args.username:
            print("  Error: --username required for HuggingFace upload.")
            print("  Example: python tutorials/05_deploy.py --target huggingface --username your-name")
            return

        print(f"\n--- Uploading to HuggingFace ---\n")
        from rustmentor.deploy import upload_model

        repo_id = f"{args.username}/rust-mentor-{args.variant}"
        upload_model(
            model_dir=model_dir,
            repo_id=repo_id,
            gguf=True,
        )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Tutorial 5 Complete!")
    print("=" * 60)
    print(f"\n  Your Rust tutor model is ready!")

    if args.target == "ollama":
        print(f"  Try it: ollama run {cfg['deploy_name']}")
    elif args.target == "huggingface":
        print(f"  View: https://huggingface.co/{args.username}/rust-mentor-{args.variant}")
        print(f"\n  Mobile usage:")
        print(f"    1. Install PocketPal AI on Android")
        print(f"    2. Download the GGUF from HuggingFace")
        print(f"    3. Chat with your Rust tutor offline!")
    print()


if __name__ == "__main__":
    main()
