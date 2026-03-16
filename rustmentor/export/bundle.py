"""
Step 5c: Bundle .tflite + tokenizer into .litertlm for Google AI Edge Gallery.

This is a post-processing step — run AFTER LiteRT conversion produces
the .tflite file. The .litertlm format bundles the model, tokenizer,
and metadata into a single file for the AI Edge Gallery app.

Requirements:
    pip install litert-torch

Usage:
    from rustmentor.export import bundle_litertlm
    output = bundle_litertlm(
        tflite_path="models/rust-mentor-0.6b-litert/model.tflite",
        tokenizer_path="models/rust-mentor-0.6b-litert/merged",
        output_dir="models/rust-mentor-0.6b-litert",
    )
"""

import glob
import os
import shutil

from rustmentor.config import STOP_TOKENS


def bundle_litertlm(
    tflite_path: str,
    tokenizer_path: str,
    output_dir: str,
    context_length: int = 2048,
    model_type: str = "qwen3",
    stop_token_ids: list = None,
) -> str:
    """Bundle a .tflite model + tokenizer.json into .litertlm.

    Args:
        tflite_path: Path to .tflite file or directory containing one.
        tokenizer_path: Path to tokenizer.json or directory containing one.
        output_dir: Where to write the .litertlm file.
        context_length: Context length for the model.
        model_type: Model type ("qwen3", "gemma3", or "generic").
        stop_token_ids: Stop token IDs. Defaults based on model_type.

    Returns:
        Path to the .litertlm file, or empty string on failure.
    """
    # Resolve .tflite path
    if os.path.isdir(tflite_path):
        tflites = glob.glob(os.path.join(tflite_path, "*.tflite"))
        if not tflites:
            print(f"  Error: No .tflite files found in {tflite_path}")
            return ""
        tflite_path = tflites[0]

    if not os.path.exists(tflite_path):
        print(f"  Error: .tflite file not found: {tflite_path}")
        return ""

    # Resolve tokenizer path
    if os.path.isdir(tokenizer_path):
        candidate = os.path.join(tokenizer_path, "tokenizer.json")
        if os.path.exists(candidate):
            tokenizer_path = candidate
        else:
            print(f"  Error: No tokenizer.json found in {tokenizer_path}")
            return ""

    if not os.path.exists(tokenizer_path):
        print(f"  Error: Tokenizer not found: {tokenizer_path}")
        return ""

    # Default stop tokens based on model type
    if stop_token_ids is None:
        stop_token_ids = STOP_TOKENS.get(model_type, STOP_TOKENS["qwen3"])

    try:
        from litert_torch.generative.utilities.litertlm_builder import build_litertlm
    except ImportError:
        print("  Error: litert-torch not installed.")
        print("  Install with: pip install litert-torch")
        return ""

    print(f"  Bundling: {os.path.basename(tflite_path)}")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Context: {context_length}")
    print(f"  Stop tokens: {stop_token_ids}")

    os.makedirs(output_dir, exist_ok=True)
    workdir = os.path.join(output_dir, "_bundle_tmp")
    os.makedirs(workdir, exist_ok=True)

    build_litertlm(
        tflite_model_path=tflite_path,
        workdir=workdir,
        output_path=output_dir,
        context_length=context_length,
        hf_tokenizer_model_path=tokenizer_path,
        llm_model_type=model_type,
        stop_token_ids=stop_token_ids,
    )

    # Clean up temp directory
    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    # Report results
    for f in os.listdir(output_dir):
        if f.endswith(".litertlm"):
            fpath = os.path.join(output_dir, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"  Bundled: {f} ({size_mb:.0f} MB)")
            return fpath

    print("  Warning: No .litertlm file was produced.")
    return ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bundle .tflite into .litertlm")
    parser.add_argument("--tflite", required=True, help=".tflite file or directory")
    parser.add_argument("--tokenizer", required=True, help="Dir with tokenizer.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--model_type", default="qwen3", choices=["generic", "qwen3", "gemma3"])
    args = parser.parse_args()

    bundle_litertlm(
        tflite_path=args.tflite,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        context_length=args.context_length,
        model_type=args.model_type,
    )
