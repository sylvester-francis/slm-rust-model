"""
Bundle a .tflite model + tokenizer into .litertlm format for Google AI Edge Gallery.

This is a POST-PROCESSING step — run AFTER convert_v3_to_tflite produces the .tflite.

Usage:
    python scripts/bundle_litertlm.py \
        --tflite models/rust-mentor-0.6b-litert/rust_mentor_0_6b_q8_ekv2048.tflite \
        --tokenizer models/rust-mentor-0.6b-litert/merged \
        --output models/rust-mentor-0.6b-litert \
        --context_length 2048
"""

import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser(description="Bundle .tflite → .litertlm")
    parser.add_argument("--tflite", required=True, help=".tflite file path or dir containing one")
    parser.add_argument("--tokenizer", required=True, help="Dir with tokenizer.json (HF format)")
    parser.add_argument("--output", required=True, help="Output directory for .litertlm")
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--model_type", default="qwen3", choices=["generic", "qwen3", "gemma3"])
    args = parser.parse_args()

    # Find .tflite file
    tflite_path = args.tflite
    if os.path.isdir(tflite_path):
        tflites = glob.glob(os.path.join(tflite_path, "*.tflite"))
        if not tflites:
            print(f"No .tflite files found in {tflite_path}")
            return
        tflite_path = tflites[0]

    # Resolve tokenizer path — build_litertlm needs the tokenizer.json FILE, not directory
    tokenizer_path = args.tokenizer
    if os.path.isdir(tokenizer_path):
        candidate = os.path.join(tokenizer_path, "tokenizer.json")
        if os.path.exists(candidate):
            tokenizer_path = candidate
        else:
            print(f"  No tokenizer.json found in {tokenizer_path}")
            return

    print(f"Bundling: {tflite_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Output: {args.output}")
    print(f"Context: {args.context_length}")

    from litert_torch.generative.utilities.litertlm_builder import build_litertlm

    os.makedirs(args.output, exist_ok=True)
    workdir = os.path.join(args.output, "_bundle_tmp")
    os.makedirs(workdir, exist_ok=True)

    # Qwen3 stop tokens: <|im_end|>=151645, <|endoftext|>=151643
    build_litertlm(
        tflite_model_path=tflite_path,
        workdir=workdir,
        output_path=args.output,
        context_length=args.context_length,
        hf_tokenizer_model_path=tokenizer_path,
        llm_model_type=args.model_type,
        stop_token_ids=[151645, 151643],
    )

    # Clean up temp
    import shutil
    if os.path.exists(workdir):
        shutil.rmtree(workdir)

    # Report
    for f in os.listdir(args.output):
        if f.endswith(".litertlm"):
            fpath = os.path.join(args.output, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"✅ {f} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
