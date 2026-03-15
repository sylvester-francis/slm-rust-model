"""
Convert merged Qwen3 checkpoint to .litertlm format for Google AI Edge Gallery.

Usage:
    python scripts/convert_litert_lm.py --variant 0.6b --checkpoint models/rust-mentor-0.6b-litert/merged
    python scripts/convert_litert_lm.py --variant 1.7b --checkpoint models/rust-mentor-1.7b-litert/merged
    python scripts/convert_litert_lm.py --variant 4b  --checkpoint models/rust-mentor-4b-litert/merged
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3 to .litertlm")
    parser.add_argument("--variant", required=True, choices=["0.6b", "1.7b", "4b"])
    parser.add_argument("--checkpoint", required=True, help="Path to merged checkpoint dir")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--quantize", default="dynamic_int8")
    parser.add_argument("--kv_cache_max_len", type=int, default=2048)
    args = parser.parse_args()

    output_dir = args.output or os.path.dirname(args.checkpoint)
    prefix = f"rust_mentor_{args.variant.replace('.', '_')}"

    # Import litert-torch generative API
    from litert_torch.generative.examples.qwen import qwen3
    from litert_torch.generative.utilities.converter import convert_to_litert

    builders = {
        "0.6b": qwen3.build_0_6b_model,
        "1.7b": qwen3.build_1_7b_model,
        "4b": qwen3.build_4b_model,
    }

    prefill_seq_lens = [8, 64, 128, 256, 512]
    if args.kv_cache_max_len >= 2048:
        prefill_seq_lens.append(1024)

    print(f"Building Qwen3-{args.variant} model from {args.checkpoint}...")
    pytorch_model = builders[args.variant](
        checkpoint_path=args.checkpoint,
    )

    # Find tokenizer in the checkpoint dir
    tokenizer_path = None
    for name in ["tokenizer.json", "tokenizer.model"]:
        candidate = os.path.join(args.checkpoint, name)
        if os.path.exists(candidate):
            tokenizer_path = candidate
            break

    print(f"Converting to .litertlm format ({args.quantize})...")
    convert_to_litert(
        pytorch_model,
        output_path=output_dir,
        output_name_prefix=prefix,
        prefill_seq_len=prefill_seq_lens,
        kv_cache_max_len=args.kv_cache_max_len,
        quantize=args.quantize,
        output_format="litertlm",
        hf_tokenizer_model_path=args.checkpoint,
    )

    # Check output
    for f in os.listdir(output_dir):
        if f.endswith(".litertlm") or f.endswith(".tflite"):
            fpath = os.path.join(output_dir, f)
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"  ✅ {f} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
