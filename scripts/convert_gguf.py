"""
GGUF Conversion: Export fine-tuned model to GGUF for mobile/edge inference.

Supports quantization levels:
- q4_k_m: Best balance for Pixel 8 Pro (~4.5GB for 8B model)
- q5_k_m: Higher quality, larger size (~5.5GB)
- q8_0:   Near-lossless, biggest (~8GB)
- q2_k:   Maximum compression, quality tradeoff (~2.5GB)
"""

import os


QUANT_MAP = {
    "q2_k": "q2_k",
    "q4_k_m": "q4_k_m",
    "q5_k_m": "q5_k_m",
    "q8_0": "q8_0",
    "f16": "f16",
}


def convert_to_gguf(
    model_dir: str = "models/rust-mentor-8b",
    quantization: str = "q4_k_m",
    output_dir: str = None,
) -> str:
    """Convert model to GGUF format using Unsloth."""
    from unsloth import FastLanguageModel

    if output_dir is None:
        output_dir = model_dir + "-GGUF"

    os.makedirs(output_dir, exist_ok=True)

    quant = QUANT_MAP.get(quantization, quantization)

    print(f"\n  Loading model from {model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"  Exporting to GGUF ({quant})...")
    print(f"  Output: {output_dir}")

    # Unsloth's save_pretrained_gguf handles merging + conversion
    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quant,
    )

    # Find the output file
    gguf_files = [f for f in os.listdir(output_dir) if f.endswith(".gguf")]
    if gguf_files:
        output_path = os.path.join(output_dir, gguf_files[0])
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n  ✅ GGUF exported: {gguf_files[0]} ({size_mb:.0f} MB)")
        return output_path
    else:
        print(f"\n  ✅ GGUF files exported to {output_dir}")
        return output_dir


if __name__ == "__main__":
    convert_to_gguf()
