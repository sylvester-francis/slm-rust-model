"""
Step 5a: GGUF Export — Convert fine-tuned model to GGUF for llama.cpp.

Two conversion backends:
  1. Unsloth (default) — fastest, works for Qwen3 models
  2. llama.cpp — needed for Gemma3 and other models Unsloth doesn't support

Requirements:
    Unsloth: pip install unsloth
    llama.cpp: git, cmake, pip install gguf sentencepiece protobuf

Usage:
    from rustmentor.export import convert_to_gguf
    output = convert_to_gguf("models/rust-mentor-0.6b", quantization="q4_k_m")

    # For Gemma3 (uses llama.cpp):
    from rustmentor.export.gguf import convert_to_gguf_llamacpp
    output = convert_to_gguf_llamacpp("models/merged", "models/output", "Q4_K_M")
"""

import os
import subprocess
import sys

from rustmentor.config import GGUF_QUANT_MAP


def convert_to_gguf(
    model_dir: str = "models/rust-mentor-8b",
    quantization: str = "q4_k_m",
    output_dir: str = None,
) -> str:
    """Convert model to GGUF format using Unsloth.

    Best for Qwen3 models. For Gemma3, use convert_to_gguf_llamacpp().

    Args:
        model_dir: Path to the trained model (adapter + tokenizer).
        quantization: Quantization level (q2_k, q4_k_m, q5_k_m, q8_0, f16).
        output_dir: Output directory. Defaults to model_dir + "-GGUF".

    Returns:
        Path to the output GGUF file or directory.
    """
    if not os.path.exists(model_dir):
        print(f"  Error: Model not found: {model_dir}")
        print("  Run Step 3 (training) first.")
        return ""

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("  Error: Unsloth not installed.")
        print("  Install with: pip install unsloth")
        print("  Or use convert_to_gguf_llamacpp() for llama.cpp-based conversion.")
        return ""

    if output_dir is None:
        output_dir = model_dir + "-GGUF"

    os.makedirs(output_dir, exist_ok=True)
    quant = GGUF_QUANT_MAP.get(quantization, quantization)

    print(f"\n  Loading model from {model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"  Exporting to GGUF ({quant})...")
    print(f"  Output: {output_dir}")

    model.save_pretrained_gguf(
        output_dir,
        tokenizer,
        quantization_method=quant,
    )

    gguf_files = [f for f in os.listdir(output_dir) if f.endswith(".gguf")]
    if gguf_files:
        output_path = os.path.join(output_dir, gguf_files[0])
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n  GGUF exported: {gguf_files[0]} ({size_mb:.0f} MB)")
        return output_path
    else:
        print(f"\n  GGUF files exported to {output_dir}")
        return output_dir


def _ensure_llamacpp(project_root: str = ".") -> str:
    """Build llama.cpp if not already built. Returns path to llama-quantize binary."""
    quantize_bin = os.path.join(project_root, "llama.cpp/build/bin/llama-quantize")
    if os.path.exists(quantize_bin):
        print("    llama.cpp already built")
        return quantize_bin

    print("    Building llama.cpp...")
    subprocess.run(["pip", "install", "-q", "gguf", "sentencepiece", "protobuf", "numpy"], check=True)

    llama_dir = os.path.join(project_root, "llama.cpp")
    if not os.path.exists(llama_dir):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", llama_dir],
            check=True,
        )

    build_dir = os.path.join(llama_dir, "build")
    subprocess.run(
        ["cmake", "-B", build_dir, "-S", llama_dir, "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", build_dir, "--config", "Release", "-j4", "--target", "llama-quantize"],
        check=True,
    )

    if not os.path.exists(quantize_bin):
        print("  Error: Failed to build llama-quantize")
        return ""

    print("    llama.cpp built successfully")
    return quantize_bin


def convert_to_gguf_llamacpp(
    merged_dir: str,
    output_dir: str,
    quantization: str = "Q4_K_M",
    model_name: str = "rust-mentor",
    base_model: str = None,
    project_root: str = ".",
) -> str:
    """Convert a merged HuggingFace checkpoint to GGUF via llama.cpp.

    This is needed for models that Unsloth can't export (e.g. Gemma3).
    Uses llama.cpp's convert_hf_to_gguf.py for HF->f16 GGUF, then
    llama-quantize for the final quantization step.

    Also restores the original tokenizer from the base model, since
    Unsloth may modify it during training and llama.cpp rejects
    unknown tokenizer hashes.

    Args:
        merged_dir: Path to the merged fp16 checkpoint.
        output_dir: Where to write the GGUF file.
        quantization: llama.cpp quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.).
        model_name: Prefix for the output filename.
        base_model: HF model ID for restoring the original tokenizer.
        project_root: Root directory for llama.cpp build.

    Returns:
        Path to the quantized GGUF file, or empty string on failure.
    """
    if not os.path.exists(merged_dir):
        print(f"  Error: Merged checkpoint not found: {merged_dir}")
        return ""

    os.makedirs(output_dir, exist_ok=True)

    f16_gguf = os.path.join(output_dir, f"{model_name}-f16.gguf")
    final_gguf = os.path.join(output_dir, f"{model_name}-{quantization}.gguf")

    # Skip if already done
    if os.path.exists(final_gguf):
        size_gb = os.path.getsize(final_gguf) / 1024**3
        print(f"    GGUF already exists: {final_gguf} ({size_gb:.2f} GB)")
        return final_gguf

    # Build llama.cpp
    quantize_bin = _ensure_llamacpp(project_root)
    if not quantize_bin:
        return ""

    # Restore original tokenizer (Unsloth may have modified it)
    if base_model:
        print(f"    Restoring original tokenizer from {base_model}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                base_model,
                local_dir=merged_dir,
                allow_patterns=["tokenizer*", "special_tokens_map*"],
                token=os.environ.get("HF_TOKEN", ""),
            )
            print("    Original tokenizer restored")
        except Exception as e:
            print(f"    Warning: Could not restore tokenizer: {e}")

    # Step 1: HuggingFace -> f16 GGUF
    print("    Converting HuggingFace checkpoint to f16 GGUF...")
    convert_script = os.path.join(project_root, "llama.cpp/convert_hf_to_gguf.py")
    result = subprocess.run(
        [sys.executable, convert_script, merged_dir, "--outfile", f16_gguf, "--outtype", "f16"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Error: HF->GGUF conversion failed:\n{result.stderr[-500:]}")
        return ""

    f16_size = os.path.getsize(f16_gguf) / 1024**3
    print(f"    f16 GGUF: {f16_size:.1f} GB")

    # Step 2: f16 -> quantized GGUF
    print(f"    Quantizing to {quantization}...")
    result = subprocess.run(
        [quantize_bin, f16_gguf, final_gguf, quantization],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Error: Quantization failed:\n{result.stderr[-500:]}")
        return ""

    final_size = os.path.getsize(final_gguf) / 1024**3
    print(f"    {quantization} GGUF: {final_size:.2f} GB")

    # Clean up f16 intermediate
    if os.path.exists(f16_gguf):
        os.remove(f16_gguf)
        print("    Removed intermediate f16 GGUF")

    return final_gguf


if __name__ == "__main__":
    convert_to_gguf()
