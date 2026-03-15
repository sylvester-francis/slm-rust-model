"""
LiteRT Conversion: Export fine-tuned model to LiteRT (.tflite) for Android GPU/NPU inference.

Converts merged QLoRA checkpoint → LiteRT format using litert-torch.
Supports Qwen3 0.6B, 1.7B, 4B natively. 8B uses a custom config.

LiteRT leverages Android GPU/NPU acceleration (NNAPI, Tensor G3) for
2-3x faster inference vs GGUF/llama.cpp on Pixel 8 Pro.

Requirements:
    pip install litert-torch peft transformers safetensors
"""

import os
import shutil
from pathlib import Path


# Quantization options for LiteRT
QUANT_OPTIONS = {
    "dynamic_int8": "dynamic_int8",       # Good balance (default)
    "dynamic_int4": "dynamic_int4_block32",  # Smallest, fast on mobile
    "fp16": "fp16",                        # Higher quality, larger
    "none": "none",                        # Full precision (debug only)
}

# Variant → litert-torch model size key
LITERT_MODEL_SIZES = {
    "0.6b": "0.6b",
    "1.7b": "1.7b",
    "4b": "4b",
}

# Unsloth saves the bnb-4bit quantized model name in the adapter config.
# For merging we need the full-precision base model to avoid rounding errors.
BNB4BIT_TO_FULL = {
    "unsloth/qwen3-0.6b-unsloth-bnb-4bit": "Qwen/Qwen3-0.6B",
    "unsloth/qwen3-1.7b-unsloth-bnb-4bit": "Qwen/Qwen3-1.7B",
    "unsloth/qwen3-4b-unsloth-bnb-4bit": "Qwen/Qwen3-4B",
    "unsloth/qwen3-8b-unsloth-bnb-4bit": "Qwen/Qwen3-8B",
    "unsloth/Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "unsloth/Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "unsloth/Qwen3-4B": "Qwen/Qwen3-4B",
    "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
}


def _resolve_full_precision_model(base_model: str) -> str:
    """Map quantized/unsloth model names to full-precision HF equivalents."""
    # Direct lookup
    resolved = BNB4BIT_TO_FULL.get(base_model)
    if resolved:
        return resolved
    # Case-insensitive fallback
    for key, val in BNB4BIT_TO_FULL.items():
        if key.lower() == base_model.lower():
            return val
    # If it contains "bnb-4bit" or "unsloth", try to derive the Qwen name
    if "bnb-4bit" in base_model.lower() or "unsloth" in base_model.lower():
        # e.g. "unsloth/qwen3-8b-unsloth-bnb-4bit" → extract "Qwen3-8B"
        name = base_model.split("/")[-1]
        name = name.replace("-unsloth-bnb-4bit", "").replace("-bnb-4bit", "")
        # Capitalize properly: qwen3-8b → Qwen3-8B
        parts = name.split("-")
        if len(parts) >= 2:
            model_name = parts[0].capitalize() + parts[0][1:].replace(parts[0][0], "") + "-" + parts[1].upper()
            return f"Qwen/{name}"
    return base_model


def merge_adapter(adapter_dir: str, output_dir: str, base_model: str = None) -> str:
    """Merge QLoRA adapter weights back into the base model.

    LiteRT requires a standard checkpoint, not a PEFT/LoRA checkpoint.
    This merges the adapter in full precision (fp16) to avoid rounding errors.

    Args:
        adapter_dir: Path to the QLoRA adapter (e.g. models/rust-mentor-0.6b)
        output_dir: Where to save the merged model
        base_model: Base model name/path. If None, reads from adapter config.

    Returns:
        Path to the merged model directory.
    """
    # Check for existing merged checkpoint BEFORE importing transformers/peft
    # (those imports fail if torch/torchvision versions are mismatched)
    merged_dir = os.path.join(output_dir, "merged")
    if os.path.exists(merged_dir) and os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"    Using existing merged checkpoint: {merged_dir}")
        return merged_dir

    import json
    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(merged_dir, exist_ok=True)

    # Determine base model from adapter config if not specified
    if base_model is None:
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model = peft_config.base_model_name_or_path
        print(f"    Base model (from adapter config): {base_model}")

    # Resolve to full-precision model to avoid 4-bit merge rounding errors
    full_model = _resolve_full_precision_model(base_model)
    if full_model != base_model:
        print(f"    Resolved to full precision: {base_model} → {full_model}")
    base_model = full_model

    print(f"    Loading base model: {base_model} (fp16)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print(f"    Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("    Merging adapter weights...")
    model = model.merge_and_unload()

    print(f"    Saving merged model → {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    return merged_dir


def convert_to_litert(
    model_dir: str = "models/rust-mentor-0.6b",
    variant: str = None,
    quantization: str = "dynamic_int8",
    output_dir: str = None,
    kv_cache_max_len: int = 2048,
    base_model: str = None,
) -> str:
    """Convert a fine-tuned Qwen3 model to LiteRT (.tflite) format.

    Two-step process:
    1. Merge QLoRA adapter into base model
    2. Convert merged model to .tflite using litert-torch

    Args:
        model_dir: Path to the QLoRA adapter directory
        variant: Model variant (0.6b, 1.7b, 4b, 8b). Auto-detected if None.
        quantization: Quantization level (dynamic_int8, dynamic_int4, fp16, none)
        output_dir: Output directory for .tflite files. Defaults to model_dir + "-litert"
        kv_cache_max_len: KV cache length for inference context
        base_model: Base model override. If None, reads from adapter config.

    Returns:
        Path to the output .tflite file or directory.
    """
    if output_dir is None:
        output_dir = model_dir + "-litert"

    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect variant from model_dir name
    if variant is None:
        dir_name = os.path.basename(model_dir.rstrip("/"))
        for v in ["0.6b", "1.7b", "4b", "8b"]:
            if v in dir_name:
                variant = v
                break
        if variant is None:
            print("  ⚠️  Could not auto-detect variant. Specify with --litert-variant.")
            print("     Supported: 0.6b, 1.7b, 4b, 8b")
            return None

    quant = QUANT_OPTIONS.get(quantization, quantization)

    print(f"\n  🔄 Converting {variant} model to LiteRT")
    print(f"     Adapter: {model_dir}")
    print(f"     Quantization: {quant}")
    print(f"     KV cache: {kv_cache_max_len}")

    # Step 1: Merge adapter
    print("\n  Step 1/2: Merging QLoRA adapter into base model...")
    merged_dir = merge_adapter(model_dir, output_dir, base_model=base_model)

    # Step 2: Convert to LiteRT
    print("\n  Step 2/2: Converting to LiteRT .tflite format...")

    if variant in LITERT_MODEL_SIZES:
        tflite_path = _convert_with_builtin(
            merged_dir, output_dir, variant, quant, kv_cache_max_len
        )
    elif variant == "8b":
        tflite_path = _convert_8b_custom(
            merged_dir, output_dir, quant, kv_cache_max_len
        )
    else:
        print(f"  ❌ Unsupported variant: {variant}")
        return None

    # Clean up merged checkpoint to save disk space (keep .tflite)
    merged_path = os.path.join(output_dir, "merged")
    if os.path.exists(merged_path):
        print(f"    Cleaning up merged checkpoint...")
        shutil.rmtree(merged_path)

    return tflite_path


def _convert_with_builtin(
    merged_dir: str, output_dir: str, variant: str, quant: str, kv_cache_max_len: int
) -> str:
    """Convert using litert-torch's built-in Qwen3 converter (0.6B, 1.7B, 4B)."""
    from litert_torch.generative.examples.qwen import qwen3
    from litert_torch.generative.utilities import converter

    model_size = LITERT_MODEL_SIZES[variant]
    builders = {
        "0.6b": qwen3.build_0_6b_model,
        "1.7b": qwen3.build_1_7b_model,
        "4b": qwen3.build_4b_model,
    }

    build_fn = builders[model_size]
    output_prefix = f"rust_mentor_{variant.replace('.', '_')}"

    print(f"    Building re-authored Qwen3-{variant} model...")
    pytorch_model = build_fn(
        checkpoint_path=merged_dir,
        kv_cache_max_len=kv_cache_max_len,
    )

    # Prefill sequence lengths — multiple sizes for efficient batched prefill
    prefill_seq_lens = [8, 64, 128, 256, 512]
    if kv_cache_max_len >= 2048:
        prefill_seq_lens.append(1024)

    print(f"    Converting to .tflite ({quant})...")
    converter.convert_to_tflite(
        pytorch_model,
        output_path=output_dir,
        output_name_prefix=output_prefix,
        prefill_seq_len=prefill_seq_lens,
        kv_cache_max_len=kv_cache_max_len,
        quantize=quant,
    )

    # Find output file
    tflite_files = [f for f in os.listdir(output_dir) if f.endswith(".tflite")]
    if tflite_files:
        tflite_path = os.path.join(output_dir, tflite_files[0])
        size_mb = os.path.getsize(tflite_path) / 1024 / 1024
        print(f"\n  ✅ LiteRT exported: {tflite_files[0]} ({size_mb:.0f} MB)")
        return tflite_path
    else:
        print(f"\n  ✅ LiteRT files exported to {output_dir}")
        return output_dir


def _convert_8b_custom(
    merged_dir: str, output_dir: str, quant: str, kv_cache_max_len: int
) -> str:
    """Convert Qwen3-8B using a custom model config (not in litert-torch builtins)."""
    from litert_torch.generative.layers import model_config
    from litert_torch.generative.layers import kv_cache as kv_utils
    from litert_torch.generative.utilities import converter, model_builder

    # Qwen3-8B architecture config
    # Matches: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
    norm_config = model_config.NormalizationConfig(
        type=model_config.NormalizationType.RMS_NORM,
        epsilon=1e-6,
    )
    attn_config = model_config.AttentionConfig(
        num_heads=32,
        head_dim=128,
        num_query_groups=8,
        rotary_base=1_000_000,
        qkv_use_bias=False,
        output_proj_use_bias=False,
        qk_norm=True,
        qk_norm_before_rope=True,
        logit_softcap=None,
    )
    ff_config = model_config.FeedForwardConfig(
        type=model_config.FeedForwardType.GATED,
        activation=model_config.ActivationType.SILU,
        intermediate_size=14336,
        use_bias=False,
    )
    block_config = model_config.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

    config = model_config.ModelConfig(
        vocab_size=151936,
        num_layers=36,
        max_seq_len=40960,
        embedding_dim=4096,
        kv_cache_max_len=kv_cache_max_len,
        block_configs=block_config,
        final_norm_config=norm_config,
        lm_head_share_weight_with_embedding=False,
        enable_hlfb=True,
    )

    # Qwen3 weight name mapping
    tensor_names = model_builder.TensorNames(
        attn_query_proj="self_attn.q_proj",
        attn_key_proj="self_attn.k_proj",
        attn_value_proj="self_attn.v_proj",
        attn_output_proj="self_attn.o_proj",
        attn_query_norm="self_attn.q_norm",
        attn_key_norm="self_attn.k_norm",
        ff_up_proj="mlp.up_proj",
        ff_down_proj="mlp.down_proj",
        ff_gate_proj="mlp.gate_proj",
        pre_attn_norm="input_layernorm",
        post_attn_norm="post_attention_layernorm",
        embedding="model.embed_tokens",
        final_norm="model.norm",
        lm_head="lm_head",
        block_prefix="model.layers",
    )

    output_prefix = "rust_mentor_8b"

    print("    Building re-authored Qwen3-8B model (custom config)...")
    model = model_builder.build_decoder_only_model(
        checkpoint_path=merged_dir,
        config=config,
        tensor_names=tensor_names,
    )

    prefill_seq_lens = [8, 64, 128, 256, 512, 1024]

    print(f"    Converting to .tflite ({quant})...")
    converter.convert_to_tflite(
        model,
        output_path=output_dir,
        output_name_prefix=output_prefix,
        prefill_seq_len=prefill_seq_lens,
        kv_cache_max_len=kv_cache_max_len,
        quantize=quant,
    )

    tflite_files = [f for f in os.listdir(output_dir) if f.endswith(".tflite")]
    if tflite_files:
        tflite_path = os.path.join(output_dir, tflite_files[0])
        size_mb = os.path.getsize(tflite_path) / 1024 / 1024
        print(f"\n  ✅ LiteRT exported: {tflite_files[0]} ({size_mb:.0f} MB)")
        return tflite_path
    else:
        print(f"\n  ✅ LiteRT files exported to {output_dir}")
        return output_dir


if __name__ == "__main__":
    convert_to_litert()
