"""
Step 5b: LiteRT Export — Convert fine-tuned model to .tflite for Android.

LiteRT leverages Android GPU/NPU acceleration (NNAPI, Tensor G3) for
2-3x faster inference vs GGUF/llama.cpp on Pixel 8 Pro.

Supports:
  - Qwen3 0.6B, 1.7B, 4B (built-in converters), 8B (custom config)
  - Gemma3 1B (Google's official converter)

Two-step process:
  1. Merge QLoRA adapter into base model (fp16)
  2. Convert merged model to .tflite using litert-torch

Requirements:
    pip install litert-torch peft transformers safetensors

Usage:
    from rustmentor.export import convert_to_litert, merge_adapter
    output = convert_to_litert("models/rust-mentor-0.6b", variant="0.6b")
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

from rustmentor.config import (
    BNB4BIT_TO_FULL,
    LITERT_QUANT_OPTIONS,
    LITERT_MODEL_SIZES,
)


def _resolve_full_precision_model(base_model: str) -> str:
    """Map quantized/unsloth model names to full-precision HF equivalents."""
    resolved = BNB4BIT_TO_FULL.get(base_model)
    if resolved:
        return resolved

    for key, val in BNB4BIT_TO_FULL.items():
        if key.lower() == base_model.lower():
            return val

    if "bnb-4bit" in base_model.lower() or "unsloth" in base_model.lower():
        name = base_model.split("/")[-1]
        name = name.replace("-unsloth-bnb-4bit", "").replace("-bnb-4bit", "")
        return f"Qwen/{name}"

    return base_model


def _is_gemma3_model(base_model: str) -> bool:
    """Check if the base model is a Gemma3 variant."""
    lower = base_model.lower()
    return "gemma-3" in lower or "gemma3" in lower


def _untie_gemma3_lm_head(model):
    """Untie lm_head from embed_tokens for Gemma3 models.

    Gemma3 ties lm_head and embed_tokens by default (weight sharing).
    The LiteRT converter and llama.cpp both need them as independent tensors.

    Handles both architectures:
      - Gemma3-4B multimodal (ForConditionalGeneration):
          model.model.language_model.embed_tokens
      - Gemma3-1B text-only (ForCausalLM):
          model.model.embed_tokens
    """
    import torch

    head = model.lm_head
    if hasattr(model.model, "language_model"):
        # Multimodal (4B): embed_tokens is on the nested Gemma3TextModel
        embed = model.model.language_model.embed_tokens
        print(f"    Detected multimodal architecture: {type(model).__name__}")
    else:
        # Text-only (1B): embed_tokens is directly on model.model
        embed = model.model.embed_tokens
        print(f"    Detected text-only architecture: {type(model).__name__}")

    model.config.tie_word_embeddings = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.tie_word_embeddings = False

    if head.weight.data_ptr() == embed.weight.data_ptr():
        head.weight = torch.nn.Parameter(embed.weight.clone())
        print("    Untied lm_head from embed_tokens")


def merge_adapter(
    adapter_dir: str,
    output_dir: str,
    base_model: str = None,
    untie_lm_head: bool = None,
) -> str:
    """Merge QLoRA adapter weights back into the base model.

    LiteRT and llama.cpp require a standard checkpoint, not a PEFT/LoRA
    checkpoint. This merges the adapter in full precision (fp16).

    For Gemma3 models, lm_head is automatically untied from embed_tokens
    (required by both LiteRT and llama.cpp converters).

    Args:
        adapter_dir: Path to the QLoRA adapter (e.g. models/rust-mentor-0.6b).
        output_dir: Where to save the merged model.
        base_model: Base model name/path. If None, reads from adapter config.
        untie_lm_head: Force untie lm_head from embed_tokens. Auto-detected
            for Gemma3 models if None.

    Returns:
        Path to the merged model directory, or empty string on failure.
    """
    merged_dir = os.path.join(output_dir, "merged")

    if os.path.exists(merged_dir) and os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"    Using existing merged checkpoint: {merged_dir}")
        return merged_dir

    try:
        import torch
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"  Error: Missing dependency: {e}")
        print("  Install with: pip install peft transformers torch safetensors")
        return ""

    os.makedirs(merged_dir, exist_ok=True)

    if base_model is None:
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model = peft_config.base_model_name_or_path
        print(f"    Base model (from adapter config): {base_model}")

    full_model = _resolve_full_precision_model(base_model)
    if full_model != base_model:
        print(f"    Resolved to full precision: {base_model} -> {full_model}")
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

    # Untie lm_head for Gemma3 (auto-detect or explicit)
    should_untie = untie_lm_head if untie_lm_head is not None else _is_gemma3_model(base_model)
    if should_untie:
        _untie_gemma3_lm_head(model)

    print(f"    Saving merged model to {merged_dir}")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    return merged_dir


def convert_gemma3_to_litert(
    model_dir: str,
    output_dir: str = None,
    model_size: str = "1b",
    output_name_prefix: str = "rust-mentor-1b-mobile",
    quantization: str = "dynamic_int8",
    kv_cache_max_len: int = 2048,
    base_model: str = None,
) -> str:
    """Convert a fine-tuned Gemma3 model to LiteRT (.tflite) + .litertlm.

    Uses Google's official Gemma3 converter. Currently supports 1B and 270M.
    Gemma3-4B is NOT supported by the LiteRT converter (use GGUF instead).

    Pipeline: merge adapter -> convert to .tflite -> bundle .litertlm

    Args:
        model_dir: Path to the QLoRA adapter directory.
        output_dir: Output directory. Defaults to model_dir + "-litert".
        model_size: Gemma3 model size ("1b" or "270m").
        output_name_prefix: Prefix for output files.
        quantization: LiteRT quantization (dynamic_int8, fp16, etc.).
        kv_cache_max_len: KV cache length.
        base_model: Full-precision base model for merging.

    Returns:
        Path to the output .tflite file, or empty string on failure.
    """
    if not os.path.exists(model_dir):
        print(f"  Error: Model not found: {model_dir}")
        return ""

    if output_dir is None:
        output_dir = model_dir + "-litert"

    os.makedirs(output_dir, exist_ok=True)
    quant = LITERT_QUANT_OPTIONS.get(quantization, quantization)

    print(f"\n  Converting Gemma3-{model_size} to LiteRT")
    print(f"    Adapter: {model_dir}")
    print(f"    Quantization: {quant}")

    # Step 1: Merge adapter (with Gemma3 lm_head untying)
    print("\n  Step 1/3: Merging adapter (with Gemma3 lm_head untying)...")
    merged_dir = merge_adapter(model_dir, output_dir, base_model=base_model, untie_lm_head=True)
    if not merged_dir:
        return ""

    # Step 2: Convert using Gemma3 converter
    print("\n  Step 2/3: Converting to .tflite (Gemma3 converter)...")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m",
                "litert_torch.generative.examples.gemma3.convert_gemma3_to_tflite",
                f"--model_size={model_size}",
                f"--checkpoint_path={merged_dir}",
                f"--output_path={output_dir}",
                f"--output_name_prefix={output_name_prefix}",
                f"--quantize={quant}",
                f"--kv_cache_max_len={kv_cache_max_len}",
                "--gpu_dynamic_shapes=true",
            ],
            check=True,
        )
    except FileNotFoundError:
        print("  Error: litert-torch not installed.")
        print("  Install with: pip install litert-torch")
        return ""
    except subprocess.CalledProcessError as e:
        print(f"  Error: Gemma3 converter failed: {e}")
        return ""

    # Step 3: Bundle .tflite + tokenizer -> .litertlm
    print("\n  Step 3/3: Bundling .litertlm...")
    from rustmentor.export.bundle import bundle_litertlm

    tflite_path = bundle_litertlm(
        tflite_path=output_dir,
        tokenizer_path=merged_dir,
        output_dir=output_dir,
        context_length=kv_cache_max_len,
        model_type="gemma3",
    )

    # Clean up merged checkpoint
    if os.path.exists(merged_dir):
        print(f"    Cleaning up merged checkpoint...")
        shutil.rmtree(merged_dir)

    return tflite_path


def convert_to_litert(
    model_dir: str = "models/rust-mentor-0.6b",
    variant: str = None,
    quantization: str = "dynamic_int8",
    output_dir: str = None,
    kv_cache_max_len: int = 2048,
    base_model: str = None,
) -> str:
    """Convert a fine-tuned Qwen3 model to LiteRT (.tflite) format.

    For Gemma3 models, use convert_gemma3_to_litert() instead.

    Args:
        model_dir: Path to the QLoRA adapter directory.
        variant: Model variant (0.6b, 1.7b, 4b, 8b). Auto-detected if None.
        quantization: Quantization (dynamic_int8, dynamic_int4, fp16, none).
        output_dir: Output directory. Defaults to model_dir + "-litert".
        kv_cache_max_len: KV cache length for inference context.
        base_model: Base model override.

    Returns:
        Path to the output .tflite file or directory.
    """
    if not os.path.exists(model_dir):
        print(f"  Error: Model not found: {model_dir}")
        print("  Run Step 3 (training) first.")
        return ""

    if output_dir is None:
        output_dir = model_dir + "-litert"

    os.makedirs(output_dir, exist_ok=True)

    if variant is None:
        dir_name = os.path.basename(model_dir.rstrip("/"))
        for v in ["0.6b", "1.7b", "4b", "8b"]:
            if v in dir_name:
                variant = v
                break
        if variant is None:
            print("  Error: Could not auto-detect variant from directory name.")
            print("  Specify with --variant (0.6b, 1.7b, 4b, 8b)")
            return ""

    quant = LITERT_QUANT_OPTIONS.get(quantization, quantization)

    print(f"\n  Converting {variant} model to LiteRT")
    print(f"    Adapter: {model_dir}")
    print(f"    Quantization: {quant}")
    print(f"    KV cache: {kv_cache_max_len}")

    print("\n  Step 1/2: Merging QLoRA adapter into base model...")
    merged_dir = merge_adapter(model_dir, output_dir, base_model=base_model)
    if not merged_dir:
        return ""

    print("\n  Step 2/2: Converting to LiteRT .tflite format...")

    try:
        if variant in LITERT_MODEL_SIZES:
            tflite_path = _convert_with_builtin(
                merged_dir, output_dir, variant, quant, kv_cache_max_len
            )
        elif variant == "8b":
            tflite_path = _convert_8b_custom(
                merged_dir, output_dir, quant, kv_cache_max_len
            )
        else:
            print(f"  Error: Unsupported variant: {variant}")
            return ""
    except ImportError:
        print("  Error: litert-torch not installed.")
        print("  Install with: pip install litert-torch")
        return ""

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

    builders = {
        "0.6b": qwen3.build_0_6b_model,
        "1.7b": qwen3.build_1_7b_model,
        "4b": qwen3.build_4b_model,
    }

    build_fn = builders[LITERT_MODEL_SIZES[variant]]
    output_prefix = f"rust_mentor_{variant.replace('.', '_')}"

    print(f"    Building re-authored Qwen3-{variant} model...")
    pytorch_model = build_fn(
        checkpoint_path=merged_dir,
        kv_cache_max_len=kv_cache_max_len,
    )

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

    tflite_files = [f for f in os.listdir(output_dir) if f.endswith(".tflite")]
    if tflite_files:
        tflite_path = os.path.join(output_dir, tflite_files[0])
        size_mb = os.path.getsize(tflite_path) / 1024 / 1024
        print(f"\n  LiteRT exported: {tflite_files[0]} ({size_mb:.0f} MB)")
        return tflite_path
    else:
        print(f"\n  LiteRT files exported to {output_dir}")
        return output_dir


def _convert_8b_custom(
    merged_dir: str, output_dir: str, quant: str, kv_cache_max_len: int
) -> str:
    """Convert Qwen3-8B using a custom model config (not in litert-torch builtins)."""
    from litert_torch.generative.layers import model_config
    from litert_torch.generative.utilities import converter, model_builder

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
        print(f"\n  LiteRT exported: {tflite_files[0]} ({size_mb:.0f} MB)")
        return tflite_path
    else:
        print(f"\n  LiteRT files exported to {output_dir}")
        return output_dir


if __name__ == "__main__":
    convert_to_litert()
