"""
RustMentor SLM - Fine-tuning pipeline for Rust programming tutor models.

Supports Qwen3 (0.6B-8B) and Gemma3 (1B, 4B) base models with QLoRA
via Unsloth. Exports to GGUF (llama.cpp) or LiteRT (.tflite) for
offline Android inference.

Usage:
    from rustmentor.config import SYSTEM_PROMPT, VARIANT_CONFIGS
    from rustmentor.data import generate_rust_dataset, preprocess_and_merge
    from rustmentor.training import train_model, evaluate_model
    from rustmentor.export import convert_to_gguf, convert_to_litert
    from rustmentor.deploy import upload_model, deploy_to_ollama
"""

__version__ = "1.0.0"
