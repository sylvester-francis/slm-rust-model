"""
Shared configuration for the RustMentor SLM pipeline.

This is the single source of truth for all constants used across
data collection, training, export, and deployment scripts.
"""

from pathlib import Path


# ── Project Paths ──────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


# ── System Prompt ──────────────────────────────────────────
# Used in training data, evaluation, and Ollama deployment.
# Change this to customize the model's persona.

SYSTEM_PROMPT = (
    "You are RustMentor, an expert Rust programming tutor. "
    "The student is an experienced Go, Python, and TypeScript "
    "developer learning Rust by building CLI tools.\n\n"
    "Your teaching style:\n"
    "- Draw parallels to Go/Python/TypeScript concepts they already know\n"
    "- Explain ownership, borrowing, and lifetimes with practical examples\n"
    "- When reviewing code, explain what the borrow checker is doing and why\n"
    "- Keep explanations concise with code snippets\n"
    "- Guide them to write the code themselves rather than giving full solutions\n"
    "- Use the Socratic method when appropriate"
)


# ── Training Defaults ──────────────────────────────────────

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B"
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4
DEFAULT_SEQ_LENGTH = 2048
DEFAULT_GGUF_QUANT = "q4_k_m"


# ── Variant Presets ────────────────────────────────────────
# Use with `--variant` flag to set model-specific defaults.

VARIANT_CONFIGS = {
    "8b": {
        "model": "unsloth/Qwen3-8B",
        "full_precision_model": "Qwen/Qwen3-8B",
        "lora_r": 32,
        "lora_alpha": 32,
        "batch_size": 2,
        "grad_accum": 4,
        "output_dir": "models/rust-mentor-8b",
        "deploy_name": "rust-mentor-8b",
        "description": "Highest quality, A100 required (~4.5GB GGUF)",
    },
    "4b": {
        "model": "unsloth/Qwen3-4B",
        "full_precision_model": "Qwen/Qwen3-4B",
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": 1,
        "grad_accum": 8,
        "output_dir": "models/rust-mentor-4b",
        "deploy_name": "rust-mentor-4b",
        "description": "Lighter, T4 compatible (~2.5GB GGUF)",
    },
    "1.7b": {
        "model": "unsloth/Qwen3-1.7B",
        "full_precision_model": "Qwen/Qwen3-1.7B",
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": 2,
        "grad_accum": 4,
        "output_dir": "models/rust-mentor-1.7b",
        "deploy_name": "rust-mentor-1.7b",
        "description": "Fast on-device chat (~1.1GB GGUF)",
    },
    "0.6b": {
        "model": "unsloth/Qwen3-0.6B",
        "full_precision_model": "Qwen/Qwen3-0.6B",
        "lora_r": 8,
        "lora_alpha": 8,
        "batch_size": 4,
        "grad_accum": 2,
        "output_dir": "models/rust-mentor-0.6b",
        "deploy_name": "rust-mentor-0.6b",
        "description": "Ultra-light, any GPU (~0.4GB GGUF)",
    },
    "gemma3-1b": {
        "model": "unsloth/gemma-3-1b-it",
        "full_precision_model": "google/gemma-3-1b-it",
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": 2,
        "grad_accum": 4,
        "output_dir": "models/rust-mentor-1b-mobile",
        "deploy_name": "rust-mentor-1b-mobile",
        "description": "Gemma3-1B for Google AI Edge Gallery (~650MB .litertlm)",
    },
    "gemma3-4b": {
        "model": "unsloth/gemma-3-4b-it",
        "full_precision_model": "google/gemma-3-4b-it",
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": 2,
        "grad_accum": 4,
        "output_dir": "models/rust-mentor-4b-mobile",
        "deploy_name": "rust-mentor-4b-mobile",
        "description": "Gemma3-4B GGUF for PocketPal AI (~2.5GB GGUF)",
    },
}


# ── Model Name Mappings ───────────────────────────────────
# Unsloth saves bnb-4bit model names in adapter configs.
# For merging we need the full-precision base model to avoid
# rounding errors in fp16 merge.

BNB4BIT_TO_FULL = {
    "unsloth/qwen3-0.6b-unsloth-bnb-4bit": "Qwen/Qwen3-0.6B",
    "unsloth/qwen3-1.7b-unsloth-bnb-4bit": "Qwen/Qwen3-1.7B",
    "unsloth/qwen3-4b-unsloth-bnb-4bit": "Qwen/Qwen3-4B",
    "unsloth/qwen3-8b-unsloth-bnb-4bit": "Qwen/Qwen3-8B",
    "unsloth/Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "unsloth/Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "unsloth/Qwen3-4B": "Qwen/Qwen3-4B",
    "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
    "unsloth/gemma-3-1b-it": "google/gemma-3-1b-it",
    "unsloth/gemma-3-4b-it": "google/gemma-3-4b-it",
}


# ── GGUF Quantization Options ─────────────────────────────

GGUF_QUANT_MAP = {
    "q2_k": "q2_k",
    "q4_k_m": "q4_k_m",
    "q5_k_m": "q5_k_m",
    "q8_0": "q8_0",
    "f16": "f16",
}


# ── LiteRT Configuration ──────────────────────────────────

LITERT_QUANT_OPTIONS = {
    "dynamic_int8": "dynamic_int8",
    "dynamic_int4": "dynamic_int4_block32",
    "fp16": "fp16",
    "none": "none",
}

LITERT_MODEL_SIZES = {
    "0.6b": "0.6b",
    "1.7b": "1.7b",
    "4b": "4b",
}


# ── Stop Token IDs ─────────────────────────────────────────
# Per-model stop tokens for .litertlm bundling.

STOP_TOKENS = {
    "qwen3": [151645, 151643],   # <|im_end|>, <|endoftext|>
    "gemma3": [1, 106],          # <eos>, <end_of_turn>
}


# ── Evaluation Prompts ─────────────────────────────────────
# Used by evaluate_model() to test Rust tutoring quality.

EVAL_PROMPTS = [
    {
        "category": "ownership",
        "prompt": "Explain Rust's ownership model to someone who knows Go.",
        "expected_keywords": ["owner", "move", "borrow", "scope", "drop"],
    },
    {
        "category": "error_handling",
        "prompt": "How do I handle errors in Rust? I'm used to Go's if err != nil pattern.",
        "expected_keywords": ["Result", "Option", "?", "unwrap", "Ok", "Err"],
    },
    {
        "category": "code_review",
        "prompt": (
            "Review this Rust code:\n```rust\n"
            "fn get_longest(a: String, b: String) -> String {\n"
            "    if a.len() > b.len() { a } else { b }\n"
            "}\n```"
        ),
        "expected_keywords": ["borrow", "&str", "reference", "ownership"],
    },
    {
        "category": "traits",
        "prompt": "How are Rust traits different from Go interfaces?",
        "expected_keywords": ["trait", "impl", "explicit", "default", "generic"],
    },
    {
        "category": "cli",
        "prompt": "How do I build a CLI tool in Rust with subcommands?",
        "expected_keywords": ["clap", "derive", "Parser", "Subcommand", "cargo"],
    },
]


# ── Model Card Template ───────────────────────────────────

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
language:
- en
tags:
- rust
- programming
- tutor
- code-generation
- qlora
- unsloth
base_model: {base_model}
datasets:
- Fortytwo-Network/Strandset-Rust-v1
pipeline_tag: text-generation
---

# {model_name}{format_suffix}

Fine-tuned {base_model_short} specialized in **Rust programming education and code review**.

Designed for experienced Go/Python/TypeScript developers learning Rust.
Runs offline on Android devices via PocketPal AI or Google AI Edge Gallery.

## Usage

### PocketPal AI (Android - Offline)
1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Download the GGUF from this repo
3. Load in PocketPal, chat in airplane mode

### Python
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{repo_id}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

## Training Details

- **Base model**: {base_model_short}
- **Method**: QLoRA with Unsloth optimization
- **Dataset**: Strandset-Rust-v1 + synthetic Rust tutor conversations

## Capabilities

- Rust ownership, borrowing, and lifetime explanations
- Error handling patterns (Result, Option, ?)
- Code review with borrow checker explanations
- Pattern matching and enum design
- Trait-based architecture guidance
- CLI tooling with clap
- Comparisons to Go/Python/TypeScript equivalents
"""


# ── Ollama Modelfile Template ──────────────────────────────

OLLAMA_MODELFILE_TEMPLATE = '''FROM {gguf_path}

SYSTEM """{system_prompt}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
'''
