#!/usr/bin/env python3
"""
RustMentor SLM - Google Colab Training Pipeline

Automated pipeline: Generate data → Preprocess → Train → Upload (adapter + GGUF)

Supports training four model variants:
- 8B  (Qwen3-8B,   ~4.5GB GGUF) — highest quality, A100 required
- 4B  (Qwen3-4B,   ~2.5GB GGUF) — lighter, T4 compatible
- 1.7B (Qwen3-1.7B, ~1.1GB GGUF) — fast on-device chat, T4/free Colab
- 0.6B (Qwen3-0.6B, ~0.4GB GGUF) — ultra-light, instant mobile responses

Set TRAIN_VARIANTS to control which variants to train:
  "8b" | "4b" | "1.7b" | "0.6b" | "all" | "mobile" (1.7b+0.6b) | "both" (8b+4b)

Requirements:
- Google Colab Pro with A100 GPU (40GB VRAM) for 8B; T4 or free Colab for smaller variants
- HF_TOKEN in Colab Secrets (🔑 icon in left sidebar)

Usage in Colab:
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    !python colab/colab_train_and_upload.py
"""

import os
import sys
import time
import json

# ──────────────────────────────────────────────
# CONFIGURATION — Edit these to customize
# ──────────────────────────────────────────────

# Which variants to train: "8b", "4b", "1.7b", "0.6b", "all", "mobile" (1.7b+0.6b), "both" (8b+4b)
TRAIN_VARIANTS = "mobile"

# 8B Model Config (A100 required, ~4.5GB GGUF)
CONFIG_8B = {
    "base_model": "unsloth/Qwen3-8B",
    "lora_r": 32,
    "lora_alpha": 32,
    "batch_size": 2,
    "grad_accum": 4,
    "output_dir": "models/rust-mentor-8b",
    "repo_name": "rust-mentor-8b",
    "base_model_name": "Qwen/Qwen3-8B",
    "param_count": "8B",
    "gguf_size": "~4.5GB",
}

# 4B Model Config (T4 compatible, ~2.5GB GGUF)
CONFIG_4B = {
    "base_model": "unsloth/Qwen3-4B",
    "lora_r": 16,
    "lora_alpha": 16,
    "batch_size": 1,
    "grad_accum": 8,
    "output_dir": "models/rust-mentor-4b",
    "repo_name": "rust-mentor-4b",
    "base_model_name": "Qwen/Qwen3-4B",
    "param_count": "4B",
    "gguf_size": "~2.5GB",
}

# 1.7B Model Config (T4/free Colab, ~1.1GB GGUF — fast on-device chat)
CONFIG_1_7B = {
    "base_model": "unsloth/Qwen3-1.7B",
    "lora_r": 16,
    "lora_alpha": 16,
    "batch_size": 2,
    "grad_accum": 4,
    "output_dir": "models/rust-mentor-1.7b",
    "repo_name": "rust-mentor-1.7b",
    "base_model_name": "Qwen/Qwen3-1.7B",
    "param_count": "1.7B",
    "gguf_size": "~1.1GB",
}

# 0.6B Model Config (any GPU, ~0.4GB GGUF — ultra-light quick exercise debugs)
CONFIG_0_6B = {
    "base_model": "unsloth/Qwen3-0.6B",
    "lora_r": 8,
    "lora_alpha": 8,
    "batch_size": 4,
    "grad_accum": 2,
    "output_dir": "models/rust-mentor-0.6b",
    "repo_name": "rust-mentor-0.6b",
    "base_model_name": "Qwen/Qwen3-0.6B",
    "param_count": "0.6B",
    "gguf_size": "~0.4GB",
}


# Shared config
MAX_SEQ_LENGTH = 2048
EPOCHS = 3
LEARNING_RATE = 2e-4

# Dataset
SYNTHETIC_SAMPLES = 500
STRANDSET_SAMPLES = 3000

# Export
GGUF_QUANT = "q4_k_m"  # q4_k_m for mobile, q5_k_m for higher quality

# HuggingFace
HF_USERNAME = ""  # Set below from secrets or manually

# ──────────────────────────────────────────────


def setup_environment():
    """Install dependencies and verify GPU."""
    print("\n" + "=" * 60)
    print("🦀 RustMentor SLM — Colab Training Pipeline")
    print("=" * 60)

    # Install deps
    print("\n📦 Installing dependencies...")
    os.system("pip install -q unsloth trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer")
    os.system("pip install -q 'transformers>=4.51.3,<=5.2.0'")

    # Verify GPU
    import torch
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Go to Runtime → Change runtime type → A100 GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU: {gpu_name} ({vram:.1f} GB)")

    if vram < 35 and TRAIN_VARIANTS in ("8b", "both", "all"):
        print("⚠️  A100 (40GB) recommended for 8B model. T4 may OOM.")
        print("   Set TRAIN_VARIANTS = 'mobile' for 1.7B+0.6B, or '4b' for 4B only.")

    # Load HF token — must be set as env var before running the script:
    #   In a Colab cell: import os; from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    #   Then:            !python colab/colab_train_and_upload.py
    global HF_USERNAME
    hf_token = os.environ.get("HF_TOKEN", "")

    if hf_token:
        print("✅ HF_TOKEN loaded")
        # Get username from token
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            HF_USERNAME = api.whoami()["name"]
            print(f"✅ HF Username: {HF_USERNAME}")
        except Exception:
            print("⚠️  Could not determine HF username. Set HF_USERNAME manually.")
    else:
        print("⚠️  HF_TOKEN not found. Upload step will be skipped.")
        print("   Add it in Colab: 🔑 Secrets → HF_TOKEN → your token")


def step_generate_data():
    """Step 1: Generate synthetic Rust tutor data."""
    print("\n" + "─" * 60)
    print("  Step 1/4: Generating Synthetic Rust Tutor Data")
    print("─" * 60)

    # Add project root to path
    sys.path.insert(0, os.getcwd())
    from scripts.data_collection import generate_rust_dataset

    os.makedirs("data/processed", exist_ok=True)
    count = generate_rust_dataset(
        output_path="data/processed/rust_tutor_synthetic.jsonl",
        num_samples=SYNTHETIC_SAMPLES,
        system_prompt="""You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools.

Your teaching style:
- Draw parallels to Go/Python/TypeScript concepts they already know
- Explain ownership, borrowing, and lifetimes with practical examples
- When reviewing code, explain what the borrow checker is doing and why
- Keep explanations concise with code snippets
- Guide them to write the code themselves rather than giving full solutions
""",
    )
    print(f"  ✅ Generated {count} synthetic samples")


def step_preprocess():
    """Step 2: Preprocess and merge datasets."""
    print("\n" + "─" * 60)
    print("  Step 2/4: Preprocessing & Merging Datasets")
    print("─" * 60)

    from scripts.data_preprocessing import preprocess_and_merge

    count = preprocess_and_merge(
        synthetic_path="data/processed/rust_tutor_synthetic.jsonl",
        strandset_samples=STRANDSET_SAMPLES,
        output_path="data/processed/train.jsonl",
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print(f"  ✅ Merged dataset: {count} samples")


def step_train_variant(config):
    """Train a single model variant with QLoRA."""
    name = config["repo_name"]
    print(f"\n  Training {name} ({config['base_model']})...")

    from scripts.training import train_model

    stats = train_model(
        base_model=config["base_model"],
        data_path="data/processed/train.jsonl",
        output_dir=config["output_dir"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        batch_size=config["batch_size"],
        grad_accum=config["grad_accum"],
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print(f"  ✅ {name} training complete (loss: {stats.metrics['train_loss']:.4f})")


def step_train():
    """Step 3: Fine-tune variant(s) with QLoRA."""
    variants = _get_variants()
    variant_names = " + ".join(c["repo_name"] for c in variants)
    print("\n" + "─" * 60)
    print(f"  Step 3/4: Training {variant_names} with QLoRA + Unsloth")
    print("─" * 60)

    for config in variants:
        step_train_variant(config)


def _get_variants():
    """Return list of model configs to train/upload based on TRAIN_VARIANTS."""
    variant_map = {
        "8b": [CONFIG_8B],
        "4b": [CONFIG_4B],
        "1.7b": [CONFIG_1_7B],
        "0.6b": [CONFIG_0_6B],
        "both": [CONFIG_8B, CONFIG_4B],
        "mobile": [CONFIG_1_7B, CONFIG_0_6B],
        "all": [CONFIG_8B, CONFIG_4B, CONFIG_1_7B, CONFIG_0_6B],
    }
    configs = variant_map.get(TRAIN_VARIANTS)
    if configs is None:
        print(f"⚠️  Unknown TRAIN_VARIANTS '{TRAIN_VARIANTS}', defaulting to mobile (1.7b+0.6b)")
        return [CONFIG_1_7B, CONFIG_0_6B]
    return configs


ADAPTER_MODEL_CARD = """---
license: apache-2.0
language:
- en
tags:
- rust
- programming
- tutor
- code-review
- code-generation
- qlora
- unsloth
- lora
base_model: {base_model_name}
datasets:
- Fortytwo-Network/Strandset-Rust-v1
pipeline_tag: text-generation
---

# RustMentor-{param_count}

RustMentor-{param_count} is a {param_count}-parameter Qwen3-based model fine-tuned for Rust programming education and code review. It bridges concepts from Go, Python, and TypeScript to teach Rust through practical examples and Socratic dialogue.

This repository hosts the **LoRA adapter** weights. For quantized local inference, see [rust-mentor-{param_count_lower}-GGUF](https://huggingface.co/{username}/rust-mentor-{param_count_lower}-GGUF).

## Model Description

- **Base Model**: {base_model_name}
- **Model Type**: Causal LM (code tutoring + review)
- **Parameters**: {param_count}
- **Context Length**: 2048 tokens
- **Fine-tuning**: QLoRA (r={lora_r}, alpha={lora_r}) with Unsloth optimization
- **License**: Apache 2.0
- **Language**: English, Rust code
- **System Prompt**: Rust programming tutor for experienced Go/Python/TypeScript developers learning Rust by building CLI tools.

## What It Is Good At

- Explaining Rust ownership, borrowing, and lifetimes with Go/Python/TS comparisons
- Code review with borrow checker explanations
- Error handling patterns (Result, Option, ?, thiserror, anyhow)
- Async/await and Tokio patterns
- Smart pointers (Box, Rc, Arc, RefCell)
- Pattern matching and enum-based design
- Trait-based architecture and generics
- Type conversions (From, Into, AsRef, Deref)
- Serde & JSON serialization
- CLI tooling with clap
- Cargo project structure, modules, and workspaces
- Testing patterns and documentation

## Intended Uses

**Primary**: Rust programming tutoring, debugging, code review, and guided learning for developers transitioning from Go/Python/TypeScript.

**Out-of-scope**: General-purpose chat, non-Rust programming, safety-sensitive or factual tasks outside Rust development.

## Prompt Examples

```
"In Go, I just pass values or pointers. What's this ownership thing in Rust?"

"Review this Rust code and explain what the borrow checker is doing:\\n\\nfn get_longest(a: String, b: String) -> String {{\\n    if a.len() > b.len() {{ a }} else {{ b }}\\n}}"

"How do I handle errors in Rust? I'm used to Go's if err != nil pattern."

"How does async work in Rust? In Go I just use goroutines and it's simple."
```

## How to Use

### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{username}/rust-mentor-{param_count_lower}",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{username}/rust-mentor-{param_count_lower}")

messages = [
    {{"role": "system", "content": "You are RustMentor, an expert Rust programming tutor."}},
    {{"role": "user", "content": "Explain Rust's ownership model to someone who knows Go."}},
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training Data (Summary)

- **Strandset-Rust-v1**: 3,000 samples of Rust code generation, review, refactoring, and bug detection tasks
- **Synthetic tutor conversations**: 46 unique hand-crafted Rust tutoring dialogues across 28 topics, covering ownership, error handling, traits, async, smart pointers, macros, serde, testing, and more
- **Style**: All conversations draw parallels to Go/Python/TypeScript equivalents

## Training Configuration (QLoRA)

| Parameter | Value |
|-----------|-------|
| Base Model | {base_model_name} |
| Method | QLoRA via Unsloth |
| LoRA Rank (r) | {lora_r} |
| LoRA Alpha | {lora_r} |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Batch Size | {batch_size} x {grad_accum} (effective {effective_batch}) |
| Learning Rate | 2e-4 (cosine schedule) |
| Max Sequence Length | 2048 |
| Hardware | NVIDIA A100 40GB (Google Colab) |

## Evaluation

Qualitative checks on Rust tutoring prompts show:

- Clear explanations with Go/Python/TypeScript comparisons
- Accurate code examples with proper ownership and borrowing
- Borrow checker explanations in code reviews
- Appropriate use of idiomatic Rust patterns

## Safety & Limitations

- May generate incorrect code or hallucinate crate APIs — review before production use.
- Not a replacement for the Rust compiler or clippy — always compile and test generated code.
- Optimized for tutoring, not production code generation at scale.
- Training data focuses on CLI/systems patterns; web framework coverage (Axum, Actix) is limited.

## License

Apache 2.0 for the fine-tuned adapter; base model ({base_model_name}) license also applies.

## Contact

- **Maintainer**: Sylvester Francis ([@sylvester-francis](https://huggingface.co/{username}))
- **Repository**: [github.com/sylvester-francis/slm-rust-model](https://github.com/sylvester-francis/slm-rust-model)
- **Issues/feedback**: Open a discussion on the model repo
"""

GGUF_MODEL_CARD = """---
license: apache-2.0
language:
- en
tags:
- rust
- programming
- tutor
- code-review
- code-generation
- qlora
- unsloth
- gguf
base_model: {base_model_name}
datasets:
- Fortytwo-Network/Strandset-Rust-v1
pipeline_tag: text-generation
---

# RustMentor-{param_count}-GGUF

RustMentor-{param_count}-GGUF is a {param_count}-parameter Qwen3-based model fine-tuned for Rust programming education and code review. It merges the base model with LoRA adapters and includes GGUF quantization for local/mobile/Ollama workflows.

This repository hosts the **GGUF quantized model** (Q4_K_M) for lightweight inference. For the LoRA adapter, see [rust-mentor-{param_count_lower}](https://huggingface.co/{username}/rust-mentor-{param_count_lower}).

## Model Description

- **Base Model**: {base_model_name}
- **Model Type**: Causal LM (code tutoring + review)
- **Parameters**: {param_count}
- **Context Length**: 2048 tokens
- **Fine-tuning**: QLoRA (r={lora_r}, alpha={lora_r}) with Unsloth optimization
- **Quantization**: Q4_K_M ({gguf_size})
- **License**: Apache 2.0
- **Language**: English, Rust code
- **System Prompt**: Rust programming tutor for experienced Go/Python/TypeScript developers learning Rust by building CLI tools.

## What It Is Good At

- Explaining Rust ownership, borrowing, and lifetimes with Go/Python/TS comparisons
- Code review with borrow checker explanations
- Error handling patterns (Result, Option, ?, thiserror, anyhow)
- Async/await and Tokio patterns
- Smart pointers (Box, Rc, Arc, RefCell)
- Pattern matching and enum-based design
- Trait-based architecture and generics
- Type conversions (From, Into, AsRef, Deref)
- Serde & JSON serialization
- CLI tooling with clap
- Cargo project structure, modules, and workspaces
- Testing patterns and documentation

## Intended Uses

**Primary**: Offline Rust programming tutor on Android (Pixel 8 Pro tested) via PocketPal AI, or local inference via Ollama/llama.cpp.

**Out-of-scope**: General-purpose chat, non-Rust programming, safety-sensitive or factual tasks outside Rust development.

## Prompt Examples

```
"In Go, I just pass values or pointers. What's this ownership thing in Rust?"

"Review this Rust code and explain what the borrow checker is doing:\\n\\nfn get_longest(a: String, b: String) -> String {{\\n    if a.len() > b.len() {{ a }} else {{ b }}\\n}}"

"How do I handle errors in Rust? I'm used to Go's if err != nil pattern."

"How does async work in Rust? In Go I just use goroutines and it's simple."
```

## How to Use

### PocketPal AI (Android — Offline)

1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai) from Play Store
2. Tap "Add from Hugging Face"
3. Search: `{username}/rust-mentor-{param_count_lower}-GGUF`
4. Download the Q4_K_M quantization ({gguf_size})
5. Create a "Pal" with the Rust tutor system prompt
6. Enable airplane mode and start learning!

### Ollama (Local)

```bash
# Download the GGUF
huggingface-cli download {username}/rust-mentor-{param_count_lower}-GGUF \\
  --local-dir ./models/rust-mentor

# Create Modelfile
cat > Modelfile << 'MODELFILE'
FROM ./models/rust-mentor/<gguf-filename>.gguf

SYSTEM \"\"\"You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools. Draw parallels to Go/Python/TypeScript concepts. Explain ownership, borrowing, and lifetimes with practical examples.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
MODELFILE

ollama create rust-mentor -f Modelfile
ollama run rust-mentor "Explain Rust's ownership vs Go's garbage collector"
```

### llama.cpp

```bash
huggingface-cli download {username}/rust-mentor-{param_count_lower}-GGUF \\
  --local-dir ./models

./llama-cli -m ./models/<gguf-filename>.gguf \\
  -p "Explain and fix this Rust borrow checker error..."
```

## Training Data (Summary)

- **Strandset-Rust-v1**: 3,000 samples of Rust code generation, review, refactoring, and bug detection tasks
- **Synthetic tutor conversations**: 46 unique hand-crafted Rust tutoring dialogues across 28 topics, covering ownership, error handling, traits, async, smart pointers, macros, serde, testing, and more
- **Style**: All conversations draw parallels to Go/Python/TypeScript equivalents

## Training Configuration (QLoRA)

| Parameter | Value |
|-----------|-------|
| Base Model | {base_model_name} |
| Method | QLoRA via Unsloth |
| LoRA Rank (r) | {lora_r} |
| LoRA Alpha | {lora_r} |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Batch Size | {batch_size} x {grad_accum} (effective {effective_batch}) |
| Learning Rate | 2e-4 (cosine schedule) |
| Max Sequence Length | 2048 |
| Hardware | NVIDIA A100 40GB (Google Colab) |

## Evaluation

Qualitative checks on Rust tutoring prompts show:

- Clear explanations with Go/Python/TypeScript comparisons
- Accurate code examples with proper ownership and borrowing
- Borrow checker explanations in code reviews
- Appropriate use of idiomatic Rust patterns

## Safety & Limitations

- May generate incorrect code or hallucinate crate APIs — review before production use.
- Not a replacement for the Rust compiler or clippy — always compile and test generated code.
- Optimized for tutoring, not production code generation at scale.
- Training data focuses on CLI/systems patterns; web framework coverage (Axum, Actix) is limited.

## License

Apache 2.0 for the fine-tuned model; base model ({base_model_name}) license also applies.

## Contact

- **Maintainer**: Sylvester Francis ([@sylvester-francis](https://huggingface.co/{username}))
- **Repository**: [github.com/sylvester-francis/slm-rust-model](https://github.com/sylvester-francis/slm-rust-model)
- **Issues/feedback**: Open a discussion on the model repo
"""


def upload_model_card(repo_id, token, card_template, config, username):
    """Upload a model card README.md to a HuggingFace repo."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    card = card_template.format(
        username=username,
        param_count=config["param_count"],
        param_count_lower=config["param_count"].lower(),
        base_model_name=config["base_model_name"],
        lora_r=config["lora_r"],
        batch_size=config["batch_size"],
        grad_accum=config["grad_accum"],
        effective_batch=config["batch_size"] * config["grad_accum"],
        gguf_size=config["gguf_size"],
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )


def step_upload_variant(config):
    """Upload a single variant's adapter + GGUF to HuggingFace."""
    from unsloth import FastLanguageModel
    from huggingface_hub import create_repo

    token = os.environ.get("HF_TOKEN", "")
    name = config["repo_name"]

    # Load the trained model
    print(f"\n  Loading trained model ({name})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["output_dir"],
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Upload LoRA adapter
    repo_id = f"{HF_USERNAME}/{name}"
    print(f"  Uploading adapter → {repo_id}")
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    except Exception:
        pass
    model.push_to_hub(repo_id, token=token)
    tokenizer.push_to_hub(repo_id, token=token)
    upload_model_card(repo_id, token, ADAPTER_MODEL_CARD, config, HF_USERNAME)
    print(f"  ✅ {name} adapter + model card uploaded")

    # Push GGUF directly to HF
    gguf_repo = f"{HF_USERNAME}/{name}-GGUF"
    print(f"  Pushing GGUF directly → {gguf_repo} (skips local save)")
    try:
        create_repo(gguf_repo, token=token, exist_ok=True, repo_type="model")
    except Exception:
        pass
    model.push_to_hub_gguf(
        gguf_repo,
        tokenizer,
        quantization_method=GGUF_QUANT,
        token=token,
    )
    upload_model_card(gguf_repo, token, GGUF_MODEL_CARD, config, HF_USERNAME)
    print(f"  ✅ {name} GGUF + model card pushed")

    # Upload LiteRT if available
    litert_dir = config["output_dir"] + "-litert"
    if os.path.exists(litert_dir):
        from huggingface_hub import HfApi
        litert_repo = f"{HF_USERNAME}/{name}-LiteRT"
        print(f"  Uploading LiteRT → {litert_repo}")
        try:
            create_repo(litert_repo, token=token, exist_ok=True, repo_type="model")
        except Exception:
            pass
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=litert_dir,
            repo_id=litert_repo,
            token=token,
        )
        print(f"  ✅ {name} LiteRT pushed")
    else:
        print(f"  ⏭️  No LiteRT output found at {litert_dir}, skipping LiteRT upload")

    print(f"  🔗 Model:  https://huggingface.co/{repo_id}")
    print(f"  🔗 GGUF:   https://huggingface.co/{gguf_repo}")
    if os.path.exists(litert_dir):
        print(f"  🔗 LiteRT: https://huggingface.co/{litert_repo}")


def step_convert_litert_variant(config):
    """Convert a single variant to LiteRT (.tflite) format."""
    name = config["repo_name"]
    variant = config["param_count"].lower()

    print(f"\n  Converting {name} to LiteRT (.tflite)...")

    from scripts.convert_litert import convert_to_litert

    output = convert_to_litert(
        model_dir=config["output_dir"],
        variant=variant,
        quantization="dynamic_int8",
        kv_cache_max_len=MAX_SEQ_LENGTH,
    )
    if output:
        print(f"  ✅ {name} LiteRT exported")
    else:
        print(f"  ⚠️  {name} LiteRT conversion failed (GGUF still available)")


def step_convert_litert():
    """Step 4: Convert all variants to LiteRT format."""
    variants = _get_variants()
    variant_names = " + ".join(c["repo_name"] for c in variants)
    print("\n" + "─" * 60)
    print(f"  Step 4/5: Converting {variant_names} to LiteRT")
    print("─" * 60)

    # Install litert-torch if needed
    print("  Installing litert-torch...")
    os.system("pip install -q litert-torch")

    for config in variants:
        try:
            step_convert_litert_variant(config)
        except Exception as e:
            print(f"  ⚠️  LiteRT conversion failed for {config['repo_name']}: {e}")
            print(f"      GGUF export will still be available in the upload step.")


def step_upload():
    """Step 5: Upload all variant(s) to HuggingFace."""
    print("\n" + "─" * 60)
    print("  Step 5/5: Uploading to HuggingFace")
    print("─" * 60)

    if not HF_USERNAME:
        print("  ⏭️  Skipping upload (no HF credentials)")
        return

    variants = _get_variants()
    for config in variants:
        step_upload_variant(config)


def main():
    """Run complete pipeline."""
    total_start = time.time()

    setup_environment()

    steps = [
        ("Generate data", step_generate_data),
        ("Preprocess", step_preprocess),
        ("Train", step_train),
        ("Convert LiteRT", step_convert_litert),
        ("Upload + GGUF", step_upload),
    ]

    timings = {}
    for name, func in steps:
        start = time.time()
        try:
            func()
            timings[name] = time.time() - start
        except Exception as e:
            print(f"\n  ❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            timings[name] = f"FAILED: {e}"

    # Summary
    total = time.time() - total_start
    print("\n" + "=" * 60)
    print("🦀 Pipeline Complete!")
    print("=" * 60)
    for name, duration in timings.items():
        if isinstance(duration, float):
            print(f"  {name}: {duration:.1f}s")
        else:
            print(f"  {name}: {duration}")
    print(f"\n  Total: {total:.1f}s ({total/60:.1f} min)")

    if HF_USERNAME:
        print(f"\n  📱 Next: Download the GGUF from HuggingFace to your Pixel 8 Pro")
        print(f"     → PocketPal AI → Add from Hugging Face")
        for config in _get_variants():
            name = config["repo_name"]
            print(f"     → {HF_USERNAME}/{name}-GGUF ({config['gguf_size']})")

    print()


if __name__ == "__main__":
    main()
