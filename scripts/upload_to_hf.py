"""Upload model to HuggingFace Hub."""

import os
from pathlib import Path


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
base_model: Qwen/Qwen3-8B
datasets:
- Fortytwo-Network/Strandset-Rust-v1
pipeline_tag: text-generation
---

# RustMentor-8B{gguf_suffix}

Fine-tuned Qwen3-8B specialized in **Rust programming education and code review**.

Designed for experienced Go/Python/TypeScript developers learning Rust. Runs offline on Android devices (Pixel 8 Pro tested) via PocketPal AI.

## Usage

### PocketPal AI (Android - Offline)
1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Download the Q4_K_M GGUF from this repo
3. Load in PocketPal, chat in airplane mode

### Ollama
```bash
ollama run your-username/rust-mentor-8b
```

### Python
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-username/rust-mentor-8b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("your-username/rust-mentor-8b")
```

## Training Details

- **Base model**: Qwen3-8B
- **Method**: QLoRA (r=32) with Unsloth optimization
- **Dataset**: Strandset-Rust-v1 + synthetic Rust tutor conversations
- **Hardware**: A100 40GB (Google Colab Pro)
- **Training time**: ~45-60 minutes

## Capabilities

- Rust ownership, borrowing, and lifetime explanations
- Error handling patterns (Result, Option, ?)
- Code review with borrow checker explanations
- Pattern matching and enum design
- Trait-based architecture guidance
- CLI tooling with clap
- Cargo project structure
- Comparisons to Go/Python/TypeScript equivalents

## Citation

```bibtex
@software{{rust_mentor_2026,
  author = {{Francis, Sylvester}},
  title = {{RustMentor-8B: Fine-tuned Rust Programming Tutor}},
  year = {{2026}},
  url = {{https://github.com/sylvester-francis/slm-rust-model}}
}}
```
"""


def upload_model(
    model_dir: str = "models/rust-mentor-8b",
    repo_id: str = "your-username/rust-mentor-8b",
    gguf: bool = False,
):
    """Upload model to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if not token:
        raise ValueError("HF_TOKEN not set. Set via environment or Colab secrets.")

    api = HfApi(token=token)

    # Create repo
    gguf_suffix = "-GGUF" if gguf else ""
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"  Repository: {repo_id}")
    except Exception as e:
        print(f"  Repo exists or error: {e}")

    # Write model card
    card_content = MODEL_CARD_TEMPLATE.format(gguf_suffix=gguf_suffix)
    card_path = os.path.join(model_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card_content)

    # Upload
    if gguf:
        gguf_dir = model_dir + "-GGUF" if not model_dir.endswith("-GGUF") else model_dir
        if os.path.exists(gguf_dir):
            model_dir = gguf_dir

    print(f"  Uploading {model_dir} → {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        token=token,
    )
    print(f"  ✅ Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    upload_model()
