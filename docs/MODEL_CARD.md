---
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
- gguf
- mobile
- offline
base_model: Qwen/Qwen3-8B
datasets:
- Fortytwo-Network/Strandset-Rust-v1
pipeline_tag: text-generation
---

# RustMentor-8B

Fine-tuned Qwen3-8B specialized in **Rust programming education and code review**, designed for experienced Go/Python/TypeScript developers learning Rust.

## Highlights

- **Tutoring Mode**: Explains ownership, borrowing, lifetimes, error handling, traits, async, smart pointers, macros, serde, and more — with Go/Python/TS comparisons
- **Code Review Mode**: Reviews Rust code and explains borrow checker behavior
- **Offline Mobile**: GGUF export runs on Pixel 8 Pro via PocketPal AI
- **CLI Focused**: Guides building Rust CLI tools (clap, cargo, modules)

## Quick Start

### PocketPal AI (Android Offline)
1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Add from Hugging Face → search `sylvester-francis/rust-mentor-8b-GGUF`
3. Download Q4_K_M → chat in airplane mode

### Python (LoRA adapter)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "sylvester-francis/rust-mentor-8b")
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-8B |
| Method | QLoRA (r=32, alpha=32) via Unsloth |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Optimizer | AdamW (lr=2e-4, cosine schedule) |
| Epochs | 3 |
| Effective Batch Size | 8 (2 x 4 grad accum) |
| Max Sequence Length | 2048 |
| Training Data | ~3,500 samples |
| Hardware | NVIDIA A100 40GB (Google Colab) |
| Framework | Unsloth + TRL + PEFT |

## Dataset

| Source | Samples | Content |
|--------|---------|---------|
| Strandset-Rust-v1 | 3,000 | Code generation, review, refactoring, bug detection |
| Synthetic Tutor Q&A | 46 unique conversations across 28 topics | Ownership, error handling, traits, async, smart pointers, macros, serde, collections, closures, testing, and more |

## Limitations

- Not a replacement for the Rust compiler — always verify generated code compiles
- Complex lifetime annotations may not always be correct
- Best at explaining concepts; for production code generation, use larger models
- Trained on Rust 2021 edition patterns; may not cover bleeding-edge nightly features

## License

Apache 2.0 for the fine-tuned model; base model (Qwen3-8B) license also applies.

## Citation

```bibtex
@software{rust_mentor_2026,
  author = {Francis, Sylvester},
  title = {RustMentor-8B: Fine-tuned Rust Programming Tutor},
  year = {2026},
  url = {https://github.com/sylvester-francis/slm-rust-model}
}
```
