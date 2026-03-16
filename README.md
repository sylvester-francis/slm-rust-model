# RustMentor SLM — Train Your Own Rust Programming Tutor

Fine-tuned small language models specialized in Rust programming education. Built on Qwen3 and Gemma3 with QLoRA via Unsloth, exported to **GGUF** (llama.cpp/Ollama) and **LiteRT** (.tflite, Android GPU/NPU) for offline mobile inference.

## Tutorial: Train a Rust Tutor in 5 Steps

Each step is a standalone script. Run them in order:

```bash
# Step 1: Generate training data (no GPU needed)
python tutorials/01_data_preparation.py

# Step 2: Fine-tune with QLoRA (needs GPU — use Colab for free T4)
python tutorials/02_fine_tuning.py --variant 0.6b

# Step 3: Evaluate on Rust tutoring prompts
python tutorials/03_evaluation.py --variant 0.6b

# Step 4: Export to GGUF or LiteRT
python tutorials/04_export.py --variant 0.6b

# Step 5: Deploy to Ollama or HuggingFace
python tutorials/05_deploy.py --variant 0.6b --target ollama
```

Or run the full pipeline in one command:

```bash
python slm.py pipeline --variant 0.6b --username your-hf-user
```

## Available Models

### Qwen3 (GGUF + LiteRT)

| Variant | Base Model | GGUF Size | GPU to Train | Best For |
|---------|-----------|-----------|-------------|----------|
| **0.6B** | Qwen3-0.6B | ~0.4GB | Any (free Colab) | Ultra-light mobile, instant responses |
| **1.7B** | Qwen3-1.7B | ~1.1GB | T4/free Colab | On-device chat, exercise debugging |
| **4B** | Qwen3-4B | ~2.5GB | T4 | Balanced quality/speed |
| **8B** | Qwen3-8B | ~4.5GB | A100 | Highest quality tutoring |

### Gemma3 (Mobile-First)

| Variant | Base Model | Output | GPU to Train | Best For |
|---------|-----------|--------|-------------|----------|
| **gemma3-1b** | Gemma3-1B-IT | ~650MB .litertlm | T4 | Google AI Edge Gallery, full GPU on Pixel 8 Pro |
| **gemma3-4b** | Gemma3-4B-IT | ~2.5GB GGUF | A100 | PocketPal AI, higher quality mobile |

## CLI Reference

```bash
python slm.py info                                    # Show system info + installed packages
python slm.py collect                                 # Step 1: Generate synthetic training data
python slm.py preprocess                              # Step 2: Merge datasets
python slm.py train --variant 1.7b                    # Step 3: Train with QLoRA
python slm.py evaluate --variant 0.6b                 # Step 4: Evaluate model quality
python slm.py convert --variant 0.6b                  # Step 5a: Export to GGUF
python slm.py convert-litert --variant 0.6b           # Step 5b: Export to LiteRT
python slm.py upload --variant 0.6b --username u --gguf    # Step 6: Upload to HuggingFace
python slm.py deploy --variant 0.6b                   # Step 7: Deploy to Ollama
```

## Quick Start: Google Colab

```python
# 1. Clone repository
!git clone https://github.com/sylvester-francis/slm-rust-model.git
%cd slm-rust-model

# 2. Set HF_TOKEN in Colab Secrets
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# 3a. Gemma3-1B mobile pipeline (recommended)
!python colab/colab_gemma3_pipeline.py

# 3b. Or Qwen3 variant
!python slm.py pipeline --variant 0.6b --username your-username
```

## Mobile Deployment

**Option A: Google AI Edge Gallery (Gemma3, GPU accelerated)**

1. Train with `colab/colab_gemma3_pipeline.py`
2. Install [Google AI Edge Gallery](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery)
3. Import the .litertlm model from HuggingFace
4. Full GPU delegation on Pixel 8 Pro (Tensor G3)

**Option B: PocketPal AI (GGUF, CPU-based)**

1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Search: `sylvester-francis/rust-mentor-0.6b-GGUF`
3. Download and chat offline

**Option C: Ollama (Desktop)**

```bash
python slm.py deploy --variant 0.6b
ollama run rust-mentor-0.6b
```

## Export Formats

| Format | Engine | Android Acceleration | Use Case |
|--------|--------|---------------------|----------|
| **GGUF** (Q4_K_M) | llama.cpp | CPU | Ollama, PocketPal AI, llama.cpp |
| **LiteRT** (.tflite) | litert-torch | GPU/NPU via NNAPI | Google AI Edge Gallery |

LiteRT provides **2-3x faster inference** on Pixel 8 Pro compared to GGUF by leveraging the Tensor G3 GPU/NPU.

## Features

- **Rust-Specialized Tutoring**: Ownership, borrowing, lifetimes, error handling, traits, pattern matching
- **Go/Python/TypeScript Comparisons**: Bridges concepts from languages you already know
- **Code Review Mode**: Explains borrow checker behavior on your code
- **CLI Project Guidance**: Cargo, clap, project structure patterns
- **Dual Export**: GGUF + LiteRT from the same training run
- **6 Variants**: From 0.6B (instant) to 8B (highest quality), plus Gemma3 mobile
- **Works Out of the Box**: Graceful error messages for missing dependencies, clear step-by-step flow

## Training Data

| Source | Type | Samples | Content |
|--------|------|---------|---------|
| Synthetic Tutor | Teaching Q&A | 500 | 46 hand-written conversations across 28 Rust topics |
| Strandset-Rust-v1 | Code tasks | 3,000 | Code generation, review, refactoring from HuggingFace |

Topics covered: ownership, borrowing, lifetimes, error handling (Result/Option/?), traits, generics, pattern matching, iterators, closures, async/await, smart pointers, macros, serde, testing, CLI tooling (clap), cargo, modules, unsafe Rust, and more.

## Project Structure

```
slm-rust-model/
├── slm.py                         # Unified CLI (dispatches to rustmentor/)
│
├── rustmentor/                    # Main Python package
│   ├── config.py                  # Shared config: prompts, variants, constants
│   ├── data/
│   │   ├── collection.py          # Synthetic data generation (46 seed conversations)
│   │   └── preprocessing.py       # Dataset merging & formatting
│   ├── training/
│   │   ├── trainer.py             # QLoRA fine-tuning with Unsloth
│   │   └── evaluation.py          # Keyword-match evaluation
│   ├── export/
│   │   ├── gguf.py                # GGUF export (Unsloth + llama.cpp for Gemma3)
│   │   ├── litert.py              # LiteRT export (Qwen3 + Gemma3 support)
│   │   └── bundle.py             # Bundle .tflite + tokenizer → .litertlm
│   └── deploy/
│       ├── huggingface.py         # HuggingFace Hub upload with model card
│       └── ollama.py              # Ollama deployment
│
├── tutorials/                     # Step-by-step tutorial scripts
│   ├── 01_data_preparation.py     # Generate & merge training data
│   ├── 02_fine_tuning.py          # Train with QLoRA (needs GPU)
│   ├── 03_evaluation.py           # Evaluate model quality
│   ├── 04_export.py               # Export to GGUF or LiteRT
│   └── 05_deploy.py               # Deploy to Ollama or HuggingFace
│
├── colab/                         # Google Colab pipelines
│   ├── colab_gemma3_pipeline.py   # Gemma3-1B: train → LiteRT → upload
│   └── colab_gemma3_4b_pipeline.py # Gemma3-4B: train → GGUF → upload
│
├── data/processed/                # Training datasets (gitignored)
└── models/                        # Trained models (gitignored)
```

## Hardware Requirements

| Variant | Colab GPU | Training Time | VRAM |
|---------|-----------|---------------|------|
| 0.6B | Any (free) | ~10 min | ~4GB |
| 1.7B | T4/free | ~20 min | ~8GB |
| 4B | T4 | ~45 min | ~14GB |
| 8B | A100 | ~60 min | ~32GB |

LiteRT conversion requires ~32GB RAM (use high-RAM Colab runtime).

## Troubleshooting

**CUDA Out of Memory**

```bash
# Use a smaller variant
python slm.py train --variant 0.6b
```

**Missing Dependencies**

```bash
# The CLI will tell you exactly what's missing. Install with:
pip install unsloth trl peft accelerate bitsandbytes datasets torch
pip install huggingface_hub hf_transfer   # for upload
pip install litert-torch                  # for LiteRT export
```

**LiteRT Conversion Fails**

```bash
# Ensure you have 32GB+ RAM and litert-torch installed
pip install litert-torch
# Convert to GGUF instead as fallback
python slm.py convert --variant 0.6b
```

**Strandset Dataset Not Loading**

```bash
# Train with synthetic data only
python slm.py preprocess --strandset-samples 0
```

## HuggingFace Repos

Each variant is uploaded in up to three formats:

- `sylvester-francis/rust-mentor-{variant}` — LoRA adapter weights
- `sylvester-francis/rust-mentor-{variant}-GGUF` — GGUF quantized (Q4_K_M)
- `sylvester-francis/rust-mentor-{variant}-LiteRT` — LiteRT .tflite for Android

## Citation

```bibtex
@software{rust_mentor_2026,
  author = {Francis, Sylvester},
  title = {RustMentor SLM: Fine-tuned Small Language Models for Rust Programming},
  year = {2026},
  url = {https://github.com/sylvester-francis/slm-rust-model}
}
```

## Acknowledgments

- Built on [Qwen3](https://huggingface.co/Qwen) by Alibaba Cloud and [Gemma3](https://huggingface.co/google/gemma-3-1b-it) by Google
- Rust code data from [Strandset-Rust-v1](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1)
- Training optimized by [Unsloth](https://github.com/unslothai/unsloth)
- LiteRT export via [litert-torch](https://github.com/google-ai-edge/litert-torch) by Google
- Training powered by [Hugging Face TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft)

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.
