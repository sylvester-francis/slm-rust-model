# 🦀 RustMentor SLM — Small Language Models for Rust Programming

Fine-tuned small language models specialized in Rust programming education and code review. Built on Qwen3-8B with QLoRA fine-tuning, optimized for offline use on Android devices.

## NEW: Unified Professional CLI

```bash
# Complete training pipeline
python slm.py pipeline --username your-username

# Individual steps
python slm.py collect           # Generate synthetic training data
python slm.py preprocess        # Merge Strandset + synthetic data
python slm.py train             # Train model with QLoRA
python slm.py evaluate          # Evaluate model quality
python slm.py convert           # Export to GGUF for mobile
python slm.py upload --username your-username --gguf  # Upload to HF
python slm.py deploy            # Deploy to Ollama
```

**Documentation:**
- **[Quick Start Guide](docs/QUICK_START.md)** — Get started in minutes
- **[Colab Guide](docs/COLAB.md)** — Complete Colab training instructions
- **[Mobile Deployment](docs/MOBILE.md)** — PocketPal AI setup for Pixel 8 Pro

## Overview

This project provides a Rust programming tutor model designed for developers transitioning from Go, Python, or TypeScript. The model is trained on high-quality Rust code examples and tutoring conversations, then exported to GGUF format for offline mobile inference.

### Available Models

| Model | Size | Context | Best For | HuggingFace |
|-------|------|---------|----------|-------------|
| RustMentor 8B | 8B params | 2048 tokens | Teaching + code review | [sylvester-francis/rust-mentor-8b](https://huggingface.co/sylvester-francis/rust-mentor-8b) |
| RustMentor 8B GGUF | ~4.5GB (Q4_K_M) | 2048 tokens | Offline mobile use | [sylvester-francis/rust-mentor-8b-GGUF](https://huggingface.co/sylvester-francis/rust-mentor-8b-GGUF) |

## Quick Start

### Google Colab (Recommended)

```python
# 1. Mount Drive and clone repository
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-rust-model.git slm_rust
%cd slm_rust

# 2. Add tokens to Colab Secrets (🔑 icon):
# - HF_TOKEN

# 3. Run training (45-60 min on A100)
!python colab/colab_train_and_upload.py
```

### Mobile Deployment (Pixel 8 Pro)

1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai) from Play Store
2. In PocketPal, tap "Add from Hugging Face"
3. Search for your model: `your-username/rust-mentor-8b-GGUF`
4. Download the Q4_K_M quantization (~4.5GB)
5. Create a "Pal" with the Rust tutor system prompt
6. Enable airplane mode and start learning!

## Features

- **Rust-Specialized Tutoring**: Ownership, borrowing, lifetimes, error handling, traits, pattern matching
- **Go/Python/TypeScript Comparisons**: Bridges concepts from languages you already know
- **Code Review Mode**: Explains borrow checker behavior on your code
- **CLI Project Guidance**: Cargo, clap, project structure patterns
- **Offline Mobile**: GGUF export runs on Pixel 8 Pro via PocketPal AI
- **Two Data Sources**: Strandset-Rust-v1 (191K code examples) + synthetic tutor conversations

## Training Data

| Source | Type | Samples | Content |
|--------|------|---------|---------|
| Strandset-Rust-v1 | Code tasks | 3,000 (subset) | Code gen, review, refactoring, bug detection |
| Synthetic Tutor | Teaching Q&A | 500 | Ownership, error handling, traits, CLI patterns |

### Topic Distribution

- **Ownership & Borrowing** (25%): Move semantics, references, clone vs borrow
- **Error Handling** (20%): Result, Option, ?, custom errors, anyhow/thiserror
- **Traits & Generics** (15%): Trait bounds, impl, dynamic dispatch
- **Pattern Matching** (10%): match, enums, destructuring, guards
- **Code Review** (15%): Idiomatic Rust patterns, borrow checker explanations
- **CLI & Project Structure** (10%): Cargo, clap, modules, testing
- **Iterators & Concurrency** (5%): Iterator chains, Rayon, channels

## Project Structure

```
slm-rust-model/
├── README.md                      # Project overview
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── slm.py                        # Unified CLI interface
│
├── colab/                         # Google Colab training
│   ├── colab_train_and_upload.py  # Automated pipeline
│   └── COLAB_CELLS.md            # Ready-to-use notebook cells
│
├── scripts/                       # Core functionality
│   ├── training.py               # QLoRA training with Unsloth
│   ├── data_collection.py        # Synthetic data generation
│   ├── data_preprocessing.py     # Dataset merging & formatting
│   ├── evaluation.py             # Model evaluation
│   ├── convert_gguf.py           # GGUF export
│   ├── upload_to_hf.py           # HuggingFace deployment
│   └── deploy_ollama.py          # Ollama deployment
│
├── docs/                          # Documentation
│   ├── QUICK_START.md            # Getting started
│   ├── COLAB.md                  # Colab training guide
│   └── MOBILE.md                 # Mobile deployment guide
│
├── data/                          # Training data (gitignored)
│   └── processed/                 # Formatted datasets
│
└── models/                        # Trained models (gitignored)
```

## Hardware Requirements

### Google Colab A100 (40GB) — Recommended

| Model | Dataset | Training Time | Memory Usage |
|-------|---------|---------------|--------------|
| 8B | 3.5k samples | 45-60 min | ~32GB |
| 8B | 1k samples | 15-20 min | ~28GB |

### Google Colab T4 (16GB)

| Model | Dataset | Training Time | Notes |
|-------|---------|---------------|-------|
| 4B | 3.5k samples | 60-90 min | Use `--model unsloth/Qwen3-4B` |
| 8B | Any | Not supported | Insufficient VRAM |

## Troubleshooting

**CUDA Out of Memory**
```bash
# Use 4B model instead
python slm.py train --model unsloth/Qwen3-4B --batch-size 1 --grad-accum 8
```

**Strandset Dataset Not Loading**
```bash
# Train with synthetic data only
python slm.py preprocess --strandset-samples 0
```

**Training Stuck on First Step**
- Normal — GPU compiling Unsloth/Triton kernels
- Wait 1-2 minutes, training will accelerate

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

- Built on [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by Alibaba Cloud
- Rust code data from [Strandset-Rust-v1](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1)
- Training optimized by [Unsloth](https://github.com/unslothai/unsloth)
- Training powered by [Hugging Face TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft)

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.
