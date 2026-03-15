# 🦀 RustMentor SLM — Small Language Models for Rust Programming

Fine-tuned small language models specialized in Rust programming education and code review. Built on Qwen3 with QLoRA fine-tuning, exported to both **GGUF** (llama.cpp/Ollama) and **LiteRT** (.tflite, Android GPU/NPU) for offline mobile inference on Pixel 8 Pro.

## CLI

```bash
# Full pipeline with variant selection
python slm.py pipeline --variant 0.6b --username your-username

# Individual steps
python slm.py collect                          # Generate synthetic training data
python slm.py preprocess                       # Merge Strandset + synthetic data
python slm.py train --variant 1.7b             # Train model with QLoRA
python slm.py evaluate --variant 0.6b          # Evaluate model quality
python slm.py convert --variant 0.6b           # Export to GGUF for llama.cpp/Ollama
python slm.py convert-litert --variant 0.6b    # Export to LiteRT for Android GPU/NPU
python slm.py upload --variant 0.6b --username your-username --gguf    # Upload GGUF
python slm.py upload --variant 0.6b --username your-username --litert  # Upload LiteRT
python slm.py deploy --variant 0.6b            # Deploy to Ollama
```

## Available Models

| Variant | Base Model | GGUF Size | LiteRT | GPU to Train | Best For |
|---------|-----------|-----------|--------|-------------|----------|
| **0.6B** | Qwen3-0.6B | ~0.4GB | .tflite INT8 | Any | Ultra-light mobile, instant responses |
| **1.7B** | Qwen3-1.7B | ~1.1GB | .tflite INT8 | T4/free | On-device chat, exercise debugging |
| **4B** | Qwen3-4B | ~2.5GB | .tflite INT8 | T4 | Balanced quality/speed |
| **8B** | Qwen3-8B | ~4.5GB | .tflite INT8 | A100 | Highest quality tutoring |

### HuggingFace Repos

Each variant is uploaded in three formats:

- `sylvester-francis/rust-mentor-{variant}` — LoRA adapter weights
- `sylvester-francis/rust-mentor-{variant}-GGUF` — GGUF quantized (Q4_K_M) for llama.cpp/Ollama
- `sylvester-francis/rust-mentor-{variant}-LiteRT` — LiteRT .tflite for Android GPU/NPU acceleration

## Export Formats

| Format | Engine | Android Acceleration | Use Case |
|--------|--------|---------------------|----------|
| **GGUF** (Q4_K_M) | llama.cpp | CPU | Ollama, PocketPal AI, llama.cpp |
| **LiteRT** (.tflite) | litert-torch | GPU/NPU via NNAPI | RustSensei app, Google AI Edge Gallery |

LiteRT provides **2-3x faster inference** on Pixel 8 Pro (Tensor G3) compared to GGUF/llama.cpp by leveraging the GPU and NPU hardware.

## Quick Start

### Google Colab (Recommended)

```python
# 1. Clone repository
!git clone https://github.com/sylvester-francis/slm-rust-model.git
%cd slm-rust-model

# 2. Set HF_TOKEN in Colab Secrets (🔑 icon)
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# 3. Run pipeline — trains all variants and exports LiteRT
!python colab/colab_train_and_upload.py
```

Configure in `colab/colab_train_and_upload.py`:
```python
TRAIN_VARIANTS = "all"       # "8b", "4b", "1.7b", "0.6b", "mobile" (1.7b+0.6b), "all"
EXPORT_FORMATS = "litert"    # "litert", "gguf", or "both"
```

### Mobile Deployment (Pixel 8 Pro)

**Option A: LiteRT (GPU/NPU accelerated)**
1. Install [Google AI Edge Gallery](https://play.google.com/store/apps/details?id=com.google.ai.edge.gallery)
2. Import the .tflite model from HuggingFace
3. Chat offline with hardware acceleration

**Option B: GGUF (CPU-based)**
1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Search: `sylvester-francis/rust-mentor-0.6b-GGUF`
3. Download and chat offline

## Features

- **Rust-Specialized Tutoring**: Ownership, borrowing, lifetimes, error handling, traits, pattern matching
- **Go/Python/TypeScript Comparisons**: Bridges concepts from languages you already know
- **Code Review Mode**: Explains borrow checker behavior on your code
- **CLI Project Guidance**: Cargo, clap, project structure patterns
- **Dual Export**: GGUF (llama.cpp) + LiteRT (Android GPU/NPU) from the same training run
- **4 Size Variants**: From 0.6B (instant) to 8B (highest quality)

## Training Data

| Source | Type | Samples | Content |
|--------|------|---------|---------|
| Strandset-Rust-v1 | Code tasks | 3,000 (subset) | Code gen, review, refactoring, bug detection |
| Synthetic Tutor | Teaching Q&A | 500 | Ownership, error handling, traits, CLI patterns |

## Project Structure

```
slm-rust-model/
├── README.md                      # Project overview
├── CLAUDE.md                      # AI assistant context
├── slm.py                        # Unified CLI (--variant flag for all commands)
│
├── colab/                         # Google Colab training
│   └── colab_train_and_upload.py  # Automated pipeline (all variants + exports)
│
├── scripts/                       # Core functionality
│   ├── training.py               # QLoRA training with Unsloth
│   ├── data_collection.py        # Synthetic data generation
│   ├── data_preprocessing.py     # Dataset merging & formatting
│   ├── evaluation.py             # Model evaluation
│   ├── convert_gguf.py           # GGUF export via Unsloth
│   ├── convert_litert.py         # LiteRT (.tflite) export via litert-torch
│   ├── upload_to_hf.py           # HuggingFace upload
│   └── deploy_ollama.py          # Ollama deployment
│
├── data/                          # Training data (gitignored)
│   └── processed/                 # Formatted datasets
│
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

**LiteRT Conversion Fails**
```bash
# Ensure litert-torch is installed and you have 32GB+ RAM
pip install litert-torch
# Convert GGUF instead as fallback
python slm.py convert --variant 0.6b
```

**Strandset Dataset Not Loading**
```bash
# Train with synthetic data only
python slm.py preprocess --strandset-samples 0
```

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

- Built on [Qwen3](https://huggingface.co/Qwen) by Alibaba Cloud (0.6B, 1.7B, 4B, 8B)
- Rust code data from [Strandset-Rust-v1](https://huggingface.co/datasets/Fortytwo-Network/Strandset-Rust-v1)
- Training optimized by [Unsloth](https://github.com/unslothai/unsloth)
- LiteRT export via [litert-torch](https://github.com/google-ai-edge/litert-torch) by Google
- Training powered by [Hugging Face TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft)

## License

Apache License 2.0 — See [LICENSE](LICENSE) for details.
