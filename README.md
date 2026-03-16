# RustMentor SLM: Fine-Tuned Rust Programming Tutor

**RustMentor SLM** is a fine-tuning pipeline that produces small language models specialized in Rust programming education and code review, optimized for offline mobile inference.

## Core Offering

The pipeline trains models that serve as an interactive Rust tutor for developers transitioning from Go, Python, or TypeScript. Models are exported in two formats for on-device deployment: **GGUF** for llama.cpp/Ollama and **LiteRT** (.litertlm) for Google AI Edge Gallery with GPU acceleration.

## Available Models

Six model variants across two base architectures, targeting different hardware profiles:

| Variant | Base | Export | Size | Target |
|---------|------|--------|------|--------|
| **0.6B** | Qwen3-0.6B | GGUF + LiteRT | ~0.4GB | Any device, instant responses |
| **1.7B** | Qwen3-1.7B | GGUF + LiteRT | ~1.1GB | On-device chat, exercise debugging |
| **4B** | Qwen3-4B | GGUF + LiteRT | ~2.5GB | Balanced quality and speed |
| **8B** | Qwen3-8B | GGUF | ~4.5GB | Highest quality tutoring |
| **gemma3-1b** | Gemma3-1B-IT | LiteRT (.litertlm) | ~650MB | Google AI Edge Gallery, full GPU on Pixel 8 Pro |
| **gemma3-4b** | Gemma3-4B-IT | GGUF | ~2.5GB | PocketPal AI, higher quality mobile |

Pre-trained models are available on [HuggingFace](https://huggingface.co/sylvester-francis).

## Training Pipeline

Built on QLoRA via Unsloth for efficient fine-tuning on consumer GPUs. The pipeline supports training on free Google Colab (T4) for smaller variants and A100 for larger models.

Training data combines 46 hand-written Rust tutoring conversations across 28 topics with 3,000 samples from the Strandset-Rust-v1 dataset. All conversations draw parallels to Go, Python, and TypeScript equivalents.

## Technical Foundation

The CLI (`slm.py`) dispatches to the `rustmentor/` Python package, which handles data generation, preprocessing, training, evaluation, export, and deployment. Tutorial scripts in `tutorials/` provide a step-by-step walkthrough.

Export supports two paths:
- **GGUF** via Unsloth's quantization (Q4_K_M) for llama.cpp, Ollama, and PocketPal AI
- **LiteRT** via Google's litert-torch converter, bundled as `.litertlm` with tokenizer and stop tokens for the Google AI Edge Gallery

## Deployment Options

The models run entirely offline after initial download:

- **Google AI Edge Gallery** (Android) — GPU-accelerated inference via LiteRT on Pixel 8 Pro
- **PocketPal AI** (Android) — CPU-based inference via GGUF
- **Ollama** (Desktop) — Local deployment via GGUF
- **RustSensei** (Android) — Companion app with structured curriculum and AI tutor

## Quick Start

```python
# Google Colab — Gemma3-1B mobile pipeline
!git clone https://github.com/sylvester-francis/slm-rust-model.git
%cd slm-rust-model
import os; from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!python colab/colab_gemma3_pipeline.py
```

```bash
# Local CLI — any variant
python slm.py pipeline --variant 0.6b --username your-hf-user
```

## Tutoring Capabilities

- Rust ownership, borrowing, and lifetime explanations with Go/Python/TS comparisons
- Code review with borrow checker analysis
- Error handling patterns (Result, Option, ?, thiserror, anyhow)
- Async/await and Tokio patterns
- Smart pointers, pattern matching, trait-based architecture
- CLI tooling with clap, Cargo project structure
- Serde serialization, testing patterns, and module organization

## Access

Models are available on [HuggingFace](https://huggingface.co/sylvester-francis) in adapter, GGUF, and LiteRT formats. The training pipeline requires Python 3.10+, PyTorch, and a CUDA GPU (or free Google Colab). Gemma3 variants require accepting Google's [Gemma license](https://ai.google.dev/gemma/terms).

## License

Apache License 2.0 for the pipeline and Qwen3 variants. Gemma3 variants follow the [Gemma license](https://ai.google.dev/gemma/terms).
