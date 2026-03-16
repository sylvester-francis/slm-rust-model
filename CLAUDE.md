# CLAUDE.md — RustMentor SLM

## Project Overview

Fine-tuning pipeline for **RustMentor**, a Rust programming tutor model. Two generations:

### Mobile-First (Gemma3, recommended)
- **rust-mentor-1b-mobile** (Gemma3-1B-IT, ~650MB .litertlm) — full GPU on Pixel 8 Pro, Google AI Edge Gallery

### Legacy (Qwen3, GGUF)
- **8B** (Qwen3-8B, ~4.5GB GGUF) — highest quality, A100 required
- **4B** (Qwen3-4B, ~2.5GB GGUF) — lighter, T4 compatible
- **1.7B** (Qwen3-1.7B, ~1.1GB GGUF) — fast on-device chat
- **0.6B** (Qwen3-0.6B, ~0.4GB GGUF) — ultra-light

Uses QLoRA via Unsloth for efficient training. Target deployment: offline on Android (Pixel 8 Pro) via Google AI Edge Gallery using .litertlm format (Gemma3) or PocketPal AI using GGUF format (Qwen3).

## Architecture

```
slm.py                          → Unified CLI (dispatches to rustmentor/*)
rustmentor/                     → Main Python package
  config.py                     → Single source of truth: prompts, variants, constants
  data/
    collection.py               → Synthetic Rust tutor conversation generator (46 seeds, 28 topics)
    preprocessing.py            → Merges Strandset-Rust-v1 + synthetic data into training JSONL
  training/
    trainer.py                  → QLoRA fine-tuning with Unsloth + TRL SFTTrainer
    evaluation.py               → Keyword-match evaluation on 5 Rust tutor prompts
  export/
    gguf.py                     → GGUF export via Unsloth + llama.cpp (Gemma3 support)
    litert.py                   → LiteRT (.tflite) export for Qwen3 + Gemma3 models
    bundle.py                   → Bundle .tflite + tokenizer → .litertlm for AI Edge Gallery
  deploy/
    huggingface.py              → HuggingFace Hub upload with model card
    ollama.py                   → Local Ollama deployment
tutorials/                      → Step-by-step tutorial scripts (01 through 05)
colab/
  colab_gemma3_pipeline.py      → Gemma3-1B: train → merge → LiteRT → upload (Pixel 8 Pro)
  colab_gemma3_4b_pipeline.py   → Gemma3-4B: train → merge → GGUF → upload (PocketPal AI)
```

## Quick Start

### Tutorial (step by step)
```bash
python tutorials/01_data_preparation.py       # Generate & merge training data
python tutorials/02_fine_tuning.py             # Train with QLoRA (needs GPU)
python tutorials/03_evaluation.py              # Evaluate model quality
python tutorials/04_export.py                  # Export to GGUF or LiteRT
python tutorials/05_deploy.py                  # Deploy to Ollama or HuggingFace
```

### CLI (all-in-one)
```bash
python slm.py pipeline --variant 0.6b --username <hf-user>
python slm.py train --variant 1.7b
python slm.py convert --variant 0.6b           # GGUF
python slm.py convert-litert --variant 0.6b    # LiteRT
```

### Colab (Gemma3 mobile)
```bash
!python colab/colab_gemma3_pipeline.py
```

## Shared Config (rustmentor/config.py)

All shared constants live in `rustmentor/config.py`:
- `SYSTEM_PROMPT` — RustMentor persona used in training, eval, and deployment
- `VARIANT_CONFIGS` — Presets for all model sizes (8b, 4b, 1.7b, 0.6b, gemma3-1b, gemma3-4b)
- `BNB4BIT_TO_FULL` — Maps Unsloth 4-bit model names to full-precision equivalents
- `EVAL_PROMPTS` — 5 evaluation prompts with expected keywords
- `STOP_TOKENS` — Per-model stop token IDs for .litertlm bundling
- `MODEL_CARD_TEMPLATE` — HuggingFace model card template

## Data Pipeline

1. `rustmentor/data/seeds.py` has 46 hand-written Rust tutoring conversations across 28 topics
2. `generate_rust_dataset()` duplicates seed examples to reach target count
3. `preprocess_and_merge()` loads Strandset-Rust-v1 from HuggingFace, reformats, merges with synthetic data

## Dependencies

torch, transformers, accelerate, peft, trl, bitsandbytes, datasets, unsloth, huggingface_hub, hf_transfer, sentencepiece, protobuf, litert-torch

## Important Notes

- Gemma3 requires accepting the license at huggingface.co/google/gemma-3-1b-it
- Gemma license (not Apache 2.0) — free for commercial use, has Prohibited Use Policy
- LiteRT converter only supports Gemma3 1B and 270M (no 4B)
- The `slm.py` CLI imports from `rustmentor.*` — the `rustmentor/` package must exist
- `rustmentor/data/seeds.py` contains the canonical 46 seed conversations
- HF_TOKEN required for upload and Gemma model access
