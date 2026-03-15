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
slm.py                          → Unified CLI (dispatches to scripts/*)
scripts/
  data_collection.py            → Synthetic Rust tutor conversation generator (46 seed examples, 28 topics)
  data_preprocessing.py         → Merges Strandset-Rust-v1 + synthetic data into training JSONL
  training.py                   → QLoRA fine-tuning with Unsloth + TRL SFTTrainer
  evaluation.py                 → Keyword-match evaluation on 5 Rust tutor prompts
  convert_gguf.py               → GGUF export via Unsloth (q4_k_m default)
  convert_litert.py             → LiteRT (.tflite) export for Qwen3 models
  bundle_litertlm.py            → Bundle .tflite + tokenizer → .litertlm for Google AI Edge Gallery
  upload_to_hf.py               → HuggingFace Hub upload with model card
  deploy_ollama.py              → Local Ollama deployment
colab/
  colab_gemma3_pipeline.py      → Full Gemma3-1B pipeline: train → merge → convert → bundle → upload
  colab_litert_convert.py       → Download from HF → merge → convert → bundle (Qwen3)
  colab_bundle_only.py          → Download .tflite from HF → bundle .litertlm → upload
  colab_train_and_upload.py     → Qwen3 training pipeline (legacy)
```

## Gemma3 Mobile Pipeline (Recommended)

```bash
# Full pipeline in Colab:
!python colab/colab_gemma3_pipeline.py
```

| Setting | Value |
|---------|-------|
| Base Model | `unsloth/gemma-3-1b-it` (Gemma license) |
| LoRA r | 16 |
| Batch Size | 2 × 4 = 8 effective |
| Epochs | 3 |
| LiteRT Quant | dynamic_int8 (~650MB) |
| Stop Tokens | `[1, 106]` (`<eos>`, `<end_of_turn>`) |
| GPU Delegation | Full on Tensor G3 (Mali-G715) |
| Chat Template | `<start_of_turn>` / `<end_of_turn>` |

Pipeline: Train (Unsloth) → Merge (fp16) → Convert (.tflite via Gemma3 converter) → Bundle (.litertlm) → Upload to HF

## Legacy Qwen3 Commands

```bash
python slm.py pipeline --variant 0.6b --username <hf-user>
python slm.py train --variant 1.7b
python slm.py convert --variant 0.6b           # GGUF
python slm.py convert-litert --variant 0.6b    # LiteRT
```

## Data Pipeline

1. `data_collection.py` has 46 hand-written Rust tutoring conversations across 28 topics (ownership, error handling, traits, async, smart pointers, macros, serde, testing, etc.)
2. `generate_rust_dataset()` duplicates seed examples to reach target count (no actual variation)
3. `data_preprocessing.py` loads Strandset-Rust-v1 from HuggingFace, reformats to chat template, merges with synthetic data

## Dependencies

torch, transformers, accelerate, peft, trl, bitsandbytes, datasets, unsloth, huggingface_hub, hf_transfer, sentencepiece, protobuf, litert-torch

## Important Notes

- Gemma3 requires accepting the license at huggingface.co/google/gemma-3-1b-it
- Gemma license (not Apache 2.0) — free for commercial use, has Prohibited Use Policy
- LiteRT converter only supports Gemma3 1B and 270M (no 4B)
- The `slm.py` CLI imports from `scripts.*` — the `scripts/` directory with `__init__.py` must exist
- HF_TOKEN required for upload and Gemma model access
