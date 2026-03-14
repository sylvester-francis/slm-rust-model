# CLAUDE.md — RustMentor SLM

## Project Overview

Fine-tuning pipeline for **RustMentor**, a Rust programming tutor model. Supports four variants:
- **8B** (Qwen3-8B, ~4.5GB GGUF) — highest quality, A100 required
- **4B** (Qwen3-4B, ~2.5GB GGUF) — lighter, T4 compatible
- **1.7B** (Qwen3-1.7B, ~1.1GB GGUF) — fast on-device chat, T4/free Colab
- **0.6B** (Qwen3-0.6B, ~0.4GB GGUF) — ultra-light, instant mobile responses

Uses QLoRA via Unsloth for efficient training. Target deployment: offline on Android (Pixel 8 Pro) via PocketPal AI or RustSensei app using GGUF format. The 1.7B and 0.6B variants are optimized for local chat and quick exercise debugging on mobile.

## Architecture

```
slm.py                          → Unified CLI (dispatches to scripts/*)
scripts/
  data_collection.py            → Synthetic Rust tutor conversation generator (46 seed examples, 28 topics)
  data_preprocessing.py         → Merges Strandset-Rust-v1 + synthetic data into training JSONL
  training.py                   → QLoRA fine-tuning with Unsloth + TRL SFTTrainer
  evaluation.py                 → Keyword-match evaluation on 5 Rust tutor prompts
  convert_gguf.py               → GGUF export via Unsloth (q4_k_m default)
  upload_to_hf.py               → HuggingFace Hub upload with model card
  deploy_ollama.py              → Local Ollama deployment
colab/
  colab_train_and_upload.py     → Self-contained Colab pipeline (supports 8B, 4B, 1.7B, 0.6B, or combos)
```

## Key Commands

```bash
python slm.py pipeline --variant 0.6b --username <hf-user>  # Full pipeline (0.6B)
python slm.py train --variant 1.7b            # Train 1.7B mobile variant
python slm.py train --variant 0.6b            # Train ultra-light 0.6B
python slm.py collect                          # Generate synthetic data
python slm.py preprocess                       # Merge datasets
python slm.py train                            # QLoRA training (default 8B)
python slm.py evaluate --variant 0.6b         # Evaluate 0.6B model
python slm.py convert --variant 1.7b          # Export 1.7B to GGUF
python slm.py upload --variant 0.6b --username <user> --gguf  # Upload to HF
```

## Training Configuration

| Variant | Base Model | LoRA r | Batch | Grad Accum | GGUF Size | GPU |
|---------|-----------|--------|-------|------------|-----------|-----|
| 8B | `unsloth/Qwen3-8B` | 32 | 2 | 4 | ~4.5GB | A100 |
| 4B | `unsloth/Qwen3-4B` | 16 | 1 | 8 | ~2.5GB | T4 |
| 1.7B | `unsloth/Qwen3-1.7B` | 16 | 2 | 4 | ~1.1GB | T4/free |
| 0.6B | `unsloth/Qwen3-0.6B` | 8 | 4 | 2 | ~0.4GB | any |

- **Shared**: 3 epochs, lr=2e-4, cosine scheduler, all attention + MLP projections
- **Data**: ~500 synthetic (46 unique, rest duplicated) + ~3000 Strandset-Rust-v1 samples
- **Format**: Qwen3 chat template (system/user/assistant turns)
- **Hardware**: A100 40GB for 8B; T4 for 4B/1.7B; any GPU (including free Colab) for 0.6B
- **Colab**: Set `TRAIN_VARIANTS = "mobile"` / `"0.6b"` / `"1.7b"` / `"all"` etc. in colab_train_and_upload.py
- **CLI**: Use `--variant 0.6b` or `--variant 1.7b` with any slm.py command

## Data Pipeline

1. `data_collection.py` has 46 hand-written Rust tutoring conversations across 28 topics (ownership, error handling, traits, async, smart pointers, macros, serde, testing, etc.)
2. `generate_rust_dataset()` duplicates seed examples to reach target count (no actual variation)
3. `data_preprocessing.py` loads Strandset-Rust-v1 from HuggingFace, reformats to chat template, merges with synthetic data

## Dependencies

torch, transformers, accelerate, peft, trl, bitsandbytes, datasets, unsloth, huggingface_hub, hf_transfer, sentencepiece, protobuf

## Important Notes

- The `slm.py` CLI imports from `scripts.*` — the `scripts/` directory with `__init__.py` must exist
- Colab script imports from `scripts.*` as well — repo must be cloned with full structure
- HF_TOKEN required for upload (via env var or Colab Secrets)
- The tar.gz at project root contains the complete repo with proper directory structure
