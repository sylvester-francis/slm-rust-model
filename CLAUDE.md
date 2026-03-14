# CLAUDE.md — RustMentor SLM

## Project Overview

Fine-tuning pipeline for **RustMentor**, a Rust programming tutor model. Supports two variants:
- **8B** (Qwen3-8B, ~4.5GB GGUF) — higher quality, A100 required
- **4B** (Qwen3-4B, ~2.5GB GGUF) — lighter, T4 compatible, better for mobile

Uses QLoRA via Unsloth for efficient training. Target deployment: offline on Android (Pixel 8 Pro) via PocketPal AI or RustSensei app using GGUF format.

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
  colab_train_and_upload.py     → Self-contained Colab pipeline (supports 8B, 4B, or both variants)
```

## Key Commands

```bash
python slm.py pipeline --username <hf-user>   # Full pipeline
python slm.py collect                          # Generate synthetic data
python slm.py preprocess                       # Merge datasets
python slm.py train                            # QLoRA training
python slm.py evaluate                         # Evaluate model
python slm.py convert                          # Export GGUF
python slm.py upload --username <user> --gguf  # Upload to HF
```

## Training Configuration

| Variant | Base Model | LoRA r | Batch | Grad Accum | GGUF Size |
|---------|-----------|--------|-------|------------|-----------|
| 8B | `unsloth/Qwen3-8B` | 32 | 2 | 4 | ~4.5GB |
| 4B | `unsloth/Qwen3-4B` | 16 | 1 | 8 | ~2.5GB |

- **Shared**: 3 epochs, lr=2e-4, cosine scheduler, all attention + MLP projections
- **Data**: ~500 synthetic (46 unique, rest duplicated) + ~3000 Strandset-Rust-v1 samples
- **Format**: Qwen3 chat template (system/user/assistant turns)
- **Hardware**: A100 40GB required for 8B; T4 for 4B variant
- **Colab**: Set `TRAIN_VARIANTS = "both"` / `"8b"` / `"4b"` in colab_train_and_upload.py

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
