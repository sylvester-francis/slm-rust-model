# CLAUDE.md — RustMentor SLM

## Project Overview

Fine-tuning pipeline for **RustMentor**, a Rust programming tutor model based on Qwen3-8B. Uses QLoRA via Unsloth for efficient training. Target deployment: offline on Android (Pixel 8 Pro) via PocketPal AI using GGUF format.

## Architecture

```
slm.py                          → Unified CLI (dispatches to scripts/*)
scripts/
  data_collection.py            → Synthetic Rust tutor conversation generator (15 seed examples)
  data_preprocessing.py         → Merges Strandset-Rust-v1 + synthetic data into training JSONL
  training.py                   → QLoRA fine-tuning with Unsloth + TRL SFTTrainer
  evaluation.py                 → Keyword-match evaluation on 5 Rust tutor prompts
  convert_gguf.py               → GGUF export via Unsloth (q4_k_m default)
  upload_to_hf.py               → HuggingFace Hub upload with model card
  deploy_ollama.py              → Local Ollama deployment
colab/
  colab_train_and_upload.py     → Self-contained Colab pipeline (installs deps, runs all steps)
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

- **Base model**: `unsloth/Qwen3-8B`
- **LoRA**: r=32, alpha=32, targets all attention + MLP projections
- **Training**: 3 epochs, batch_size=2, grad_accum=4, lr=2e-4, cosine scheduler
- **Data**: ~500 synthetic (15 unique, rest duplicated) + ~3000 Strandset-Rust-v1 samples
- **Format**: Qwen3 chat template (system/user/assistant turns)
- **Hardware**: A100 40GB required for 8B; T4 for 4B variant

## Data Pipeline

1. `data_collection.py` has 15 hand-written Rust tutoring conversations covering ownership, error handling, traits, lifetimes, strings, iterators, cargo, CLI, code review, concurrency, testing
2. `generate_rust_dataset()` duplicates seed examples to reach target count (no actual variation)
3. `data_preprocessing.py` loads Strandset-Rust-v1 from HuggingFace, reformats to chat template, merges with synthetic data

## Dependencies

torch, transformers, accelerate, peft, trl, bitsandbytes, datasets, unsloth, huggingface_hub, hf_transfer, sentencepiece, protobuf

## Important Notes

- The `slm.py` CLI imports from `scripts.*` — the `scripts/` directory with `__init__.py` must exist
- Colab script imports from `scripts.*` as well — repo must be cloned with full structure
- HF_TOKEN required for upload (via env var or Colab Secrets)
- The tar.gz at project root contains the complete repo with proper directory structure
