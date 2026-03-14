# Quick Start Guide

Get RustMentor running in under 5 minutes.

## Option 1: Google Colab (Recommended)

**Requirements**: Google Colab with A100 GPU, HuggingFace account with write token

### Steps

1. Open [Google Colab](https://colab.research.google.com/)
2. Set runtime: **Runtime → Change runtime type → A100 GPU**
3. Add your HF token: **🔑 Secrets → Add HF_TOKEN**
4. Run these cells:

```python
# Cell 1: Clone repo and set HF token
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!git clone https://github.com/sylvester-francis/slm-rust-model.git
%cd slm-rust-model
```

```python
# Cell 2: Run full pipeline
!python colab/colab_train_and_upload.py
```

The script will:
- Generate 46 synthetic Rust tutor conversations across 28 topics
- Download and merge Strandset-Rust-v1 code examples (~3,000 samples)
- Fine-tune Qwen3-8B with QLoRA on A100
- Upload LoRA adapter to HuggingFace
- Push GGUF (Q4_K_M) directly to HuggingFace (no local disk needed)

## Option 2: Use the Pre-trained Model

If you just want to use the model without training:

### On Android (PocketPal AI)
1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Tap "Add from Hugging Face"
3. Search: `sylvester-francis/rust-mentor-8b-GGUF`
4. Download Q4_K_M (~4.5GB)
5. Create a Pal with the system prompt from the README

## Next Steps

- **[Colab Guide](COLAB.md)** — Detailed training options and customization
- **[Mobile Guide](MOBILE.md)** — PocketPal setup and flight preparation
