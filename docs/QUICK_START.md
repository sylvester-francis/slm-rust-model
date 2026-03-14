# Quick Start Guide

Get RustMentor running in under 5 minutes.

## Option 1: Google Colab (Recommended)

**Requirements**: Google Colab Pro (A100 GPU), HuggingFace account

### Steps

1. Open [Google Colab](https://colab.research.google.com/)
2. Set runtime: **Runtime → Change runtime type → A100 GPU**
3. Add your HF token: **🔑 Secrets → Add HF_TOKEN**
4. Run these cells:

```python
# Cell 1: Clone repo
!git clone https://github.com/sylvester-francis/slm-rust-model.git
%cd slm-rust-model

# Cell 2: Run full pipeline (~60 min)
!python colab/colab_train_and_upload.py
```

That's it! The script will:
- Generate synthetic Rust tutor data
- Download and merge Strandset-Rust-v1 code examples
- Fine-tune Qwen3-8B with QLoRA
- Export to GGUF (Q4_K_M)
- Upload to your HuggingFace account

## Option 2: Local Training

**Requirements**: NVIDIA GPU with 40GB+ VRAM (A100, A6000, etc.)

```bash
git clone https://github.com/sylvester-francis/slm-rust-model.git
cd slm-rust-model
pip install -r requirements.txt
python slm.py pipeline --username your-hf-username
```

## Option 3: Use a Pre-trained Model

If you just want to use the model without training:

### On Android (PocketPal AI)
1. Install [PocketPal AI](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Tap "Add from Hugging Face"
3. Search: `sylvester-francis/rust-mentor-8b-GGUF`
4. Download Q4_K_M (~4.5GB)
5. Create a Pal with the system prompt from the README

### On Desktop (Ollama)
```bash
# After converting locally
python slm.py deploy
ollama run rust-mentor-8b
```

## Next Steps

- **[Colab Guide](COLAB.md)** — Detailed training options and customization
- **[Mobile Guide](MOBILE.md)** — PocketPal setup and tips for airplane use
