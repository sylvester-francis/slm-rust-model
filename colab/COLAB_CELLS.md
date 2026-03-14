# RustMentor SLM — Google Colab Cells

Copy these cells into a Colab notebook, or run the automated script directly.

## Cell 1: Setup

```python
# Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/
!git clone https://github.com/sylvester-francis/slm-rust-model.git slm_rust
%cd slm_rust
```

## Cell 2: Add Secrets

```
# Click 🔑 Secrets icon in left sidebar
# Add: HF_TOKEN → your HuggingFace write token
```

## Cell 3: Run Pipeline (A100 — ~60 min)

```python
!python colab/colab_train_and_upload.py
```

## Alternative: Step by Step

```python
# Install deps
!pip install -q unsloth trl peft accelerate bitsandbytes datasets huggingface_hub

# Generate data
!python slm.py collect --samples 500

# Preprocess (merges with Strandset-Rust-v1)
!python slm.py preprocess --strandset-samples 3000

# Train
!python slm.py train --epochs 3 --lora-r 32

# Convert to GGUF
!python slm.py convert --quant q4_k_m

# Upload
!python slm.py upload --username YOUR_USERNAME --gguf
```
