# Google Colab Training Guide

Complete guide for training RustMentor on Google Colab.

## Prerequisites

- **Google Colab Pro** ($12/month) — required for A100 GPU access
- **HuggingFace account** — free at [huggingface.co](https://huggingface.co)
  - Create a **write** access token at [Settings → Tokens](https://huggingface.co/settings/tokens)

## GPU Selection

| GPU | VRAM | Model | Training Time | Cost |
|-----|------|-------|---------------|------|
| A100 | 40GB | Qwen3-8B | 45-60 min | Colab Pro |
| T4 | 16GB | Qwen3-4B only | 60-90 min | Free tier |
| L4 | 24GB | Qwen3-8B (tight) | 75-90 min | Colab Pro |

**Recommendation**: A100 for the 8B model. If budget is tight, T4 + Qwen3-4B still produces a usable tutor.

## Setup

### 1. Create Notebook

Open [colab.research.google.com](https://colab.research.google.com/) → New Notebook

### 2. Set Runtime

Runtime → Change runtime type → **A100 GPU**

### 3. Add Secrets

Click 🔑 in the left sidebar → Add new secret:
- Name: `HF_TOKEN`
- Value: Your HuggingFace write token

### 4. Clone and Run

```python
# Cell 1
!git clone https://github.com/sylvester-francis/slm-rust-model.git
%cd slm-rust-model

# Cell 2 — Full automated pipeline
!python colab/colab_train_and_upload.py
```

## Customization

### Using a Smaller Model (T4 Compatible)

Edit `colab/colab_train_and_upload.py` or run manually:

```python
!python slm.py train \
    --model unsloth/Qwen3-4B \
    --batch-size 1 \
    --grad-accum 8 \
    --lora-r 16 \
    --epochs 3
```

### Adjusting Training Parameters

```python
# More epochs = better but risk overfitting
!python slm.py train --epochs 5 --lr 1e-4

# Higher LoRA rank = more capacity
!python slm.py train --lora-r 64 --lora-alpha 64

# Larger effective batch size = smoother training
!python slm.py train --batch-size 4 --grad-accum 8
```

### Different GGUF Quantization

```python
# Higher quality (larger file, ~5.5GB)
!python slm.py convert --quant q5_k_m

# Maximum compression (smaller, ~2.5GB, quality tradeoff)
!python slm.py convert --quant q2_k

# Near-lossless (largest, ~8GB)
!python slm.py convert --quant q8_0
```

### Synthetic Data Only (No Strandset)

If Strandset-Rust-v1 is inaccessible:

```python
!python slm.py collect --samples 500
!python slm.py preprocess --strandset-samples 0
!python slm.py train --data data/processed/train.jsonl
```

## Step-by-Step Manual Pipeline

For more control, run each step separately:

```python
# 1. Check system
!python slm.py info

# 2. Generate synthetic tutor data
!python slm.py collect --samples 500

# 3. Merge with Strandset-Rust-v1
!python slm.py preprocess --strandset-samples 3000

# 4. Train (this is the long step)
!python slm.py train --epochs 3 --lora-r 32

# 5. Evaluate
!python slm.py evaluate

# 6. Export to GGUF
!python slm.py convert --quant q4_k_m

# 7. Upload to HuggingFace
!python slm.py upload --username YOUR_USERNAME --gguf
```

## Troubleshooting

### "CUDA out of memory"

Reduce memory usage:
```python
!python slm.py train --batch-size 1 --grad-accum 16 --lora-r 16
```

Or switch to 4B model:
```python
!python slm.py train --model unsloth/Qwen3-4B
```

### "Training stuck on first step"

Normal behavior — Unsloth/Triton kernels are compiling. Wait 1-2 minutes.

### "Strandset dataset not found"

The dataset may be gated. Train with synthetic data only:
```python
!python slm.py preprocess --strandset-samples 0
```

### "Session disconnected"

Colab sessions timeout after ~90 min of inactivity. Tips:
- Keep the tab active
- Use Colab Pro for longer sessions
- Save checkpoints to Drive (training saves per epoch by default)

### "Import errors after pip install"

Restart the runtime after installing packages:
Runtime → Restart runtime, then re-run the training cell.
