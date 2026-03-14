# Google Colab Training Guide

Complete guide for training RustMentor on Google Colab.

## Prerequisites

- **Google Colab** with A100 GPU access (Colab Pro recommended)
- **HuggingFace account** — free at [huggingface.co](https://huggingface.co)
  - Create a **write** access token at [Settings → Tokens](https://huggingface.co/settings/tokens)

## GPU Selection

| GPU | VRAM | Model | Notes |
|-----|------|-------|-------|
| A100 | 40GB | Qwen3-8B | Recommended. Full pipeline works. |
| T4 | 16GB | Qwen3-4B only | 8B model will OOM. Use `--model unsloth/Qwen3-4B`. |

## Setup

### 1. Create Notebook

Open [colab.research.google.com](https://colab.research.google.com/) → New Notebook

### 2. Set Runtime

Runtime → Change runtime type → **A100 GPU**

### 3. Add HF Token to Secrets

Click 🔑 in the left sidebar → Add new secret:
- Name: `HF_TOKEN`
- Value: Your HuggingFace write token
- Toggle "Notebook access" ON

### 4. Run the Pipeline

**Cell 1** — Set token and clone:
```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!git clone https://github.com/sylvester-francis/slm-rust-model.git
%cd slm-rust-model
```

**Cell 2** — Run the full pipeline:
```python
!python colab/colab_train_and_upload.py
```

> **Important**: The HF token must be set as an environment variable in Cell 1 before running the script. The `userdata.get()` API only works inside notebook cells, not from `!python` subprocesses.

The pipeline runs 4 steps:
1. **Generate data** — 46 unique Rust tutor conversations across 28 topics
2. **Preprocess** — Merge with Strandset-Rust-v1 (~3,000 code samples)
3. **Train** — QLoRA fine-tuning with Unsloth on A100
4. **Upload** — Push LoRA adapter + GGUF directly to HuggingFace (GGUF is streamed to HF to avoid Colab disk space limits)

## Customization

### Using a Smaller Model (T4 Compatible)

Edit the constants at the top of `colab/colab_train_and_upload.py`:

```python
BASE_MODEL = "unsloth/Qwen3-4B"  # instead of Qwen3-8B
BATCH_SIZE = 1
GRAD_ACCUM = 8
LORA_R = 16
```

### Adjusting Training Parameters

```python
# More epochs (risk overfitting with small dataset)
# Edit EPOCHS = 5 in colab_train_and_upload.py

# Higher LoRA rank = more capacity
# Edit LORA_R = 64 and LORA_ALPHA = 64

# Different quantization
# Edit GGUF_QUANT = "q5_k_m"  (higher quality, ~5.5GB)
```

### Synthetic Data Only (No Strandset)

If Strandset-Rust-v1 is inaccessible, edit `colab/colab_train_and_upload.py`:

```python
STRANDSET_SAMPLES = 0
```

## Step-by-Step Manual Pipeline

For more control, run each step separately in Colab cells:

```python
# 1. Check system
!python slm.py info

# 2. Generate synthetic tutor data (46 unique conversations)
!python slm.py collect --samples 500

# 3. Merge with Strandset-Rust-v1
!python slm.py preprocess --strandset-samples 3000

# 4. Train (the long step)
!python slm.py train --epochs 3 --lora-r 32

# 5. Evaluate
!python slm.py evaluate
```

> **Note**: For GGUF conversion and upload, use the automated `colab_train_and_upload.py` script. It pushes GGUF directly to HuggingFace without saving locally, which avoids Colab's disk space limits (~78GB).

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

The dataset may be gated or unavailable. Train with synthetic data only:
```python
!python slm.py preprocess --strandset-samples 0
```

### "Session disconnected"

Colab sessions can timeout. Tips:
- Keep the tab active
- Use Colab Pro for longer sessions
- Training saves checkpoints per epoch automatically

### "transformers version incompatible"

Unsloth requires `transformers<=5.2.0`. The Colab script pins this automatically. If running manually:
```python
!pip install 'transformers>=4.51.3,<=5.2.0'
```

### "AttributeError: total_mem"

Newer PyTorch renamed this to `total_memory`. This is already fixed in the latest code — make sure you pulled the latest from GitHub.
