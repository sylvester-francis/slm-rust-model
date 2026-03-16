"""
Step 3: QLoRA Fine-Tuning with Unsloth

Supports all Qwen3 variants (0.6B-8B) and Gemma3 (1B, 4B).
Uses Unsloth for 2x faster training with 70% less VRAM.

Requirements:
    pip install unsloth trl peft accelerate bitsandbytes datasets torch

Usage:
    from rustmentor.training import train_model

    train_model(
        base_model="unsloth/Qwen3-0.6B",
        data_path="data/processed/train.jsonl",
        output_dir="models/rust-mentor-0.6b",
        lora_r=8,
    )
"""

import json
import os
from pathlib import Path


def _check_gpu():
    """Check GPU availability and return device info."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            vram_gb = gpu.total_memory / 1024**3
            return {
                "device": "cuda",
                "name": gpu.name,
                "vram_gb": round(vram_gb, 1),
                "bf16": torch.cuda.is_bf16_supported(),
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"device": "mps", "name": "Apple Silicon", "vram_gb": 0, "bf16": False}
    except ImportError:
        pass
    return {"device": "cpu", "name": "CPU", "vram_gb": 0, "bf16": False}


def format_chat_template(example, tokenizer):
    """Format a conversation example using the model's chat template.

    Falls back to manual Qwen3 chatml format if the tokenizer
    doesn't support apply_chat_template.
    """
    messages = example.get("conversations", [])
    if not messages:
        return {"text": ""}

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    except Exception:
        # Fallback: manual ChatML formatting (Qwen3 format)
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return {"text": "\n".join(parts)}


def train_model(
    base_model: str = "unsloth/Qwen3-8B",
    data_path: str = "data/processed/train.jsonl",
    output_dir: str = "models/rust-mentor-8b",
    lora_r: int = 32,
    lora_alpha: int = 32,
    batch_size: int = 2,
    grad_accum: int = 4,
    epochs: int = 3,
    lr: float = 2e-4,
    max_seq_length: int = 2048,
):
    """Fine-tune a model with QLoRA using Unsloth.

    This is the core training function. It:
    1. Loads the base model in 4-bit quantization
    2. Applies LoRA adapters to all attention + MLP projections
    3. Trains with SFTTrainer using cosine LR schedule
    4. Saves the adapter + tokenizer + training config

    Args:
        base_model: HuggingFace model ID (use unsloth/ prefix for speed).
        data_path: Path to training JSONL (conversations format).
        output_dir: Where to save the trained adapter.
        lora_r: LoRA rank (8 for 0.6B, 16 for 1.7B-4B, 32 for 8B).
        lora_alpha: LoRA alpha (usually same as lora_r).
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        epochs: Number of training epochs.
        lr: Learning rate.
        max_seq_length: Maximum sequence length.

    Returns:
        TrainerStats object with training metrics.
    """
    # Check GPU first
    gpu_info = _check_gpu()
    if gpu_info["device"] == "cpu":
        print("  Error: No GPU detected. Training requires a CUDA GPU.")
        print("  Options:")
        print("    - Use Google Colab (free T4 GPU)")
        print("    - Use a cloud GPU instance (A100, L4, T4)")
        return None

    # Import heavy dependencies here (not at module level)
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("  Error: Unsloth not installed.")
        print("  Install with: pip install unsloth")
        return None

    try:
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError as e:
        print(f"  Error: Missing dependency: {e}")
        print("  Install with: pip install trl datasets")
        return None

    import torch

    # Check training data exists
    if not os.path.exists(data_path):
        print(f"  Error: Training data not found: {data_path}")
        print("  Run Steps 1-2 (data collection + preprocessing) first.")
        return None

    # ── Load Model ──
    print(f"\n  Loading {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect (bf16 on A100, fp16 on T4)
        load_in_4bit=True,
    )

    # ── Apply LoRA ──
    print(f"  Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    # ── Load Dataset ──
    print(f"  Loading dataset: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"    Samples: {len(dataset)}")

    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"    After formatting: {len(dataset)} samples")

    # ── Train ──
    os.makedirs(output_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=50,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=not gpu_info["bf16"],
            bf16=gpu_info["bf16"],
            logging_steps=10,
            save_strategy="epoch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            report_to="none",
        ),
    )

    print(f"\n  Starting training...")
    print(f"    GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
    print(f"    Effective batch size: {batch_size * grad_accum}")
    print(f"    Epochs: {epochs}")
    print(f"    Learning rate: {lr}")
    print()

    trainer_stats = trainer.train()

    # Memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"\n  Training complete!")
    print(f"    Duration: {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"    Loss: {trainer_stats.metrics['train_loss']:.4f}")
    print(f"    Peak VRAM: {used_memory} GB / {gpu_info['vram_gb']} GB")

    # ── Save ──
    print(f"\n  Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "max_seq_length": max_seq_length,
        "train_samples": len(dataset),
        "train_loss": trainer_stats.metrics["train_loss"],
        "train_runtime_seconds": trainer_stats.metrics["train_runtime"],
        "peak_vram_gb": used_memory,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Model saved!")
    return trainer_stats


if __name__ == "__main__":
    train_model()
