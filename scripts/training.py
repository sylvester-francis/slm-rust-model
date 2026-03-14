"""
Training: QLoRA fine-tuning with Unsloth optimization.

Supports:
- Qwen3-8B (A100 required, recommended)
- Qwen3-4B (T4 compatible, lighter)
- Custom base models via --model flag

Uses Unsloth for 2x faster training with 70% less VRAM.
"""

import json
import os
import torch
from pathlib import Path


def format_chat_template(example, tokenizer):
    """Format a conversation example using the model's chat template."""
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
        # Fallback: manual formatting
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
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
    """
    Fine-tune a model with QLoRA using Unsloth.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    # ── Load Model ──
    print(f"\n📦 Loading {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect (bf16 on A100)
        load_in_4bit=True,
    )

    # ── Apply LoRA ──
    print(f"🔧 Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
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
    print(f"📊 Loading dataset: {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"   Samples: {len(dataset)}")

    # Format using chat template
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Filter empty
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"   After formatting: {len(dataset)} samples")

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
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
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

    print(f"\n🚀 Starting training...")
    print(f"   Effective batch size: {batch_size * grad_accum}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_mem / 1024 / 1024 / 1024, 3)
    print(f"   GPU: {gpu_stats.name} ({max_memory} GB)")
    print(f"   Reserved: {start_gpu_memory} GB")
    print()

    trainer_stats = trainer.train()

    # Memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"\n📊 Training complete!")
    print(f"   Duration: {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"   Loss: {trainer_stats.metrics['train_loss']:.4f}")
    print(f"   Peak VRAM: {used_memory} GB / {max_memory} GB")

    # ── Save ──
    print(f"\n💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config
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

    print(f"✅ Model saved!")
    return trainer_stats


if __name__ == "__main__":
    train_model()
