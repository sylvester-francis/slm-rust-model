"""
Data Preprocessing: Merge and format datasets for training.

Combines:
1. Strandset-Rust-v1 (code tasks) — subset and reformat
2. Synthetic tutor Q&A — already formatted

Outputs Qwen3 chat-template compatible JSONL.
"""

import json
import os
import random
from typing import Optional

SYSTEM_PROMPT = """You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools.

Your teaching style:
- Draw parallels to Go/Python/TypeScript concepts they already know
- Explain ownership, borrowing, and lifetimes with practical examples
- When reviewing code, explain what the borrow checker is doing and why
- Keep explanations concise with code snippets
- Guide them to write the code themselves rather than giving full solutions
- Use the Socratic method when appropriate
"""


def format_strandset_sample(sample: dict) -> Optional[dict]:
    """Convert a Strandset-Rust-v1 sample to chat format."""
    try:
        # Strandset uses various formats — adapt based on available fields
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if "instruction" in sample and "output" in sample:
            messages.append({"role": "user", "content": sample["instruction"]})
            messages.append({"role": "assistant", "content": sample["output"]})
        elif "prompt" in sample and "response" in sample:
            messages.append({"role": "user", "content": sample["prompt"]})
            messages.append({"role": "assistant", "content": sample["response"]})
        elif "messages" in sample:
            for msg in sample["messages"]:
                if msg.get("role") in ("user", "assistant"):
                    messages.append(msg)
        elif "conversations" in sample:
            for msg in sample["conversations"]:
                if msg.get("role") in ("user", "assistant"):
                    messages.append(msg)
        else:
            return None

        # Quality check: need at least user + assistant
        roles = [m["role"] for m in messages]
        if "user" not in roles or "assistant" not in roles:
            return None

        # Length check
        total_len = sum(len(m["content"]) for m in messages)
        if total_len < 50 or total_len > 15000:
            return None

        return {"conversations": messages, "category": "code_task"}

    except (KeyError, TypeError):
        return None


def load_strandset(num_samples: int = 3000) -> list:
    """Load and subset Strandset-Rust-v1 from HuggingFace."""
    try:
        from datasets import load_dataset

        print(f"  Loading Strandset-Rust-v1 from HuggingFace...")
        ds = load_dataset("Fortytwo-Network/Strandset-Rust-v1", split="train")
        print(f"  Loaded {len(ds)} total examples")

        # Shuffle and subset
        ds = ds.shuffle(seed=42)
        subset = ds.select(range(min(num_samples * 2, len(ds))))  # take extra for filtering

        formatted = []
        for sample in subset:
            result = format_strandset_sample(sample)
            if result:
                formatted.append(result)
            if len(formatted) >= num_samples:
                break

        print(f"  Formatted {len(formatted)} Strandset samples")
        return formatted

    except Exception as e:
        print(f"  ⚠️  Could not load Strandset: {e}")
        print(f"  Continuing with synthetic data only...")
        return []


def load_synthetic(path: str) -> list:
    """Load synthetic tutor data from JSONL."""
    if not os.path.exists(path):
        print(f"  ⚠️  Synthetic data not found: {path}")
        return []

    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"  Loaded {len(samples)} synthetic samples")
    return samples


def preprocess_and_merge(
    synthetic_path: str,
    strandset_samples: int = 3000,
    output_path: str = "data/processed/train.jsonl",
    max_seq_length: int = 2048,
) -> int:
    """
    Merge datasets and produce final training JSONL.

    Returns number of samples written.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load both sources
    print("\n  Loading datasets...")
    synthetic = load_synthetic(synthetic_path)
    strandset = load_strandset(strandset_samples)

    # Merge
    all_samples = synthetic + strandset
    random.seed(42)
    random.shuffle(all_samples)

    print(f"\n  Total merged: {len(all_samples)} samples")
    print(f"    Synthetic: {len(synthetic)}")
    print(f"    Strandset: {len(strandset)}")

    # Write output
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Also create size variants (matching TypeScript SLM pattern)
    variants = {
        "train_small.jsonl": min(500, len(all_samples)),
        "train_medium.jsonl": min(2000, len(all_samples)),
        "train.jsonl": len(all_samples),
    }

    for filename, count in variants.items():
        variant_path = os.path.join(os.path.dirname(output_path), filename)
        with open(variant_path, "w") as f:
            for sample in all_samples[:count]:
                f.write(json.dumps(sample) + "\n")
        print(f"  Created {filename}: {count} samples")

    return len(all_samples)


if __name__ == "__main__":
    count = preprocess_and_merge(
        synthetic_path="data/processed/rust_tutor_synthetic.jsonl",
        strandset_samples=3000,
    )
    print(f"\nTotal: {count} samples")
