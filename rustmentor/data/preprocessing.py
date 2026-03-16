"""
Step 2: Data Preprocessing — Merge and format datasets for training.

Combines two data sources:
  1. Strandset-Rust-v1 (Rust code tasks from HuggingFace)
  2. Synthetic RustMentor tutor Q&A (generated in Step 1)

Outputs chat-template compatible JSONL for Qwen3/Gemma3 training.

Usage:
    from rustmentor.data import preprocess_and_merge
    count = preprocess_and_merge(
        synthetic_path="data/processed/rust_tutor_synthetic.jsonl",
        strandset_samples=3000,
        output_path="data/processed/train.jsonl",
    )
"""

import json
import os
import random
from typing import Optional

from rustmentor.config import SYSTEM_PROMPT


def format_strandset_sample(sample: dict) -> Optional[dict]:
    """Convert a Strandset-Rust-v1 sample to chat format.

    Handles multiple field formats (instruction/output, prompt/response,
    messages, conversations) and applies quality filters.
    """
    try:
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

        # Length check: filter out too-short or too-long samples
        total_len = sum(len(m["content"]) for m in messages)
        if total_len < 50 or total_len > 15000:
            return None

        return {"conversations": messages, "category": "code_task"}

    except (KeyError, TypeError):
        return None


def load_strandset(num_samples: int = 3000) -> list:
    """Load and subset Strandset-Rust-v1 from HuggingFace.

    Downloads the dataset, shuffles, formats, and filters to the
    target count. Returns empty list if download fails.
    """
    try:
        from datasets import load_dataset

        print(f"  Loading Strandset-Rust-v1 from HuggingFace...")
        ds = load_dataset("Fortytwo-Network/Strandset-Rust-v1", split="train")
        print(f"  Loaded {len(ds)} total examples")

        ds = ds.shuffle(seed=42)
        subset = ds.select(range(min(num_samples * 2, len(ds))))

        formatted = []
        for sample in subset:
            result = format_strandset_sample(sample)
            if result:
                formatted.append(result)
            if len(formatted) >= num_samples:
                break

        print(f"  Formatted {len(formatted)} Strandset samples")
        return formatted

    except ImportError:
        print("  Warning: 'datasets' package not installed.")
        print("  Install with: pip install datasets")
        print("  Continuing with synthetic data only...")
        return []
    except Exception as e:
        print(f"  Warning: Could not load Strandset: {e}")
        print("  Continuing with synthetic data only...")
        return []


def load_synthetic(path: str) -> list:
    """Load synthetic tutor data from JSONL file."""
    if not os.path.exists(path):
        print(f"  Warning: Synthetic data not found: {path}")
        print("  Run Step 1 (data collection) first.")
        return []

    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"  Loaded {len(samples)} synthetic samples")
    return samples


def preprocess_and_merge(
    synthetic_path: str,
    strandset_samples: int = 3000,
    output_path: str = "data/processed/train.jsonl",
    max_seq_length: int = 2048,
) -> int:
    """Merge datasets and produce final training JSONL.

    Combines synthetic tutor Q&A with Strandset-Rust-v1 code tasks,
    shuffles, and writes output files including size variants.

    Args:
        synthetic_path: Path to the synthetic JSONL from Step 1.
        strandset_samples: How many Strandset samples to include.
        output_path: Where to write the merged JSONL.
        max_seq_length: Maximum sequence length (for reference).

    Returns:
        Total number of samples written.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n  Loading datasets...")
    synthetic = load_synthetic(synthetic_path)
    strandset = load_strandset(strandset_samples)

    all_samples = synthetic + strandset
    if not all_samples:
        print("  Error: No training data available.")
        print("  Run Step 1 (data collection) first, or check network for Strandset.")
        return 0

    random.seed(42)
    random.shuffle(all_samples)

    print(f"\n  Total merged: {len(all_samples)} samples")
    print(f"    Synthetic: {len(synthetic)}")
    print(f"    Strandset: {len(strandset)}")

    # Write full dataset
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Create size variants for quick iteration
    variants = {
        "train_small.jsonl": min(500, len(all_samples)),
        "train_medium.jsonl": min(2000, len(all_samples)),
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
