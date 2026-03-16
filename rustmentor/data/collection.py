"""
Step 1: Synthetic Rust Tutor Dataset Generator

Generates Rust tutoring conversations covering 28+ topics:
  ownership, borrowing, lifetimes, error handling, traits, generics,
  pattern matching, iterators, closures, async/await, smart pointers,
  macros, serde, testing, CLI tooling, and more.

Each conversation is prefixed with the RustMentor system prompt and
targets developers coming from Go, Python, and TypeScript.

Usage:
    from rustmentor.data import generate_rust_dataset
    count = generate_rust_dataset("data/processed/synthetic.jsonl", num_samples=500)
"""

import json
import os
from typing import List, Dict

from rustmentor.config import SYSTEM_PROMPT


# ── Seed Conversations ─────────────────────────────────────
# 46 hand-written Q&A pairs across 28 Rust topics.
# Stored in rustmentor/data/seeds.py (~112KB of conversation data).

def _load_seed_conversations() -> List[Dict]:
    """Load seed conversations from the seeds module."""
    try:
        from rustmentor.data.seeds import RUST_TUTOR_CONVERSATIONS
        return RUST_TUTOR_CONVERSATIONS
    except ImportError:
        print("  Warning: Could not import seed data from rustmentor.data.seeds")
        print("  Using minimal built-in seed set.")
        return _MINIMAL_SEEDS


# Minimal fallback seeds if the scripts module is unavailable
_MINIMAL_SEEDS = [
    {
        "category": "ownership",
        "conversations": [
            {
                "role": "user",
                "content": "In Go, I just pass values or pointers. What's this ownership thing in Rust?",
            },
            {
                "role": "assistant",
                "content": (
                    "Great question! In Rust, each value has exactly one owner. "
                    "When the owner goes out of scope, the value is dropped. "
                    "Ownership can be transferred (moved) or borrowed.\n\n"
                    "```rust\n"
                    'let s1 = String::from("hello");\n'
                    "let s2 = s1;  // ownership MOVED to s2\n"
                    "// println!(\"{}\", s1);  // compile error!\n"
                    "```\n\n"
                    "In Go, `s2 := s1` copies the string. In Rust, heap types "
                    "like `String` are *moved* instead. Use `&` to borrow."
                ),
            },
        ],
    },
    {
        "category": "error_handling",
        "conversations": [
            {
                "role": "user",
                "content": "How do I handle errors in Rust? I'm used to Go's if err != nil.",
            },
            {
                "role": "assistant",
                "content": (
                    "Rust uses `Result<T, E>` instead of Go's `(T, error)` tuple. "
                    "The `?` operator propagates errors automatically:\n\n"
                    "```rust\n"
                    "fn read_config() -> Result<String, std::io::Error> {\n"
                    '    let content = std::fs::read_to_string("config.toml")?;\n'
                    "    Ok(content)\n"
                    "}\n"
                    "```\n\n"
                    "The `?` is like `if err != nil { return err }` in one character. "
                    "And the compiler *forces* you to handle every Result."
                ),
            },
        ],
    },
    {
        "category": "traits",
        "conversations": [
            {
                "role": "user",
                "content": "How are Rust traits different from Go interfaces?",
            },
            {
                "role": "assistant",
                "content": (
                    "Traits are Rust's version of interfaces, but more powerful. "
                    "Key differences:\n\n"
                    "- **Explicit**: you must write `impl Trait for Type` (Go is implicit)\n"
                    "- **Default methods**: traits can carry implementations\n"
                    "- **Generic bounds**: `fn print<T: Display>(item: &T)` constrains at compile time\n\n"
                    "```rust\n"
                    "trait Summary {\n"
                    "    fn summarize(&self) -> String;\n"
                    "    fn preview(&self) -> String { format!(\"{}...\", &self.summarize()[..50]) }\n"
                    "}\n"
                    "```"
                ),
            },
        ],
    },
]


def generate_rust_dataset(
    output_path: str,
    num_samples: int = 500,
    system_prompt: str = None,
) -> int:
    """Generate a synthetic Rust tutor training dataset.

    Takes seed conversations and duplicates them to reach the target
    sample count. Each conversation is prefixed with the system prompt.

    Args:
        output_path: Where to write the JSONL file.
        num_samples: Target number of training samples.
        system_prompt: Override the default RustMentor system prompt.

    Returns:
        Number of samples written.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    seeds = _load_seed_conversations()
    if not seeds:
        print("  Error: No seed conversations available.")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        idx = 0
        while count < num_samples:
            seed = seeds[idx % len(seeds)]
            idx += 1

            # Normalize field name: some seeds use "messages", others "conversations"
            messages = seed.get("conversations") or seed.get("messages", [])
            if not messages:
                continue

            # Prepend system prompt if not already present
            if messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system_prompt}] + messages

            sample = {
                "conversations": messages,
                "category": seed.get("category", "general"),
            }
            f.write(json.dumps(sample) + "\n")
            count += 1

    print(f"  Generated {count} samples from {len(seeds)} seed conversations")
    return count


if __name__ == "__main__":
    count = generate_rust_dataset(
        output_path="data/processed/rust_tutor_synthetic.jsonl",
        num_samples=500,
    )
    print(f"Total: {count} samples")
