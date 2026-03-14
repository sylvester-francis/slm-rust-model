"""
Evaluation: Test the fine-tuned Rust tutor model on sample prompts.
"""

import json
import os


EVAL_PROMPTS = [
    {
        "category": "ownership",
        "prompt": "Explain Rust's ownership model to someone who knows Go.",
        "expected_keywords": ["owner", "move", "borrow", "scope", "drop"],
    },
    {
        "category": "error_handling",
        "prompt": "How do I handle errors in Rust? I'm used to Go's if err != nil pattern.",
        "expected_keywords": ["Result", "Option", "?", "unwrap", "Ok", "Err"],
    },
    {
        "category": "code_review",
        "prompt": "Review this Rust code:\n```rust\nfn get_longest(a: String, b: String) -> String {\n    if a.len() > b.len() { a } else { b }\n}\n```",
        "expected_keywords": ["borrow", "&str", "reference", "ownership"],
    },
    {
        "category": "traits",
        "prompt": "How are Rust traits different from Go interfaces?",
        "expected_keywords": ["trait", "impl", "explicit", "default", "generic"],
    },
    {
        "category": "cli",
        "prompt": "How do I build a CLI tool in Rust with subcommands?",
        "expected_keywords": ["clap", "derive", "Parser", "Subcommand", "cargo"],
    },
]


def evaluate_model(model_dir: str = "models/rust-mentor-8b") -> dict:
    """Evaluate the model on Rust tutoring prompts."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading model from {model_dir}...")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto", torch_dtype="auto"
        )

    results = {
        "total_prompts": len(EVAL_PROMPTS),
        "keyword_matches": 0,
        "total_keywords": 0,
        "avg_response_length": 0,
        "responses": [],
    }

    total_length = 0

    for eval_item in EVAL_PROMPTS:
        messages = [
            {"role": "system", "content": "You are RustMentor, an expert Rust programming tutor."},
            {"role": "user", "content": eval_item["prompt"]},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Score keyword coverage
        found = sum(1 for kw in eval_item["expected_keywords"] if kw.lower() in response.lower())
        total = len(eval_item["expected_keywords"])

        results["keyword_matches"] += found
        results["total_keywords"] += total
        total_length += len(response)

        results["responses"].append({
            "category": eval_item["category"],
            "prompt": eval_item["prompt"][:80] + "...",
            "response_length": len(response),
            "keyword_score": f"{found}/{total}",
            "response_preview": response[:200] + "...",
        })

        print(f"  ✅ {eval_item['category']}: {found}/{total} keywords, {len(response)} chars")

    results["avg_response_length"] = total_length // max(len(EVAL_PROMPTS), 1)
    results["keyword_accuracy"] = f"{results['keyword_matches']}/{results['total_keywords']}"

    # Save results
    eval_path = os.path.join(model_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {eval_path}")

    return results


if __name__ == "__main__":
    evaluate_model()
