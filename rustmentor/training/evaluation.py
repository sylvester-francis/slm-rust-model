"""
Step 4: Evaluate the fine-tuned Rust tutor model.

Tests the model on 5 Rust tutoring prompts and scores keyword
coverage to measure domain knowledge retention.

Usage:
    from rustmentor.training import evaluate_model
    results = evaluate_model("models/rust-mentor-0.6b")
"""

import json
import os

from rustmentor.config import EVAL_PROMPTS, SYSTEM_PROMPT


def evaluate_model(model_dir: str = "models/rust-mentor-8b") -> dict:
    """Evaluate the model on Rust tutoring prompts.

    Loads the model, generates responses for each evaluation prompt,
    and scores keyword coverage. Results are saved to eval_results.json.

    Args:
        model_dir: Path to the trained model directory.

    Returns:
        Dict with keyword_accuracy, avg_response_length, and per-prompt scores.
    """
    if not os.path.exists(model_dir):
        print(f"  Error: Model not found: {model_dir}")
        print("  Run Step 3 (training) first.")
        return {}

    # Try Unsloth first, fall back to transformers
    model = None
    tokenizer = None

    try:
        from unsloth import FastLanguageModel
        print(f"\n  Loading model from {model_dir} (Unsloth)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except (ImportError, Exception):
        pass

    if model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"\n  Loading model from {model_dir} (transformers)...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, device_map="auto", torch_dtype="auto"
            )
        except ImportError:
            print("  Error: Neither unsloth nor transformers is installed.")
            print("  Install with: pip install transformers torch")
            return {}
        except Exception as e:
            print(f"  Error loading model: {e}")
            return {}

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
            {"role": "system", "content": SYSTEM_PROMPT},
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

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Score keyword coverage
        found = sum(
            1 for kw in eval_item["expected_keywords"]
            if kw.lower() in response.lower()
        )
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

        print(f"    {eval_item['category']}: {found}/{total} keywords, {len(response)} chars")

    results["avg_response_length"] = total_length // max(len(EVAL_PROMPTS), 1)
    results["keyword_accuracy"] = f"{results['keyword_matches']}/{results['total_keywords']}"

    # Save results
    eval_path = os.path.join(model_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {eval_path}")

    return results


if __name__ == "__main__":
    evaluate_model()
