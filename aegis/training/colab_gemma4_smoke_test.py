#!/usr/bin/env python3
"""
Aegis Mobile -- Gemma 4 toolchain smoke test (Days 1-3 de-risk gate).

Validates the full pipeline end-to-end with a 3-example dataset before we
commit the rest of the 19-day plan to it.

Gates (in order; abort on first failure):
  1. Gemma 4 E2B base weights pullable via Unsloth
  2. Unsloth fine-tunes the architecture for >=1 step
  3. LiteRT export produces a .litertlm artifact
  4. (manual) artifact loads on Pixel via RustSensei's loader path

If any gate fails, fall back to Path A: Gemma 4 4B via Ollama/llama.cpp on
desktop, with the phone as a thin client. Special-Tech eligibility shifts
from LiteRT to Ollama in that case.

Usage on Colab T4:
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    !git checkout claude/kaggle-competition-review-bYNNA
    import os
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python aegis/training/colab_gemma4_smoke_test.py
"""

import os
import subprocess
import sys
import time

# Confirm against https://www.kaggle.com/models/google/gemma-4 before running.
# Google's pattern is google/gemma-N-<variant>-it; Unsloth mirrors as
# unsloth/gemma-N-<variant>-it. If unsloth has not mirrored yet, swap to the
# google/ id and let Unsloth load it directly.
BASE_MODEL = "unsloth/gemma-4-e2b-it"
FULL_PRECISION_MODEL = "google/gemma-4-e2b-it"
LITERT_MODEL_SIZE = "e2b"

LORA_R = 8
EPOCHS = 1
MAX_SEQ_LENGTH = 1024

SMOKE_EXAMPLES = [
    {
        "instruction": "Classify this caller and explain.",
        "input": "Hello, this is the IRS. You owe back taxes. Pay $500 in iTunes "
                 "gift cards within the hour or you will be arrested.",
        "output": "Tech-support / impersonation scam. The IRS never demands "
                  "payment via gift cards and never threatens immediate arrest. "
                  "Hang up and do not engage.",
    },
    {
        "instruction": "Is this SMS legitimate?",
        "input": "Your Amazon order #112-9000 is delayed. Confirm at "
                 "http://amzn-track.tk/x",
        "output": "Phishing. The domain amzn-track.tk is not owned by Amazon. "
                  "Do not tap the link. Block the sender.",
    },
    {
        "instruction": "Classify this caller.",
        "input": "Hi, this is Microsoft Support. We detected a virus on your "
                 "computer and need you to install AnyDesk so we can fix it.",
        "output": "Tech-support scam. Microsoft does not cold-call users, and "
                  "any caller asking you to install remote-access software "
                  "(AnyDesk, TeamViewer, Quick Assist) is attempting fraud.",
    },
]


def sh(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def py(label, script):
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}\n")
    rc = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    ).returncode
    if rc != 0:
        print(f"\nGATE FAILED: {label}")
        sys.exit(rc)


def main():
    t0 = time.time()
    print(f"Aegis Mobile -- Gemma 4 toolchain smoke test")
    print(f"  base: {BASE_MODEL}")
    print(f"  size: {LITERT_MODEL_SIZE}")

    # Phase 1: train-side deps. Must come before any litert-torch install
    # because tensorflow/keras get uninstalled in phase 4.
    print("\nInstalling training deps...")
    sh("pip install -q unsloth trl peft accelerate bitsandbytes "
       "datasets huggingface_hub hf_transfer")
    sh("pip install -q 'transformers>=4.51.3'")

    # Gate 1: pull base
    py("Gate 1/4 -- pull Gemma 4 base", f"""
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name={BASE_MODEL!r},
    max_seq_length={MAX_SEQ_LENGTH},
    load_in_4bit=True,
)
print('  ok: base loaded')
""")

    # Gate 2: smoke fine-tune (write examples to disk so the subscript stays clean)
    os.makedirs("aegis/training/_smoke", exist_ok=True)
    import json
    with open("aegis/training/_smoke/examples.jsonl", "w") as f:
        for ex in SMOKE_EXAMPLES:
            f.write(json.dumps(ex) + "\n")

    py("Gate 2/4 -- smoke fine-tune (1 epoch, 3 examples)", f"""
import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

with open('aegis/training/_smoke/examples.jsonl') as f:
    rows = [json.loads(l) for l in f]

def fmt(ex):
    msg = (
        '<start_of_turn>user\\n'
        + ex['instruction'] + '\\n' + ex['input']
        + '<end_of_turn>\\n<start_of_turn>model\\n'
        + ex['output']
        + '<end_of_turn>'
    )
    return {{'text': msg}}

ds = Dataset.from_list([fmt(e) for e in rows])

model, tok = FastLanguageModel.from_pretrained(
    model_name={BASE_MODEL!r},
    max_seq_length={MAX_SEQ_LENGTH},
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r={LORA_R}, lora_alpha={LORA_R})

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    dataset_text_field='text',
    max_seq_length={MAX_SEQ_LENGTH},
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs={EPOCHS},
        learning_rate=2e-4,
        logging_steps=1,
        output_dir='models/aegis-smoke',
        save_strategy='no',
        report_to='none',
    ),
)
trainer.train()
model.save_pretrained('models/aegis-smoke')
tok.save_pretrained('models/aegis-smoke')
print('  ok: fine-tune complete')
""")

    # Gate 3: LiteRT export. The existing rustmentor.export.convert_gemma3_to_litert
    # may or may not generalize to Gemma 4 -- if it raises an architecture error,
    # port it to a convert_gemma4_to_litert variant. That porting work is the
    # deliverable for Day 2 if this gate fails.
    print("\nSwapping deps for LiteRT export...")
    sh("pip uninstall -y tensorflow tensorflow-cpu keras -q 2>/dev/null || true")
    sh("pip install -q litert-torch 'protobuf>=5.26,<7.0'")
    sh("pip install -q 'torchao==0.11.0' --force-reinstall --no-deps")

    py("Gate 3/4 -- LiteRT export", f"""
import sys, os
sys.path.insert(0, os.getcwd())
try:
    from rustmentor.export import convert_gemma3_to_litert as convert
except Exception as e:
    print(f'  import failed: {{e}}')
    sys.exit(1)

try:
    out = convert(
        model_dir='models/aegis-smoke',
        output_dir='models/aegis-smoke-litert',
        model_size={LITERT_MODEL_SIZE!r},
        output_name_prefix='aegis-smoke',
        quantization='dynamic_int8',
        kv_cache_max_len=2048,
        base_model={FULL_PRECISION_MODEL!r},
    )
except Exception as e:
    print(f'  convert raised: {{e}}')
    print('  ACTION: port rustmentor.export.convert_gemma3_to_litert to a')
    print('          convert_gemma4_to_litert variant for the Gemma 4 arch.')
    sys.exit(1)

if not out:
    print('  convert returned empty')
    sys.exit(1)
print(f'  ok: {{out}}')
""")

    print(f"\n{'=' * 60}")
    print(f"All automated gates passed in {time.time() - t0:.0f}s")
    print(f"{'=' * 60}")
    print("Gate 4 (manual): side-load models/aegis-smoke-litert/*.litertlm")
    print("onto a Pixel via RustSensei's ModelManager + LiteRtEngine path and")
    print("confirm a single-token generation succeeds. Once green, proceed to")
    print("Days 4-6: build the real fine-tune corpus.")


if __name__ == "__main__":
    main()
