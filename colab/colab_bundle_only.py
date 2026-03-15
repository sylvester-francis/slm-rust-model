#!/usr/bin/env python3
"""
Download .tflite + tokenizer from HuggingFace, bundle into .litertlm, upload.

No merging. No conversion. Just bundle + upload.

Usage:
    !git clone https://github.com/sylvester-francis/slm-rust-model.git
    %cd slm-rust-model
    import os; from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python colab/colab_bundle_only.py
"""

import os
import sys
import subprocess
import glob
import time

# ── CONFIG ──────────────────────────────────────────────
HF_USERNAME = "sylvester-francis"
VARIANTS = ["0.6b", "1.7b", "4b"]
KV_CACHE_LEN = 2048
# ────────────────────────────────────────────────────────


def run(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    total_start = time.time()
    token = os.environ.get("HF_TOKEN", "")

    print("🦀 RustMentor — Bundle .tflite → .litertlm")
    print(f"   Variants: {VARIANTS}")
    print()

    # Install deps
    run("pip install -q litert-torch huggingface_hub 'protobuf>=5.26,<7.0'")
    run("pip install -q 'torchao==0.11.0' --force-reinstall --no-deps")

    from huggingface_hub import hf_hub_download, HfApi, create_repo

    api = HfApi(token=token)
    username = api.whoami()["name"]

    for variant in VARIANTS:
        litert_repo = f"{HF_USERNAME}/rust-mentor-{variant}-LiteRT"
        adapter_repo = f"{HF_USERNAME}/rust-mentor-{variant}"
        work_dir = f"models/rust-mentor-{variant}-litert"
        os.makedirs(work_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  {variant}: Download .tflite + tokenizer")
        print(f"{'=' * 60}\n")

        # Download .tflite from LiteRT repo
        try:
            files = api.list_repo_files(litert_repo)
            tflite_files = [f for f in files if f.endswith(".tflite")]
            if not tflite_files:
                print(f"  No .tflite found in {litert_repo}, skipping")
                continue
            for tf in tflite_files:
                print(f"  Downloading {tf}...")
                hf_hub_download(litert_repo, tf, local_dir=work_dir, token=token)
        except Exception as e:
            print(f"  Failed to download from {litert_repo}: {e}")
            continue

        # Download tokenizer.json from adapter repo
        try:
            print(f"  Downloading tokenizer.json from {adapter_repo}...")
            hf_hub_download(adapter_repo, "tokenizer.json", local_dir=work_dir, token=token)
        except Exception as e:
            print(f"  Failed to download tokenizer: {e}")
            continue

        # Bundle into .litertlm
        print(f"\n{'=' * 60}")
        print(f"  {variant}: Bundle → .litertlm")
        print(f"{'=' * 60}\n")

        tokenizer_path = os.path.join(work_dir, "tokenizer.json")
        run(
            f"python scripts/bundle_litertlm.py"
            f" --tflite={work_dir}"
            f" --tokenizer={tokenizer_path}"
            f" --output={work_dir}"
            f" --context_length={KV_CACHE_LEN}"
            f" --model_type=qwen3"
        )

        # Upload .litertlm to HF
        print(f"\n{'=' * 60}")
        print(f"  {variant}: Upload to HuggingFace")
        print(f"{'=' * 60}\n")

        create_repo(litert_repo, token=token, exist_ok=True, repo_type="model")
        for f in glob.glob(os.path.join(work_dir, "*.litertlm")):
            fname = os.path.basename(f)
            size_mb = os.path.getsize(f) / 1024 / 1024
            print(f"  Uploading {fname} ({size_mb:.0f} MB)...")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=fname,
                repo_id=litert_repo,
                token=token,
            )
        print(f"  ✅ https://huggingface.co/{litert_repo}")

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"🦀 Done! ({total:.0f}s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
