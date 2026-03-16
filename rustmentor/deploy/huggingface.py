"""
Step 6a: Upload model to HuggingFace Hub with model card.

Usage:
    from rustmentor.deploy import upload_model
    upload_model("models/rust-mentor-0.6b", "username/rust-mentor-0.6b")
"""

import os
from pathlib import Path

from rustmentor.config import MODEL_CARD_TEMPLATE


def upload_model(
    model_dir: str = "models/rust-mentor-8b",
    repo_id: str = "your-username/rust-mentor-8b",
    gguf: bool = False,
):
    """Upload model to HuggingFace Hub.

    Creates the repository if it doesn't exist, generates a model card,
    and uploads all files from the model directory.

    Args:
        model_dir: Path to the model directory to upload.
        repo_id: HuggingFace repository ID (username/model-name).
        gguf: If True, upload from the GGUF directory.
    """
    if not os.path.exists(model_dir):
        print(f"  Error: Model directory not found: {model_dir}")
        return

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("  Error: huggingface_hub not installed.")
        print("  Install with: pip install huggingface_hub")
        return

    # Get HF token
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if not token:
        print("  Error: HF_TOKEN not set.")
        print("  Set via: export HF_TOKEN=your_write_token")
        print("  Or add to Colab secrets.")
        return

    api = HfApi(token=token)

    # Create repo
    format_suffix = "-GGUF" if gguf else ""
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"  Repository: {repo_id}")
    except Exception as e:
        print(f"  Repo exists or error: {e}")

    # Determine base model info for model card
    base_model_short = "Qwen3-8B"
    base_model = "Qwen/Qwen3-8B"
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    # Try to read base model from training config
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
            base_model = config.get("base_model", base_model)
            base_model_short = base_model.split("/")[-1]

    # Write model card
    card_content = MODEL_CARD_TEMPLATE.format(
        base_model=base_model,
        base_model_short=base_model_short,
        model_name=model_name,
        format_suffix=format_suffix,
        repo_id=repo_id,
    )
    card_path = os.path.join(model_dir, "README.md")
    with open(card_path, "w") as f:
        f.write(card_content)

    # Upload
    if gguf:
        gguf_dir = model_dir + "-GGUF" if not model_dir.endswith("-GGUF") else model_dir
        if os.path.exists(gguf_dir):
            model_dir = gguf_dir

    print(f"  Uploading {model_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        token=token,
    )
    print(f"  Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    upload_model()
