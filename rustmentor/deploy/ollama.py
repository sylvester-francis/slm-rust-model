"""
Step 6b: Deploy GGUF model to Ollama for local inference.

Usage:
    from rustmentor.deploy import deploy_to_ollama
    deploy_to_ollama("rust-mentor-0.6b", "models/rust-mentor-0.6b")
"""

import os
import subprocess

from rustmentor.config import SYSTEM_PROMPT, OLLAMA_MODELFILE_TEMPLATE


def deploy_to_ollama(
    model_name: str = "rust-mentor-8b",
    model_dir: str = "models/rust-mentor-8b",
):
    """Deploy GGUF model to Ollama.

    Finds the GGUF file, writes an Ollama Modelfile with the
    RustMentor system prompt, and creates the model.

    Args:
        model_name: Name for the Ollama model.
        model_dir: Base model directory (looks in dir and dir-GGUF).
    """
    gguf_dir = model_dir + "-GGUF"
    gguf_path = None

    for d in [gguf_dir, model_dir]:
        if os.path.exists(d):
            gguf_files = [f for f in os.listdir(d) if f.endswith(".gguf")]
            if gguf_files:
                gguf_path = os.path.join(d, gguf_files[0])
                break

    if not gguf_path:
        print("  Error: No GGUF file found.")
        print(f"  Searched: {gguf_dir}, {model_dir}")
        print("  Run Step 5a (GGUF conversion) first.")
        return

    # Write Modelfile
    modelfile_content = OLLAMA_MODELFILE_TEMPLATE.format(
        gguf_path=os.path.abspath(gguf_path),
        system_prompt=SYSTEM_PROMPT,
    )
    modelfile_path = os.path.join(os.path.dirname(gguf_path), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"  Modelfile: {modelfile_path}")
    print(f"  GGUF: {gguf_path}")

    # Create Ollama model
    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            check=True,
        )
        print(f"\n  Model created: {model_name}")
        print(f"  Run with: ollama run {model_name}")
    except FileNotFoundError:
        print("  Error: Ollama not found.")
        print("  Install from https://ollama.ai")
    except subprocess.CalledProcessError as e:
        print(f"  Error: Ollama command failed: {e}")


if __name__ == "__main__":
    deploy_to_ollama()
