"""Deploy model to Ollama."""

import os
import subprocess


MODELFILE_TEMPLATE = """FROM {gguf_path}

SYSTEM \"\"\"You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools.

Draw parallels to Go/Python/TypeScript concepts. Explain ownership, borrowing, and lifetimes with practical examples. When reviewing code, explain what the borrow checker is doing. Keep explanations concise with code snippets. Guide them to write code themselves.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
"""


def deploy_to_ollama(
    model_name: str = "rust-mentor-8b",
    model_dir: str = "models/rust-mentor-8b",
):
    """Deploy GGUF model to Ollama."""
    gguf_dir = model_dir + "-GGUF"
    gguf_files = []

    for d in [gguf_dir, model_dir]:
        if os.path.exists(d):
            gguf_files = [f for f in os.listdir(d) if f.endswith(".gguf")]
            if gguf_files:
                gguf_path = os.path.join(d, gguf_files[0])
                break

    if not gguf_files:
        print("  ❌ No GGUF file found. Run `python slm.py convert` first.")
        return

    # Write Modelfile
    modelfile_path = os.path.join(os.path.dirname(gguf_path), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(MODELFILE_TEMPLATE.format(gguf_path=os.path.abspath(gguf_path)))

    print(f"  Modelfile: {modelfile_path}")
    print(f"  GGUF: {gguf_path}")

    # Create Ollama model
    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            check=True,
        )
        print(f"\n  ✅ Model created: {model_name}")
        print(f"  Run with: ollama run {model_name}")
    except FileNotFoundError:
        print("  ❌ Ollama not found. Install from https://ollama.ai")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Ollama error: {e}")


if __name__ == "__main__":
    deploy_to_ollama()
