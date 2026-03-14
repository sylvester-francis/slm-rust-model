# Documentation

## Guides

| Guide | Description |
|-------|-------------|
| [Quick Start](QUICK_START.md) | Get started in 5 minutes |
| [Colab Training](COLAB.md) | Complete Google Colab training guide |
| [Mobile Deployment](MOBILE.md) | PocketPal AI setup for Pixel 8 Pro |
| [Android App Prompt](ANDROID_APP_PROMPT.md) | Spec for building a custom Android app |

## CLI Reference

```bash
python slm.py --help          # Show all commands
python slm.py info             # System information
python slm.py pipeline         # Run complete pipeline
python slm.py collect          # Generate synthetic data (46 unique conversations)
python slm.py preprocess       # Merge synthetic + Strandset-Rust-v1
python slm.py train            # Train with QLoRA via Unsloth
python slm.py evaluate         # Evaluate model on Rust prompts
python slm.py convert          # Export to GGUF (local use)
python slm.py upload           # Upload to HuggingFace
python slm.py deploy           # Deploy to Ollama (local use)
```

> **Note**: On Colab, use `colab/colab_train_and_upload.py` instead of individual commands. It pushes GGUF directly to HuggingFace to avoid disk space limits.

## HuggingFace Repos

- **LoRA adapter**: [sylvester-francis/rust-mentor-8b](https://huggingface.co/sylvester-francis/rust-mentor-8b)
- **GGUF (Q4_K_M)**: [sylvester-francis/rust-mentor-8b-GGUF](https://huggingface.co/sylvester-francis/rust-mentor-8b-GGUF)

## License

Apache 2.0 — See [LICENSE](../LICENSE) for details.
