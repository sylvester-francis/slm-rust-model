# Contributing to RustMentor SLM

Thank you for your interest in contributing! Here's how to get started.

## Ways to Contribute

### Training Data
The highest-impact contribution is improving training data quality:

1. **Add tutor conversations** — Create new Q&A pairs in `scripts/data_collection.py`
   - Cover Rust concepts not yet represented
   - Include code review examples with borrow checker explanations
   - Draw comparisons to Go, Python, or TypeScript

2. **Improve existing examples** — Review and refine the seed conversations
   - Fix code errors or outdated patterns
   - Add more practical, real-world examples
   - Ensure Rust 2024 edition compatibility

### Code Improvements
- Improve training efficiency or add new model support
- Add evaluation benchmarks (e.g., Rust-specific HumanEval)
- Improve GGUF export or mobile deployment tooling

### Documentation
- Add tutorials for different deployment targets
- Improve the Colab guide
- Write guides for other Android AI apps beyond PocketPal

## Development Setup

```bash
# Clone
git clone https://github.com/sylvester-francis/slm-rust-model.git
cd slm-rust-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python slm.py info
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Test on Colab if training-related
5. Update documentation as needed
6. Submit a PR with a clear description

## Code Style

- Follow existing patterns in the codebase
- Use type hints for function signatures
- Add docstrings to new functions
- Keep the CLI interface consistent with the existing commands

## Adding New Tutor Conversations

Add entries to the `RUST_TUTOR_CONVERSATIONS` list in `scripts/data_collection.py`:

```python
{
    "category": "your_category",
    "conversations": [
        {"role": "user", "content": "Question from a Go/Python/TS developer..."},
        {"role": "assistant", "content": "Teaching response with code examples..."},
    ]
}
```

Guidelines for tutor conversations:
- Always assume the student knows Go, Python, and TypeScript
- Draw explicit comparisons to equivalent patterns in those languages
- Include runnable Rust code snippets
- Explain what the borrow checker is doing when relevant
- Use the Socratic method — ask follow-up questions
- Reference practical CLI tool patterns when possible

## Questions?

Open an issue or reach out — all contributions welcome!
