# Mobile Deployment Guide — Pixel 8 Pro

Run RustMentor offline on your Pixel 8 Pro for airplane Rust learning.

## Hardware: Pixel 8 Pro

- **SoC**: Google Tensor G3
- **RAM**: 12GB
- **Storage**: Ensure 6GB+ free for model + app

## App: PocketPal AI

[PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) is an open-source app that runs GGUF models locally on Android using llama.cpp.

### Install

1. [Download from Google Play](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Or install from [GitHub](https://github.com/a-ghorbani/pocketpal-ai)

### Load Your Model

1. Open PocketPal AI
2. Tap the **+** button → **Add from Hugging Face**
3. Search for: `sylvester-francis/rust-mentor-8b-GGUF`
4. Select the **Q4_K_M** quantization (~4.5GB download)
5. Wait for download to complete (use WiFi)

### Create a Rust Tutor "Pal"

1. Go to **Pals** tab
2. Tap **Create New Pal**
3. Name: `RustMentor`
4. System prompt:

```
You are RustMentor, an expert Rust programming tutor. I'm an experienced Go, Python, and TypeScript developer learning Rust by building a CLI tool.

Guide me through Rust concepts — ownership, borrowing, lifetimes, error handling, traits, pattern matching, async, smart pointers — using practical examples.

When I share code, review it and explain what the borrow checker is doing.

Draw parallels to Go/Python/TypeScript concepts I already know.

Keep explanations concise with code snippets. Don't write the full solution — help me understand so I can write it myself.
```

5. Select your downloaded model
6. Save the Pal

### Verify Offline Mode

1. Enable **Airplane Mode**
2. Open PocketPal → select RustMentor Pal
3. Ask: "Explain Rust's ownership model compared to Go"
4. Confirm it responds without internet

## Flight Preparation Checklist

**Before the flight:**

- [ ] Charge phone to 100%
- [ ] Download model on WiFi (4.5GB)
- [ ] Load model once to verify it works
- [ ] Test in airplane mode
- [ ] Create RustMentor Pal with system prompt
- [ ] Optional: bring a battery pack
- [ ] Optional: have Rust code snippets saved in Notes app

**Tips:**

- Keep screen brightness low to extend battery
- Close all background apps before loading the model
- If responses are slow, try reducing context size in PocketPal settings

## Suggested Learning Flow (5.5hr Flight)

### Hour 1: Foundations
- "Explain ownership vs Go's value/pointer model"
- "What's the difference between & and &mut?"
- "When should I clone vs borrow?"

### Hour 2: Error Handling & Types
- "How does Result compare to Go's error pattern?"
- "Show me how to use the ? operator"
- "Why does Rust have both String and &str?"

### Hour 3: Building Blocks
- "How do structs and impl blocks work?"
- "Explain traits vs Go interfaces"
- "How do I structure a Cargo project?"

### Hour 4: Practical Patterns
- "How do I parse CLI args with clap?"
- "Show me iterator chains for processing data"
- "How do I serialize structs to JSON with serde?"

### Hour 5: Advanced & Code Review
- "How does async/await work compared to goroutines?"
- "What are Box, Rc, and Arc?"
- Share code you've been drafting and ask for review
- "What would the borrow checker say about this?"

### Last 30 min: Wrap Up
- Ask for a summary of key concepts
- Note down topics to explore further

## License

Apache 2.0 — See [LICENSE](../LICENSE) for details.
