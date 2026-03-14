# Mobile Deployment Guide — Pixel 8 Pro

Run RustMentor offline on your Pixel 8 Pro for airplane Rust learning.

## Hardware: Pixel 8 Pro

- **SoC**: Google Tensor G3
- **RAM**: 12GB
- **Storage**: Ensure 6GB+ free for model + app
- **Battery**: Expect 3-4 hours of active inference on full charge

## App: PocketPal AI

PocketPal AI is an open-source app that runs GGUF models locally on Android using llama.cpp.

### Install

1. [Download from Google Play](https://play.google.com/store/apps/details?id=com.pocketpalai)
2. Or install APK from [GitHub releases](https://github.com/a-ghorbani/pocketpal-ai)

### Load Your Model

1. Open PocketPal AI
2. Tap the **+** button → **Add from Hugging Face**
3. Sign in to your HuggingFace account
4. Search for: `YOUR_USERNAME/rust-mentor-8b-GGUF`
5. Select the **Q4_K_M** quantization (~4.5GB download)
6. Wait for download to complete (use WiFi)

### Create a Rust Tutor "Pal"

PocketPal's Pals feature lets you set a system prompt:

1. Go to **Pals** tab
2. Tap **Create New Pal**
3. Name: `RustMentor`
4. System prompt:

```
You are RustMentor, an expert Rust programming tutor. I'm an experienced Go, Python, and TypeScript developer learning Rust by building a CLI password auditor tool called passaudit.

Guide me through Rust concepts — ownership, borrowing, lifetimes, error handling, traits, pattern matching — using practical examples related to my project.

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

## Performance Expectations

| Setting | Value |
|---------|-------|
| Model | Qwen3-8B Q4_K_M |
| File size | ~4.5GB |
| RAM usage | ~5-6GB |
| Speed | ~5-8 tokens/sec |
| Response time | 5-15 seconds for typical answers |
| Battery drain | ~20-25% per hour of active use |

## Flight Preparation Checklist

**Before the flight:**

- [ ] Charge phone to 100%
- [ ] Download model on WiFi (4.5GB)
- [ ] Load model once to verify it works
- [ ] Test in airplane mode
- [ ] Create RustMentor Pal with system prompt
- [ ] Optional: bring a battery pack for 5.5hr flight
- [ ] Optional: have Rust code snippets saved in Notes app

**Optimization tips:**

- Keep screen brightness low (30-40%)
- Close all background apps before loading the model
- Enable PocketPal's "Auto Offload/Load" to manage memory
- If responses are slow, try reducing context size in PocketPal settings

## Suggested Learning Flow (5.5hr Flight)

### Hour 1: Foundations
- "Explain ownership vs Go's value/pointer model"
- "What's the difference between & and &mut?"
- "When should I clone vs borrow?"

### Hour 2: Error Handling
- "How does Result compare to Go's error pattern?"
- "Show me how to use the ? operator"
- "How do I create custom error types?"

### Hour 3: Building Blocks
- "How do structs and impl blocks work?"
- "Explain traits vs Go interfaces"
- "How do I structure a Cargo project?"

### Hour 4: Practical Patterns
- "How do I parse CLI args with clap?"
- "Show me iterator chains for processing data"
- "How do I write tests in Rust?"

### Hour 5: Code Review
- Share code you've been drafting and ask for review
- "Review this function — is it idiomatic?"
- "What would the borrow checker say about this?"

### Last 30 min: Wrap Up
- Ask for a summary of key concepts
- Note down topics to explore further
- Draft your next passaudit module

## Alternative: Google AI Edge Gallery

If PocketPal doesn't work for your setup:

1. Install [AI Edge Gallery](https://github.com/google-ai-edge/gallery) APK
2. Download a Gemma model (tested on Pixel 8 Pro)
3. Note: No custom model support — you'd use a pre-built model, not your fine-tune

PocketPal is recommended because it supports custom GGUF models.

## Alternative: MLC Chat

Another option for running models on Android:

1. Download [MLC Chat APK](https://github.com/mlc-ai/mlc-llm/releases)
2. Supports pre-configured models like Gemma 2B, Phi-2
3. Less flexible than PocketPal for custom models
