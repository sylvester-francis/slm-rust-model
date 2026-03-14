# RustMentor Android App — Build Prompt

Use this prompt with Claude or another AI assistant to generate the Android app.

---

## Prompt

Build me a native Android app called **RustMentor** — an offline Rust programming tutor that runs a GGUF language model locally on-device using llama.cpp.

### Core Concept

A chat-based Rust tutor that runs entirely offline on a Pixel 8 Pro. The user is an experienced Go/Python/TypeScript developer learning Rust. The app loads a Q4_K_M quantized GGUF model (~4.5GB) and performs inference locally — no internet required.

### Tech Stack

- **Language**: Kotlin
- **UI**: Jetpack Compose (Material 3, dynamic color)
- **Local LLM inference**: [llama.cpp Android bindings](https://github.com/anthropics/anthropic-cookbook) via the [android-llama.cpp](https://github.com/nicholasgasior/android-llama.cpp) library or build llama.cpp as a native library with CMake/NDK
- **Model format**: GGUF (Q4_K_M quantization)
- **Model source**: Download from HuggingFace `sylvester-francis/rust-mentor-8b-GGUF` on first launch
- **Min SDK**: 26 (Android 8.0)
- **Target**: Pixel 8 Pro (8GB RAM, Tensor G3)

### Architecture

```
app/
├── src/main/
│   ├── java/com/rustmentor/
│   │   ├── MainActivity.kt
│   │   ├── RustMentorApp.kt              # App-level composable + navigation
│   │   ├── ui/
│   │   │   ├── theme/
│   │   │   │   └── Theme.kt              # Material 3 theme (rust-orange accent)
│   │   │   ├── screens/
│   │   │   │   ├── ChatScreen.kt         # Main chat interface
│   │   │   │   ├── ModelSetupScreen.kt   # Model download + status
│   │   │   │   └── SettingsScreen.kt     # Temperature, system prompt, etc.
│   │   │   └── components/
│   │   │       ├── MessageBubble.kt      # Chat bubble with markdown/code rendering
│   │   │       ├── CodeBlock.kt          # Syntax-highlighted Rust code block
│   │   │       ├── InputBar.kt           # Message input with send button
│   │   │       └── StreamingIndicator.kt # Typing indicator during generation
│   │   ├── llm/
│   │   │   ├── LlamaEngine.kt           # llama.cpp JNI wrapper
│   │   │   ├── ModelManager.kt          # Download, load, unload GGUF
│   │   │   └── InferenceConfig.kt       # Temperature, top_p, max_tokens, etc.
│   │   ├── data/
│   │   │   ├── ChatRepository.kt        # Room database for chat history
│   │   │   ├── ChatMessage.kt           # Message entity
│   │   │   ├── Conversation.kt          # Conversation entity
│   │   │   └── AppDatabase.kt           # Room database
│   │   └── viewmodel/
│   │       ├── ChatViewModel.kt         # Chat state + inference orchestration
│   │       └── ModelViewModel.kt        # Model download/load state
│   ├── cpp/
│   │   └── llama-android.cpp            # JNI bridge to llama.cpp
│   └── res/
│       └── ...
├── build.gradle.kts
└── CMakeLists.txt                        # Build llama.cpp native lib
```

### Features (Priority Order)

#### P0 — Must Have
1. **Model download**: On first launch, show a setup screen that downloads the GGUF from HuggingFace. Show progress bar. Save to app's internal storage.
2. **Chat interface**: Scrollable message list with user/assistant bubbles. Input bar at bottom with send button.
3. **Streaming inference**: Token-by-token output displayed as it generates (not wait-for-complete). Use llama.cpp's callback mechanism.
4. **System prompt**: Hardcode the RustMentor system prompt:
   ```
   You are RustMentor, an expert Rust programming tutor. The student is an experienced Go, Python, and TypeScript developer learning Rust by building CLI tools.

   Your teaching style:
   - Draw parallels to Go/Python/TypeScript concepts they already know
   - Explain ownership, borrowing, and lifetimes with practical examples
   - When reviewing code, explain what the borrow checker is doing and why
   - Keep explanations concise with code snippets
   - Guide them to write the code themselves rather than giving full solutions
   ```
5. **Code block rendering**: Detect ```rust fenced code blocks in responses and render with:
   - Monospace font
   - Dark background
   - Copy-to-clipboard button
   - Basic Rust syntax highlighting (keywords, strings, comments, types)
6. **Stop generation**: Button to cancel in-progress generation.
7. **Markdown rendering**: Bold, italic, bullet lists, inline code in responses.

#### P1 — Should Have
8. **Conversation history**: Persist conversations with Room. List of past conversations in a drawer/side panel.
9. **New conversation**: Button to start fresh (clears context, keeps model loaded).
10. **Settings screen**:
    - Temperature slider (0.1 - 1.5, default 0.7)
    - Max tokens slider (128 - 2048, default 512)
    - Context length display (2048 tokens)
    - Model info (file size, quantization type)
    - Clear all conversations
    - Delete model (re-download)
11. **Quick prompts**: Floating chips above input bar for common questions:
    - "Explain ownership vs Go pointers"
    - "Review my Rust code"
    - "How do I handle errors?"
    - "What's the difference between &str and String?"
    - "Help me with lifetimes"

#### P2 — Nice to Have
12. **Share code**: Share a code block from the chat to another app.
13. **Dark/light theme**: Follow system theme with Material You dynamic colors.
14. **Export conversation**: Export chat as markdown file.
15. **Paste code for review**: A dedicated "paste code" button that wraps input in ```rust blocks.
16. **Offline indicator**: Show a badge confirming "Offline Mode — No data leaves your device."

### LLM Engine Details

#### llama.cpp Integration

Use llama.cpp compiled for Android via CMake/NDK. The JNI bridge (`LlamaEngine.kt` + `llama-android.cpp`) should expose:

```kotlin
class LlamaEngine {
    // Load model from file path
    fun loadModel(modelPath: String, contextSize: Int = 2048): Boolean

    // Run inference with streaming callback
    fun generate(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        onToken: (String) -> Unit,    // called per token
        onComplete: () -> Unit,
        onError: (String) -> Unit
    )

    // Stop current generation
    fun stopGeneration()

    // Release model from memory
    fun unloadModel()

    // Check if model is loaded
    fun isModelLoaded(): Boolean
}
```

#### Chat Template (Qwen3 format)

The model uses Qwen3 chat template. Format messages as:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

The engine should format the full conversation history (system + all user/assistant turns) into this template before each inference call.

#### Memory Management

- Load model on app start (takes ~5-10 seconds on Pixel 8 Pro)
- Keep model in memory while app is in foreground
- On `onTrimMemory(TRIM_MEMORY_RUNNING_LOW)`, show warning but keep model
- On app backgrounded for >5 min, consider unloading to free RAM
- Use a foreground service notification while model is loaded: "RustMentor is ready"

### Model Download

```kotlin
// Download from HuggingFace using their CDN
val modelUrl = "https://huggingface.co/sylvester-francis/rust-mentor-8b-GGUF/resolve/main/<filename>.gguf"

// Use WorkManager for reliable download
// Show notification with progress
// Resume partial downloads
// Verify file size after download
// Store in: context.filesDir / "models" / "rust-mentor-8b.gguf"
```

### UI/UX Design

- **Color scheme**: Rust-orange (#CE412B) as primary accent on Material 3 dark theme
- **App icon**: Ferris the crab (Rust mascot) with a graduation cap, or a simple crab emoji silhouette
- **Typography**: Use monospace (JetBrains Mono or Fira Code) for code, default sans-serif for text
- **Splash screen**: Show Ferris + "Loading model..." with a progress indicator
- **Empty state**: When no messages, show suggested prompts:
  - "I'm new to Rust. Where do I start?"
  - "Explain ownership like I'm a Go developer"
  - "Help me understand the borrow checker"
- **Input bar**: TextField with hint "Ask about Rust..." and a send icon button
- **Message bubbles**: User messages right-aligned (primary color), assistant messages left-aligned (surface variant)
- **Code blocks**: Dark background (#1E1E1E), rounded corners, "Copy" button top-right, language label top-left ("rust")

### Dependencies

```kotlin
// build.gradle.kts
dependencies {
    // Compose
    implementation("androidx.compose.material3:material3:1.2.0")
    implementation("androidx.navigation:navigation-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")

    // Room (chat history)
    implementation("androidx.room:room-runtime:2.6.0")
    implementation("androidx.room:room-ktx:2.6.0")
    ksp("androidx.room:room-compiler:2.6.0")

    // Markdown rendering
    implementation("com.halilibo.compose-richtext:richtext-commonmark:0.17.0")

    // Download
    implementation("androidx.work:work-runtime-ktx:2.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
```

### Build Configuration

```kotlin
// build.gradle.kts
android {
    ndkVersion = "26.1.10909125"

    defaultConfig {
        ndk {
            abiFilters += listOf("arm64-v8a")  // Pixel 8 Pro only needs arm64
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
        }
    }
}
```

### Key Constraints

1. **Fully offline**: Zero network calls after model download. No analytics, no telemetry, no crash reporting.
2. **Memory**: Pixel 8 Pro has 12GB RAM. The Q4_K_M model uses ~4.5GB. Keep total app memory under 6GB.
3. **Storage**: Model is ~4.5GB. Warn user about storage requirements before download.
4. **Battery**: LLM inference is CPU/GPU intensive. Show inference time per response so user is aware.
5. **Threading**: All inference on a background thread/coroutine. Never block the UI thread. Use `Dispatchers.Default` for inference.
6. **Context window**: 2048 tokens. When conversation exceeds context, implement sliding window — keep system prompt + last N messages that fit.

### Deliverables

1. Complete Android Studio project with all source files
2. CMakeLists.txt that fetches and builds llama.cpp
3. Working chat UI with streaming responses
4. Model download flow
5. Chat history persistence
6. Settings screen
7. README with build instructions

---

## Quick Start (for the AI building this)

1. Start with a minimal working prototype: hardcoded model path + chat UI + llama.cpp inference
2. Add streaming token display
3. Add model download flow
4. Add chat history (Room)
5. Add settings
6. Polish UI (code blocks, markdown, quick prompts)
