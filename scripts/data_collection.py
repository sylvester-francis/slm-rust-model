"""
Synthetic Rust Tutor Dataset Generator

Generates high-quality Rust tutoring conversations covering:
- Ownership, borrowing, lifetimes
- Error handling (Result, Option, ?)
- Pattern matching & enums
- Traits & generics
- Structs, impl blocks
- CLI tooling patterns
- Iterators & closures
- Cargo, crates, modules
- String types (&str vs String)
- Concurrency basics
- Async/await & Tokio
- Smart pointers (Box, Rc, Arc, RefCell)
- Enums as algebraic data types
- Option/Result combinators
- Type conversions (From, Into, AsRef, Deref)
- Macros (macro_rules!, derive)
- Collections (HashMap, HashSet, BTreeMap, VecDeque)
- Serde & serialization
- File I/O & BufReader
- Unsafe Rust
- Closures in depth (Fn, FnMut, FnOnce)
- Derive macros (Debug, Clone, PartialEq)
- Cargo workspaces
- Modules & visibility
- Documentation & rustdoc
- Type aliases & newtype pattern
"""

import json
import os
from typing import List, Dict

RUST_TUTOR_CONVERSATIONS: List[Dict] = [
    # ── OWNERSHIP & BORROWING ──
    {
        "category": "ownership",
        "conversations": [
            {"role": "user", "content": "In Go, I just pass values or pointers. What's this ownership thing in Rust?"},
            {"role": "assistant", "content": """Great question — and your Go intuition is actually a good starting point. In Go, when you assign `b := a` for a struct, you get a copy. When you pass `&a`, you share a pointer. Rust takes this further with three rules:

1. **Each value has exactly one owner**
2. **When the owner goes out of scope, the value is dropped** (like Go's GC, but deterministic)
3. **Ownership can be transferred (moved) or borrowed**

```rust
let password = String::from("hunter2");
let alias = password;  // ownership MOVED to alias
// println!("{}", password);  // ❌ compile error! password is gone
println!("{}", alias);      // ✅ alias owns it now
```

In Go, `alias := password` would copy the string. In Rust, `String` is heap-allocated, so it *moves* instead of copying. Think of it like transferring a file rather than copying it.

For your password auditor, when you read a password from a file into a `String`, that string has one owner. If you pass it to a checking function, you'll need to decide: move it (give it away) or borrow it (lend it temporarily with `&`).

Try this: what do you think happens if you write a function that takes a `String` parameter and then try to use the string after calling it?"""}
        ]
    },
    {
        "category": "ownership",
        "conversations": [
            {"role": "user", "content": "What's the difference between & and &mut? In Go I just use pointers."},
            {"role": "assistant", "content": """In Go, a `*T` pointer lets you both read and write — there's no compile-time distinction. Rust splits this into two kinds of borrows:

```rust
fn check_length(pw: &str) -> bool {      // shared borrow: read-only
    pw.len() >= 8
}

fn mask_password(pw: &mut String) {       // mutable borrow: read + write
    pw.replace_range(.., "********");
}
```

The key constraint: you can have **many `&T`** OR **one `&mut T`** at a time, never both. This prevents data races at compile time — something Go catches at runtime with the race detector.

For your CLI tool, most functions will take `&str` (borrowed string slice) since they only need to *read* the password to analyze it. You'd use `&mut` only when you need to modify something in place.

Quick mental model from Go:
- `&T` in Rust ≈ passing a value in Go (read-only by convention)
- `&mut T` in Rust ≈ passing `*T` pointer in Go
- But Rust *enforces* the read-only guarantee at compile time"""}
        ]
    },
    {
        "category": "ownership",
        "conversations": [
            {"role": "user", "content": "When should I clone vs borrow? I keep adding .clone() everywhere to make the compiler happy."},
            {"role": "assistant", "content": """The `.clone()` everywhere pattern is a classic beginner phase — it works, but it's like using `copy.deepcopy()` in Python for everything. Let's fix that.

**Rule of thumb**: Borrow first, clone only when you must.

```rust
// ❌ Unnecessary clone
fn check_password(pw: String) -> bool {
    pw.len() >= 8
}
let password = String::from("hunter2");
check_password(password.clone()); // clone just to keep using password
println!("{}", password);

// ✅ Just borrow it
fn check_password(pw: &str) -> bool {
    pw.len() >= 8
}
let password = String::from("hunter2");
check_password(&password); // borrow — no allocation
println!("{}", password);  // still valid
```

**When cloning IS appropriate**:
1. You need two independent owners (e.g., passing data to a thread)
2. The data is small and `Copy` isn't implemented
3. You're building a collection from transformed data

```rust
// Legitimate clone: each thread needs its own copy
let passwords: Vec<String> = load_passwords();
let handle = std::thread::spawn(move || {
    // this thread owns its copy
    analyze(passwords)
});
```

For your passaudit, the pattern will be: read passwords into a `Vec<String>`, then pass `&[String]` or iterate with `&str` references for all your analysis functions. Zero clones needed for the common path."""}
        ]
    },
    # ── ERROR HANDLING ──
    {
        "category": "error_handling",
        "conversations": [
            {"role": "user", "content": "I'm used to Go's if err != nil pattern. How does Rust handle errors?"},
            {"role": "assistant", "content": """You'll love this — Rust's error handling is like Go's `(value, error)` returns, but the compiler forces you to handle every case.

```rust
// Go style:
// data, err := os.ReadFile("passwords.txt")
// if err != nil { return err }

// Rust equivalent:
let data = std::fs::read_to_string("passwords.txt")?;
```

The `?` operator is Rust's answer to `if err != nil { return err }`. It unwraps `Ok(value)` or early-returns `Err(e)`.

Rust uses two main types:
- `Result<T, E>` — like Go's `(T, error)` return pattern
- `Option<T>` — like a nullable value, either `Some(value)` or `None`

```rust
use std::fs;
use std::io;

fn load_passwords(path: &str) -> Result<Vec<String>, io::Error> {
    let content = fs::read_to_string(path)?;  // ? propagates errors
    let passwords: Vec<String> = content
        .lines()
        .map(|line| line.to_string())
        .collect();
    Ok(passwords)  // Wrap success in Ok()
}
```

The big difference from Go: you **cannot** ignore a `Result`. The compiler won't let you. No more forgotten error checks.

For your passaudit, your `main()` can return `Result<(), Box<dyn Error>>` so you can use `?` throughout. Want to try writing a function that reads a password file and returns the passwords as a `Vec<String>`?"""}
        ]
    },
    {
        "category": "error_handling",
        "conversations": [
            {"role": "user", "content": "What's the difference between unwrap(), expect(), and the ? operator?"},
            {"role": "assistant", "content": """These are three ways to handle `Result` and `Option`, from most dangerous to most idiomatic:

```rust
// 1. unwrap() — PANICS on error (like Go's log.Fatal)
let content = fs::read_to_string("passwords.txt").unwrap();
// If file doesn't exist → program crashes with ugly error

// 2. expect() — PANICS with your message (better for debugging)
let content = fs::read_to_string("passwords.txt")
    .expect("Failed to read password file");
// If file doesn't exist → crashes with YOUR message

// 3. ? operator — PROPAGATES error to caller (idiomatic!)
let content = fs::read_to_string("passwords.txt")?;
// If file doesn't exist → returns Err to caller gracefully
```

**When to use each**:
- `unwrap()` / `expect()`: Only in tests, examples, or places where failure is truly impossible
- `?` operator: Production code, library code, anywhere errors should bubble up
- `match`: When you need custom error recovery

```rust
// Pattern: custom recovery with match
fn load_or_default(path: &str) -> Vec<String> {
    match fs::read_to_string(path) {
        Ok(content) => content.lines().map(String::from).collect(),
        Err(e) => {
            eprintln!("Warning: {e}, using empty list");
            Vec::new()
        }
    }
}
```

In Python terms: `unwrap()` is like no try/except, `?` is like letting exceptions propagate, and `match` is like try/except with handling.

For passaudit, use `?` in your core logic and `expect()` only in `main()` for truly critical setup failures."""}
        ]
    },
    {
        "category": "error_handling",
        "conversations": [
            {"role": "user", "content": "How do I create custom error types? In Go I use fmt.Errorf or custom error structs."},
            {"role": "assistant", "content": """Rust gives you two main approaches, similar to Go's `errors.New()` vs custom error types.

**Quick approach with `thiserror` (recommended for libraries)**:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum AuditError {
    #[error("Failed to read password file: {0}")]
    FileRead(#[from] std::io::Error),

    #[error("Password too short: {length} chars (minimum: {min})")]
    TooShort { length: usize, min: usize },

    #[error("Password found in breach database: {0}")]
    Breached(String),
}

fn audit_password(pw: &str) -> Result<(), AuditError> {
    if pw.len() < 8 {
        return Err(AuditError::TooShort {
            length: pw.len(),
            min: 8,
        });
    }
    Ok(())
}
```

**Quick approach with `anyhow` (recommended for applications/CLIs)**:
```rust
use anyhow::{Context, Result};

fn load_and_audit(path: &str) -> Result<()> {
    let content = std::fs::read_to_string(path)
        .context("Failed to read password file")?;  // adds context!

    for (i, line) in content.lines().enumerate() {
        audit_password(line)
            .with_context(|| format!("Password on line {}", i + 1))?;
    }
    Ok(())
}
```

**The Go comparison**:
- `thiserror` ≈ custom error types with `Error()` method
- `anyhow` ≈ `fmt.Errorf("context: %w", err)` wrapping
- `#[from]` ≈ Go's error wrapping with `%w`

For passaudit as a CLI app, start with `anyhow` for simplicity. If you later extract a library crate, switch the library portion to `thiserror`. Add both to your `Cargo.toml`:
```toml
[dependencies]
anyhow = "1"
thiserror = "2"
```"""}
        ]
    },
    # ── PATTERN MATCHING ──
    {
        "category": "pattern_matching",
        "conversations": [
            {"role": "user", "content": "What's match in Rust? Is it like a switch statement?"},
            {"role": "assistant", "content": """`match` is like a switch on steroids. Unlike Go's `switch`, Rust's `match` must be **exhaustive** — you must handle every possible case, or the compiler rejects it.

```rust
// Classifying password strength for your auditor
enum Strength {
    Weak,
    Medium,
    Strong,
    Excellent,
}

fn classify(score: u32) -> Strength {
    match score {
        0..=25  => Strength::Weak,
        26..=50 => Strength::Medium,
        51..=75 => Strength::Strong,
        _       => Strength::Excellent,  // _ is the catch-all
    }
}
```

Where it really shines is with enums and destructuring:

```rust
enum AuditResult {
    Pass,
    Fail { reason: String },
    Warning { message: String, severity: u8 },
}

fn report(result: &AuditResult) {
    match result {
        AuditResult::Pass => println!("✅ Password OK"),
        AuditResult::Fail { reason } => println!("❌ Failed: {reason}"),
        AuditResult::Warning { message, severity } if *severity > 5 => {
            println!("⚠️  HIGH: {message}");
        }
        AuditResult::Warning { message, .. } => {
            println!("ℹ️  {message}");
        }
    }
}
```

Notice the guard clause (`if *severity > 5`) and the `..` to ignore fields. In Go, you'd need a series of `if-else` or a `switch` with type assertions. Here the compiler guarantees you covered everything."""}
        ]
    },
    # ── TRAITS & GENERICS ──
    {
        "category": "traits",
        "conversations": [
            {"role": "user", "content": "Traits look like Go interfaces. What's the difference?"},
            {"role": "assistant", "content": """Good eye — traits *are* Rust's version of interfaces, but with key differences:

**Go interfaces**: implicit (just implement the methods)
**Rust traits**: explicit (you must write `impl Trait for Type`)

```rust
trait PasswordChecker {
    fn check(&self, password: &str) -> bool;

    // Unlike Go, traits can have default implementations!
    fn name(&self) -> &str {
        "unnamed checker"
    }
}

struct LengthChecker { min_length: usize }

impl PasswordChecker for LengthChecker {
    fn check(&self, password: &str) -> bool {
        password.len() >= self.min_length
    }

    fn name(&self) -> &str { "length checker" }
}
```

The biggest superpower: **trait bounds on generics**.

```rust
fn run_audit<C: PasswordChecker>(checker: &C, password: &str) {
    if checker.check(password) {
        println!("✅ {} passed", checker.name());
    } else {
        println!("❌ {} failed", checker.name());
    }
}
```

This is resolved at compile time (monomorphization) — zero runtime cost, unlike Go's interface dispatch which uses vtables.

For your passaudit, define a `PasswordChecker` trait and implement it for `LengthChecker`, `EntropyChecker`, `CommonPasswordChecker`, etc. Then run them all in a pipeline:

```rust
fn run_all_checks(checkers: &[&dyn PasswordChecker], password: &str) {
    for checker in checkers {
        // dyn = dynamic dispatch (like Go interfaces)
        let passed = checker.check(password);
        println!("{}: {}", checker.name(), if passed { "✅" } else { "❌" });
    }
}
```"""}
        ]
    },
    # ── LIFETIMES ──
    {
        "category": "lifetimes",
        "conversations": [
            {"role": "user", "content": "I keep getting lifetime errors. What are lifetimes and why does Rust need them?"},
            {"role": "assistant", "content": """Lifetimes are Rust's way of tracking how long references are valid — something Go's garbage collector handles automatically.

The problem lifetimes solve:
```rust
fn longest(a: &str, b: &str) -> &str {  // ❌ Won't compile!
    if a.len() > b.len() { a } else { b }
}
```

Rust asks: "The return value is a reference, but to which input? How long is it valid?" You annotate:

```rust
fn longest<'a>(a: &'a str, b: &'a str) -> &'a str {  // ✅
    if a.len() > b.len() { a } else { b }
}
```

`'a` says: "the returned reference lives as long as the *shorter* of the two input lifetimes."

**Mental model for a Go developer**: Imagine if Go's escape analysis were visible in function signatures. That's what lifetimes are.

**When you DON'T need them** (lifetime elision):
```rust
// Rust infers the lifetime here — one input ref, one output ref
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}
```

**Practical rule**: If you're returning a reference, the compiler needs to know what it's borrowing from. If you get stuck, try returning an owned type (`String` instead of `&str`) first, then optimize later.

Most lifetime errors early on are solved by:
1. Returning owned data instead of references
2. Adding a single `'a` lifetime parameter"""}
        ]
    },
    # ── STRUCTS & IMPL ──
    {
        "category": "structs",
        "conversations": [
            {"role": "user", "content": "How do I create a struct with methods? In Go I use receiver functions."},
            {"role": "assistant", "content": """Very similar concept! Go uses receiver functions, Rust uses `impl` blocks.

```go
// Go
type Auditor struct {
    MinLength int
    MaxLength int
}

func (a *Auditor) Check(password string) bool {
    return len(password) >= a.MinLength
}

func NewAuditor(min, max int) *Auditor {
    return &Auditor{MinLength: min, MaxLength: max}
}
```

```rust
// Rust equivalent
struct Auditor {
    min_length: usize,
    max_length: usize,
}

impl Auditor {
    // Associated function (like Go's NewAuditor)
    fn new(min_length: usize, max_length: usize) -> Self {
        Self { min_length, max_length }
    }

    // Method with &self (like Go's pointer receiver)
    fn check(&self, password: &str) -> bool {
        password.len() >= self.min_length
    }

    // Method that mutates (like Go's *Auditor receiver)
    fn set_min_length(&mut self, min: usize) {
        self.min_length = min;
    }
}
```

Key differences:
- `Self` = the type itself (like `Auditor`)
- `&self` = immutable borrow of the instance (read-only)
- `&mut self` = mutable borrow (can modify fields)
- `self` = takes ownership (rare, usually for builders)
- No constructor keyword — `fn new()` is convention, not magic

Usage:
```rust
let mut auditor = Auditor::new(8, 128);
println!("Valid: {}", auditor.check("hunter2"));
auditor.set_min_length(12);
```

For passaudit, your main struct might look like:
```rust
struct PasswordAuditor {
    checkers: Vec<Box<dyn PasswordChecker>>,
    config: AuditConfig,
}
```"""}
        ]
    },
    # ── STRING TYPES ──
    {
        "category": "strings",
        "conversations": [
            {"role": "user", "content": "Why does Rust have both String and &str? In Go there's just string."},
            {"role": "assistant", "content": """This is one of the most common Rust confusions! Think of it like Go's `string` vs `[]byte`, but more principled:

- **`String`** = owned, heap-allocated, growable (like Go's `string` but mutable)
- **`&str`** = borrowed string slice, a view into string data (like Go's `string` header)

```rust
let owned: String = String::from("hunter2");  // owns the data
let borrowed: &str = &owned;                    // borrows a view
let literal: &str = "hunter2";                  // string literal (static)
```

**The rule**: Functions that only READ strings should take `&str`. Functions that need to OWN or MODIFY should take `String`.

```rust
// ✅ Good: accepts both String and &str via auto-deref
fn check_strength(password: &str) -> u32 {
    let mut score = 0;
    if password.len() >= 8 { score += 25; }
    if password.chars().any(|c| c.is_uppercase()) { score += 25; }
    if password.chars().any(|c| c.is_numeric()) { score += 25; }
    if password.chars().any(|c| !c.is_alphanumeric()) { score += 25; }
    score
}

// Works with both:
let owned = String::from("P@ssw0rd!");
check_strength(&owned);       // &String auto-derefs to &str
check_strength("literal");     // &str directly
```

**Converting between them**:
```rust
let s: &str = "hello";
let owned: String = s.to_string();  // or String::from(s) or s.to_owned()
let back: &str = &owned;            // borrow
```

For passaudit, your functions will mostly take `&str` parameters — this is the most flexible since callers can pass either `String` references or string literals."""}
        ]
    },
    # ── ITERATORS & CLOSURES ──
    {
        "category": "iterators",
        "conversations": [
            {"role": "user", "content": "How do iterators and closures work in Rust? I love Python's list comprehensions and Go's for-range."},
            {"role": "assistant", "content": """Rust iterators are like Python generators meets Go's for-range, but with zero-cost abstractions — they compile to the same machine code as hand-written loops.

```rust
let passwords = vec!["hunter2", "P@ssw0rd!", "123456", "correcthorsebatterystaple"];

// Python: [pw for pw in passwords if len(pw) >= 8]
// Go:     for _, pw := range passwords { if len(pw) >= 8 { ... } }

// Rust: chain iterator methods
let strong: Vec<&str> = passwords.iter()
    .filter(|pw| pw.len() >= 8)
    .copied()  // &&str → &str
    .collect();

// With transformation (like Python's map)
let lengths: Vec<usize> = passwords.iter()
    .map(|pw| pw.len())
    .collect();

// Chaining multiple operations
let report: Vec<String> = passwords.iter()
    .enumerate()
    .filter(|(_, pw)| pw.len() < 8)
    .map(|(i, pw)| format!("Line {}: '{}' too short ({} chars)", i+1, pw, pw.len()))
    .collect();
```

**Closures** (anonymous functions):
```rust
// Go:   func(pw string) bool { return len(pw) >= 8 }
// Python: lambda pw: len(pw) >= 8
// Rust:
let is_strong = |pw: &str| -> bool { pw.len() >= 8 };

// Type inference usually makes it shorter:
let is_strong = |pw: &str| pw.len() >= 8;
```

**Practical passaudit pattern** — scoring all passwords:
```rust
fn audit_all(passwords: &[String]) -> Vec<(usize, &str, u32)> {
    passwords.iter()
        .enumerate()
        .map(|(i, pw)| (i + 1, pw.as_str(), check_strength(pw)))
        .filter(|(_, _, score)| *score < 50)
        .collect()
}
```

The key insight: iterator chains are lazy — nothing executes until `.collect()` or another terminal operation. This means Rust can optimize the entire chain as one pass."""}
        ]
    },
    # ── CARGO & MODULES ──
    {
        "category": "cargo",
        "conversations": [
            {"role": "user", "content": "How do I structure a Rust project? In Go I use go mod and packages."},
            {"role": "assistant", "content": """Cargo is Rust's build tool + package manager, similar to `go mod` + `go build` combined.

**Create a new project**:
```bash
cargo new passaudit       # binary project
cargo new passaudit --lib # library project
```

**Project structure** (maps to your Go intuition):
```
passaudit/
├── Cargo.toml           # like go.mod + go.sum
├── src/
│   ├── main.rs          # entry point (like main.go)
│   ├── lib.rs           # library root (optional, like package exports)
│   ├── checkers/        # module directory (like a Go package)
│   │   ├── mod.rs       # module declaration (like package declaration)
│   │   ├── length.rs    # sub-module
│   │   └── entropy.rs   # sub-module
│   └── config.rs        # top-level module
└── tests/
    └── integration.rs   # integration tests
```

**Module declaration** (`src/checkers/mod.rs`):
```rust
// Declare sub-modules (Go just uses directories)
mod length;
mod entropy;

// Re-export for convenience
pub use length::LengthChecker;
pub use entropy::EntropyChecker;
```

**Using modules** (`src/main.rs`):
```rust
mod checkers;  // loads src/checkers/mod.rs
mod config;    // loads src/config.rs

use checkers::{LengthChecker, EntropyChecker};

fn main() -> anyhow::Result<()> {
    let checker = LengthChecker::new(8);
    // ...
    Ok(())
}
```

**Cargo.toml** (your go.mod equivalent):
```toml
[package]
name = "passaudit"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }  # CLI args
serde = { version = "1", features = ["derive"] }  # serialization
```

**Key commands**:
```bash
cargo build              # go build
cargo run                # go run .
cargo test               # go test ./...
cargo add clap           # go get (with version resolution)
cargo clippy             # go vet (but much more thorough)
cargo fmt                # gofmt
```"""}
        ]
    },
    # ── CLI WITH CLAP ──
    {
        "category": "cli",
        "conversations": [
            {"role": "user", "content": "How do I parse CLI arguments in Rust? I want to build passaudit as a proper CLI tool."},
            {"role": "assistant", "content": """Use `clap` with the derive macro — it's the standard choice, similar to Go's `cobra` or Python's `click`:

```bash
cargo add clap --features derive
```

```rust
use clap::{Parser, Subcommand};

/// passaudit - A password security auditor
#[derive(Parser)]
#[command(name = "passaudit", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Audit passwords from a file
    Audit {
        /// Path to the password file
        #[arg(short, long)]
        file: String,

        /// Minimum password length
        #[arg(short, long, default_value_t = 8)]
        min_length: usize,

        /// Output format
        #[arg(short, long, default_value = "text")]
        output: String,
    },

    /// Check a single password
    Check {
        /// The password to check (will prompt if not provided)
        password: Option<String>,
    },

    /// Generate a secure password
    Generate {
        /// Password length
        #[arg(short, long, default_value_t = 16)]
        length: usize,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Audit { file, min_length, output } => {
            if cli.verbose {
                println!("Auditing {} with min length {}", file, min_length);
            }
            // audit logic here
        }
        Commands::Check { password } => {
            let pw = password.unwrap_or_else(|| {
                // prompt for password
                String::new()
            });
            // check logic here
        }
        Commands::Generate { length } => {
            // generate logic here
        }
    }

    Ok(())
}
```

Usage becomes:
```bash
passaudit audit --file passwords.txt --min-length 12
passaudit check "hunter2"
passaudit generate --length 20
passaudit --verbose audit --file pw.txt
```

The derive macro generates all the help text, argument parsing, and validation at compile time. If you've used Go's `cobra`, the structure is very similar but more type-safe — invalid argument combinations are caught at compile time."""}
        ]
    },
    # ── CODE REVIEW EXAMPLES ──
    {
        "category": "code_review",
        "conversations": [
            {"role": "user", "content": "Can you review this code?\n\n```rust\nfn check_passwords(passwords: Vec<String>) -> Vec<String> {\n    let mut weak = Vec::new();\n    for i in 0..passwords.len() {\n        if passwords[i].len() < 8 {\n            weak.push(passwords[i].clone());\n        }\n    }\n    return weak;\n}\n```"},
            {"role": "assistant", "content": """Good start! It works, but there are several Rust-idiomatic improvements. Let me walk through them:

**1. Take a reference instead of owning the Vec**
```rust
// ❌ Takes ownership — caller can't use passwords after this
fn check_passwords(passwords: Vec<String>) -> Vec<String>

// ✅ Borrows — caller keeps ownership
fn check_passwords(passwords: &[String]) -> Vec<&str>
```

**2. Use iterators instead of index-based loops**
```rust
// ❌ C-style indexing (works but not idiomatic)
for i in 0..passwords.len() { passwords[i]... }

// ✅ Iterator (idiomatic, bounds-checked, optimizable)
for pw in passwords { ... }
```

**3. Avoid unnecessary clones by returning references**

**4. Drop explicit `return` (Rust convention)**

Here's the idiomatic version:
```rust
fn check_passwords(passwords: &[String]) -> Vec<&str> {
    passwords.iter()
        .filter(|pw| pw.len() < 8)
        .map(|pw| pw.as_str())
        .collect()
}
```

Or if you want to keep it as a loop (totally fine):
```rust
fn check_passwords(passwords: &[String]) -> Vec<&str> {
    let mut weak = Vec::new();
    for pw in passwords {
        if pw.len() < 8 {
            weak.push(pw.as_str());  // borrow instead of clone
        }
    }
    weak  // implicit return (no semicolon)
}
```

**What the borrow checker is doing**: By returning `Vec<&str>`, the compiler knows the returned references point into the original `passwords` slice. The caller must keep `passwords` alive as long as they use the result. This is enforced at compile time — zero runtime cost, zero possibility of dangling references."""}
        ]
    },
    {
        "category": "code_review",
        "conversations": [
            {"role": "user", "content": "```rust\nuse std::collections::HashMap;\n\nfn count_char_types(password: &str) -> HashMap<&str, i32> {\n    let mut counts = HashMap::new();\n    counts.insert(\"uppercase\", 0);\n    counts.insert(\"lowercase\", 0);\n    counts.insert(\"digits\", 0);\n    counts.insert(\"special\", 0);\n    \n    for c in password.chars() {\n        if c.is_uppercase() {\n            *counts.get_mut(\"uppercase\").unwrap() += 1;\n        } else if c.is_lowercase() {\n            *counts.get_mut(\"lowercase\").unwrap() += 1;\n        } else if c.is_numeric() {\n            *counts.get_mut(\"digits\").unwrap() += 1;\n        } else {\n            *counts.get_mut(\"special\").unwrap() += 1;\n        }\n    }\n    counts\n}\n```\nIs this idiomatic?"},
            {"role": "assistant", "content": """It works, but there's a cleaner pattern. A few observations:

**1. A struct is better than a HashMap for known keys**
You have a fixed set of categories — this is a perfect struct use case:

```rust
#[derive(Debug, Default)]
struct CharCounts {
    uppercase: u32,
    lowercase: u32,
    digits: u32,
    special: u32,
}

impl CharCounts {
    fn total(&self) -> u32 {
        self.uppercase + self.lowercase + self.digits + self.special
    }

    fn has_variety(&self, min_categories: u32) -> bool {
        let cats = [self.uppercase, self.lowercase, self.digits, self.special];
        cats.iter().filter(|&&c| c > 0).count() as u32 >= min_categories
    }
}
```

**2. Use `match` or `if-else` chains with the struct**
```rust
fn count_char_types(password: &str) -> CharCounts {
    let mut counts = CharCounts::default();

    for c in password.chars() {
        match c {
            _ if c.is_uppercase() => counts.uppercase += 1,
            _ if c.is_lowercase() => counts.lowercase += 1,
            _ if c.is_numeric()   => counts.digits += 1,
            _                     => counts.special += 1,
        }
    }
    counts
}
```

**3. Or go full iterator with `fold`**
```rust
fn count_char_types(password: &str) -> CharCounts {
    password.chars().fold(CharCounts::default(), |mut acc, c| {
        if c.is_uppercase()      { acc.uppercase += 1 }
        else if c.is_lowercase() { acc.lowercase += 1 }
        else if c.is_numeric()   { acc.digits += 1 }
        else                     { acc.special += 1 }
        acc
    })
}
```

**Why struct > HashMap here**:
- Compile-time field access (no runtime key lookups)
- No `.unwrap()` calls (can't have missing keys)
- You can add methods like `has_variety()` and `total()`
- Better performance (stack-allocated, no hashing)

The HashMap approach is valid when your categories are dynamic (user-defined), but for a fixed schema, structs are the Rust way."""}
        ]
    },
    # ── CONCURRENCY ──
    {
        "category": "concurrency",
        "conversations": [
            {"role": "user", "content": "How does Rust handle concurrency? I love Go's goroutines and channels."},
            {"role": "assistant", "content": """Rust's concurrency model is philosophically different from Go's. Go says "share memory by communicating" (channels). Rust says "if it compiles, there are no data races."

**Threads (like goroutines but OS-level)**:
```rust
use std::thread;

let passwords = vec!["pw1".to_string(), "pw2".to_string()];
let handle = thread::spawn(move || {
    // `move` transfers ownership into the thread
    for pw in &passwords {
        println!("Checking: {}", pw);
    }
    passwords.len()
});

let count = handle.join().unwrap(); // wait for thread
```

**Channels (like Go channels)**:
```rust
use std::sync::mpsc; // multi-producer, single-consumer

let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    tx.send("weak password found!").unwrap();
});

let msg = rx.recv().unwrap();
println!("Got: {}", msg);
```

**Rayon for easy parallelism** (like a parallel for-range):
```rust
// cargo add rayon
use rayon::prelude::*;

// Sequential
let results: Vec<u32> = passwords.iter()
    .map(|pw| check_strength(pw))
    .collect();

// Parallel — just change .iter() to .par_iter()!
let results: Vec<u32> = passwords.par_iter()
    .map(|pw| check_strength(pw))
    .collect();
```

**The killer feature**: The compiler catches data races. This won't compile:
```rust
let mut data = vec![1, 2, 3];
thread::spawn(|| {
    data.push(4);  // ❌ can't borrow `data` as mutable in thread
});
data.push(5);      // because it's also used here
```

For passaudit, Rayon is your best bet — change `.iter()` to `.par_iter()` and your password analysis parallelizes automatically across all CPU cores. No mutexes, no race conditions, no debugging concurrency bugs at 2am."""}
        ]
    },
    # ── TESTING ──
    {
        "category": "testing",
        "conversations": [
            {"role": "user", "content": "How do I write tests in Rust? In Go I use _test.go files."},
            {"role": "assistant", "content": """Rust tests live alongside your code (unlike Go's separate files), which is actually really convenient:

**Unit tests (in the same file)**:
```rust
// src/checkers/length.rs

pub fn check_length(password: &str, min: usize) -> bool {
    password.len() >= min
}

#[cfg(test)]  // only compiled when running tests
mod tests {
    use super::*;  // import everything from parent

    #[test]
    fn test_strong_password() {
        assert!(check_length("P@ssw0rd!", 8));
    }

    #[test]
    fn test_weak_password() {
        assert!(!check_length("short", 8));
    }

    #[test]
    fn test_exact_length() {
        assert!(check_length("12345678", 8));
    }

    #[test]
    #[should_panic(expected = "empty password")]
    fn test_empty_panics() {
        validate_not_empty(""); // should panic
    }
}
```

**Integration tests** (`tests/` directory, like Go's `_test.go`):
```rust
// tests/integration.rs
use passaudit::Auditor;

#[test]
fn test_full_audit_pipeline() {
    let auditor = Auditor::new(8, 128);
    let results = auditor.audit_file("tests/fixtures/sample_passwords.txt")
        .expect("should read test file");

    assert!(results.len() > 0);
    assert!(results.iter().any(|r| r.strength == Strength::Weak));
}
```

**Run tests**:
```bash
cargo test                    # all tests (like go test ./...)
cargo test check_length       # tests matching name
cargo test -- --nocapture     # show println! output
cargo test --lib              # only unit tests
cargo test --test integration # only integration tests
```

**Key differences from Go**:
- `assert!()` instead of `if got != want { t.Error() }`
- `#[cfg(test)]` means test code is never in your release binary
- No separate files needed for unit tests
- `cargo test` runs ALL tests (unit + integration + doc tests)

Start with the `#[cfg(test)] mod tests` pattern in each file — it's the Rust convention and keeps tests close to the code they verify."""}
        ]
    },
    # ── ASYNC/AWAIT & TOKIO ──
    {
        "category": "async_await",
        "conversations": [
            {"role": "user", "content": "How does async work in Rust? In Go I just use goroutines and it's simple."},
            {"role": "assistant", "content": """Good question — async in Rust is more explicit than Go's goroutines, but gives you more control.

In Go, `go func()` spawns a goroutine on Go's built-in runtime. Rust doesn't have a built-in async runtime — you choose one. **Tokio** is the standard choice (like Go's runtime, but opt-in).

```bash
cargo add tokio --features full
```

```rust
use tokio;

// Go: func main() { go doWork() }
// Rust:
#[tokio::main]
async fn main() {
    let result = fetch_data("https://api.example.com").await;
    println!("{}", result);
}

async fn fetch_data(url: &str) -> String {
    // .await is like Go's blocking call, but non-blocking under the hood
    let response = reqwest::get(url).await.unwrap();
    response.text().await.unwrap()
}
```

**Key differences from Go**:
- `async fn` returns a `Future` — nothing runs until you `.await` it
- `.await` is explicit — you always know when you're yielding
- No hidden goroutine scheduler — you pick Tokio, async-std, etc.
- `async fn` can't be called from sync code without a runtime

```rust
// Spawn concurrent tasks (like goroutines)
let handle1 = tokio::spawn(async {
    fetch_data("https://api1.com").await
});
let handle2 = tokio::spawn(async {
    fetch_data("https://api2.com").await
});

// Wait for both (like sync.WaitGroup)
let (r1, r2) = tokio::join!(handle1, handle2);
```

**Mental model**: `async fn` in Rust ≈ a function that returns a state machine. `.await` ≈ "run this until it's ready, yield if waiting." `tokio::spawn` ≈ `go func()`.

For a CLI tool like passaudit, you probably don't need async — it's mainly for network-heavy or I/O-heavy services. But if you add a feature to check passwords against an online breach API, async would be the way to go."""}
        ]
    },
    {
        "category": "async_await",
        "conversations": [
            {"role": "user", "content": "When should I use async vs threads in Rust? In Go everything is goroutines."},
            {"role": "assistant", "content": """Great question — this trips up a lot of Go developers because Go blurs the line. In Rust, the choice is explicit:

**Use threads (`std::thread`)** when:
- CPU-bound work (crunching numbers, hashing passwords)
- Simple parallelism (Rayon is even easier)
- No need for high concurrency

**Use async (Tokio)** when:
- I/O-bound work (HTTP requests, database queries, file I/O)
- High concurrency (thousands of connections)
- Building a server or network client

```rust
// CPU-bound: use threads or Rayon
use rayon::prelude::*;
let hashes: Vec<String> = passwords.par_iter()
    .map(|pw| expensive_hash(pw))
    .collect();

// I/O-bound: use async
async fn check_breaches(passwords: &[String]) -> Vec<bool> {
    let mut tasks = Vec::new();
    for pw in passwords {
        let pw = pw.clone();
        tasks.push(tokio::spawn(async move {
            check_hibp_api(&pw).await
        }));
    }

    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap());
    }
    results
}
```

**The Go comparison**:
- Go goroutines ≈ Tokio tasks (lightweight, multiplexed on OS threads)
- Go's `runtime.GOMAXPROCS` ≈ Tokio's thread pool size
- Go channels ≈ `tokio::sync::mpsc`
- Go `select {}` ≈ `tokio::select!`

```rust
// tokio::select! — like Go's select statement
tokio::select! {
    result = fetch_from_api() => println!("API: {result:?}"),
    _ = tokio::time::sleep(Duration::from_secs(5)) => println!("Timeout!"),
}
```

**Rule of thumb**: If your Go code uses goroutines mainly for network I/O, use async in Rust. If it's for CPU parallelism, use Rayon. Most CLI tools need neither — synchronous code is fine."""}
        ]
    },
    {
        "category": "async_await",
        "conversations": [
            {"role": "user", "content": "How do I make an HTTP request in Rust? In Python I use requests, in Go I use net/http."},
            {"role": "assistant", "content": """The go-to crate is `reqwest` — it's Rust's equivalent of Python's `requests` library:

```bash
cargo add reqwest --features json
cargo add tokio --features full
cargo add serde --features derive
cargo add serde_json
```

**Simple GET request**:
```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Python: requests.get(url).text
    // Go:     resp, err := http.Get(url)
    let body = reqwest::get("https://httpbin.org/get")
        .await?
        .text()
        .await?;

    println!("{body}");
    Ok(())
}
```

**POST with JSON** (like Python's `requests.post(url, json=data)`):
```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct PasswordCheck {
    password_hash: String,
}

#[derive(Deserialize, Debug)]
struct BreachResult {
    found: bool,
    count: u64,
}

async fn check_breach(hash: &str) -> anyhow::Result<BreachResult> {
    let client = reqwest::Client::new();
    let result: BreachResult = client
        .post("https://api.example.com/check")
        .json(&PasswordCheck { password_hash: hash.to_string() })
        .send()
        .await?
        .json()  // auto-deserialize with serde
        .await?;

    Ok(result)
}
```

**Key pattern**: Notice the double `.await` — first for the network request, second for reading the body. This is because HTTP is a streaming protocol.

**Without async** (blocking, simpler for CLI tools):
```rust
// cargo add reqwest --features blocking
let body = reqwest::blocking::get("https://httpbin.org/get")?.text()?;
```

For passaudit, if you add breach checking, use `reqwest::blocking` to keep things simple in a CLI context."""}
        ]
    },
    # ── SMART POINTERS ──
    {
        "category": "smart_pointers",
        "conversations": [
            {"role": "user", "content": "What is Box<T> and when do I use it? I keep seeing it in Rust code."},
            {"role": "assistant", "content": """`Box<T>` is the simplest smart pointer — it puts data on the heap instead of the stack. In Go, the compiler decides heap vs stack automatically (escape analysis). In Rust, you're explicit about it.

**When you need Box**:

**1. Trait objects (dynamic dispatch)**:
```rust
// You can't have Vec<dyn PasswordChecker> — size unknown at compile time
// Box gives it a known size (pointer width)
let checkers: Vec<Box<dyn PasswordChecker>> = vec![
    Box::new(LengthChecker::new(8)),
    Box::new(EntropyChecker::new(3.0)),
    Box::new(CommonPasswordChecker::new()),
];

for checker in &checkers {
    checker.check("hunter2");
}
```

**2. Recursive types**:
```rust
// ❌ Won't compile — infinite size!
enum AuditTree {
    Leaf(String),
    Node(AuditTree, AuditTree),  // how big is this?
}

// ✅ Box breaks the recursion
enum AuditTree {
    Leaf(String),
    Node(Box<AuditTree>, Box<AuditTree>),  // pointer = fixed size
}
```

**3. Large data you don't want on the stack**:
```rust
// Stack might overflow with huge arrays
let big_data = Box::new([0u8; 1_000_000]);
```

**Mental model from Go**: In Go, `new(T)` returns `*T` (heap pointer). Rust's `Box::new(value)` is similar, but Box *owns* the data and frees it when dropped — no GC needed.

```rust
let boxed = Box::new(42);
println!("{}", *boxed);  // dereference with *
// boxed is freed here automatically (RAII)
```

For passaudit, you'll mainly use Box for `Vec<Box<dyn PasswordChecker>>` — a collection of different checker implementations."""}
        ]
    },
    {
        "category": "smart_pointers",
        "conversations": [
            {"role": "user", "content": "What are Rc and Arc? When do I need shared ownership?"},
            {"role": "assistant", "content": """In Rust, every value has **one owner**. But sometimes multiple parts of your code need to own the same data. That's where `Rc` and `Arc` come in — they add reference counting.

- **`Rc<T>`** = Reference Counted (single-threaded only)
- **`Arc<T>`** = Atomic Reference Counted (thread-safe)

```rust
use std::rc::Rc;

// Scenario: multiple checkers share a common wordlist
let wordlist = Rc::new(load_common_passwords());
// Rc::clone doesn't copy the data — just increments a counter
let checker1_words = Rc::clone(&wordlist);
let checker2_words = Rc::clone(&wordlist);

// Both point to the same data. When all Rc's are dropped, data is freed.
println!("Refs: {}", Rc::strong_count(&wordlist)); // 3
```

**Arc for threads** (Rc is not thread-safe):
```rust
use std::sync::Arc;
use std::thread;

let shared_config = Arc::new(AuditConfig::default());

let handles: Vec<_> = passwords.chunks(100).map(|chunk| {
    let config = Arc::clone(&shared_config);
    let chunk = chunk.to_vec();
    thread::spawn(move || {
        audit_chunk(&chunk, &config)
    })
}).collect();
```

**Go comparison**:
- Go: shared data + GC handles it automatically
- Rust `Rc`: shared data + deterministic cleanup, no GC
- Rust `Arc`: same as Rc but safe across threads (like Go's automatic handling)

**When to use what**:
| Need | Use |
|------|-----|
| Single owner | Just the value (default) |
| Multiple owners, single thread | `Rc<T>` |
| Multiple owners, multi-thread | `Arc<T>` |
| Shared + mutable, single thread | `Rc<RefCell<T>>` |
| Shared + mutable, multi-thread | `Arc<Mutex<T>>` |

**Rule of thumb**: Start without Rc/Arc. Only add them when the compiler tells you multiple things need to own the same data."""}
        ]
    },
    {
        "category": "smart_pointers",
        "conversations": [
            {"role": "user", "content": "What's RefCell? I see Rc<RefCell<T>> in some code and it looks complicated."},
            {"role": "assistant", "content": """RefCell is Rust's escape hatch for the borrowing rules. Normally, Rust checks borrows at **compile time**. RefCell moves those checks to **runtime**.

**The problem it solves**: Sometimes you need to mutate data behind a shared reference, and you can prove it's safe but the compiler can't.

```rust
use std::cell::RefCell;

let data = RefCell::new(vec![1, 2, 3]);

// Multiple immutable borrows — fine
let r1 = data.borrow();
let r2 = data.borrow();
println!("{:?}, {:?}", r1, r2);
drop(r1);
drop(r2);

// Mutable borrow — must be the only borrow
let mut r3 = data.borrow_mut();
r3.push(4);
// data.borrow(); // ❌ would PANIC at runtime (not compile error!)
```

**Rc<RefCell<T>> pattern** — shared + mutable:
```rust
use std::rc::Rc;
use std::cell::RefCell;

struct AuditLog {
    entries: Rc<RefCell<Vec<String>>>,
}

impl AuditLog {
    fn new() -> Self {
        Self { entries: Rc::new(RefCell::new(Vec::new())) }
    }

    fn log(&self, msg: String) {
        // &self (immutable) but we can still mutate via RefCell!
        self.entries.borrow_mut().push(msg);
    }

    fn share(&self) -> Rc<RefCell<Vec<String>>> {
        Rc::clone(&self.entries)
    }
}
```

**Go comparison**: In Go, you'd use a mutex for shared mutable state. RefCell is similar but single-threaded:
- `RefCell<T>` ≈ single-threaded `sync.Mutex` (panics instead of deadlocks)
- `Mutex<T>` ≈ Go's `sync.Mutex` (blocks, thread-safe)
- `RwLock<T>` ≈ Go's `sync.RWMutex`

**Cell vs RefCell**:
- `Cell<T>`: for `Copy` types (integers, bools) — zero overhead
- `RefCell<T>`: for any type — has runtime borrow checking cost

**When to use**: Interior mutability is a last resort. Prefer restructuring your code so mutation flows naturally through `&mut` references. Use RefCell when you're implementing patterns like observer, caching, or shared logging where the API requires `&self` but you need to mutate."""}
        ]
    },
    # ── ENUMS AS ALGEBRAIC DATA TYPES ──
    {
        "category": "enums",
        "conversations": [
            {"role": "user", "content": "Rust enums seem more powerful than Go's iota constants. What can I do with them?"},
            {"role": "assistant", "content": """Rust enums are one of the language's killer features — they're **algebraic data types**, not just named constants. Each variant can hold different data.

**Go iota vs Rust enum**:
```go
// Go: just named integers
type Strength int
const (
    Weak Strength = iota
    Medium
    Strong
)
// Can't attach data to variants. Need separate structs.
```

```rust
// Rust: each variant is its own type with its own data
enum AuditResult {
    Pass,                                    // no data (like Go's iota)
    Fail(String),                           // holds a String
    Warning { message: String, code: u32 }, // named fields (like a struct)
    MultipleIssues(Vec<String>),            // holds a Vec
}
```

**The power**: match forces you to handle every variant:
```rust
fn display_result(result: &AuditResult) {
    match result {
        AuditResult::Pass => println!("✅ Passed"),
        AuditResult::Fail(reason) => println!("❌ {reason}"),
        AuditResult::Warning { message, code } => {
            println!("⚠️  [{code}] {message}")
        }
        AuditResult::MultipleIssues(issues) => {
            for issue in issues {
                println!("  - {issue}");
            }
        }
    }
}
```

**Real-world pattern — modeling state machines**:
```rust
enum ConnectionState {
    Disconnected,
    Connecting { attempt: u32, timeout: Duration },
    Connected { session_id: String },
    Error { message: String, retryable: bool },
}

// Invalid states are unrepresentable!
// Can't have a session_id while disconnected.
```

**Python comparison**: Like Python's tagged unions (or `@dataclass` variants), but enforced by the compiler. TypeScript's discriminated unions (`type Result = { kind: "ok", value: T } | { kind: "err", error: E }`) are the closest analogy.

**Option and Result are just enums**:
```rust
// This is literally how Option is defined:
enum Option<T> {
    Some(T),
    None,
}

// And Result:
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

For passaudit, model your checker results as an enum — it's the idiomatic way to represent "one of several possible outcomes" with associated data."""}
        ]
    },
    {
        "category": "enums",
        "conversations": [
            {"role": "user", "content": "How do I add methods to enums? Can they have behavior like structs?"},
            {"role": "assistant", "content": """Absolutely! Enums get `impl` blocks just like structs — this is something Go can't do with iota constants.

```rust
#[derive(Debug)]
enum PasswordStrength {
    Weak,
    Fair,
    Strong,
    Excellent,
}

impl PasswordStrength {
    // Constructor-like method
    fn from_score(score: u32) -> Self {
        match score {
            0..=25  => Self::Weak,
            26..=50 => Self::Fair,
            51..=75 => Self::Strong,
            _       => Self::Excellent,
        }
    }

    // Methods on the enum
    fn is_acceptable(&self) -> bool {
        matches!(self, Self::Strong | Self::Excellent)
    }

    fn label(&self) -> &str {
        match self {
            Self::Weak      => "WEAK",
            Self::Fair      => "FAIR",
            Self::Strong    => "STRONG",
            Self::Excellent => "EXCELLENT",
        }
    }

    fn color_code(&self) -> &str {
        match self {
            Self::Weak      => "\\x1b[31m",  // red
            Self::Fair      => "\\x1b[33m",  // yellow
            Self::Strong    => "\\x1b[32m",  // green
            Self::Excellent => "\\x1b[36m",  // cyan
        }
    }
}

// Implement Display for easy printing
impl std::fmt::Display for PasswordStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// Usage
let strength = PasswordStrength::from_score(65);
println!("{}", strength);           // "STRONG"
println!("{}", strength.is_acceptable()); // true
```

**Enums with data + methods get really powerful**:
```rust
enum CheckResult {
    Passed { checker: String },
    Failed { checker: String, reason: String, suggestion: Option<String> },
}

impl CheckResult {
    fn is_passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }

    fn summary(&self) -> String {
        match self {
            Self::Passed { checker } => format!("✅ {checker}"),
            Self::Failed { checker, reason, suggestion } => {
                let mut s = format!("❌ {checker}: {reason}");
                if let Some(hint) = suggestion {
                    s.push_str(&format!(" (tip: {hint})"));
                }
                s
            }
        }
    }
}
```

**The `matches!` macro** is handy for simple boolean checks without full pattern matching — it's like a one-liner match that returns bool.

In TypeScript terms, this is like having methods on discriminated union types — something TS can't do natively."""}
        ]
    },
    # ── OPTION COMBINATORS ──
    {
        "category": "option_combinators",
        "conversations": [
            {"role": "user", "content": "I keep writing match statements for Option. Is there a more concise way?"},
            {"role": "assistant", "content": """Yes! Option (and Result) have combinator methods that let you chain operations — like Python's optional chaining or TypeScript's `?.` operator, but more powerful.

**Instead of match/if-let, use combinators**:

```rust
// ❌ Verbose match
let display_name = match user.nickname {
    Some(nick) => nick,
    None => user.email.clone(),
};

// ✅ Combinator
let display_name = user.nickname.unwrap_or_else(|| user.email.clone());
```

**Essential combinators**:
```rust
let port: Option<u16> = Some(8080);

// map: transform the inner value (like .map() on arrays)
let port_str: Option<String> = port.map(|p| format!(":{p}"));
// Some("8080") → Some(":8080"), None → None

// and_then (flatmap): chain operations that return Option
fn parse_port(s: &str) -> Option<u16> {
    s.parse().ok()  // Result → Option
}
let port: Option<u16> = config.get("port").and_then(|s| parse_port(s));

// unwrap_or: provide a default
let port = config_port.unwrap_or(3000);

// unwrap_or_else: compute default lazily
let port = config_port.unwrap_or_else(|| find_free_port());

// unwrap_or_default: use Default trait (0 for numbers, "" for strings)
let port: u16 = config_port.unwrap_or_default();

// filter: keep Some only if predicate matches
let valid_port = port.filter(|&p| p > 1024);

// is_some / is_none: boolean checks
if config_port.is_some() { /* ... */ }
```

**Chaining combinators** (like method chaining in Python/JS):
```rust
fn get_password_hint(pw: &str) -> Option<String> {
    pw.chars()
        .find(|c| c.is_uppercase())   // Option<char>
        .map(|c| c.to_lowercase().to_string())  // Option<String>
        .filter(|s| s.len() == 1)     // Option<String>
        .map(|s| format!("Hint: starts with '{s}'"))
}
```

**The `?` operator works on Option too** (in functions returning Option):
```rust
fn first_strong_password(passwords: &[String]) -> Option<&str> {
    let first = passwords.first()?;  // returns None if empty
    if first.len() >= 8 {
        Some(first.as_str())
    } else {
        None
    }
}
```

**TypeScript comparison**: `option.map(f)` is like `value?.let(f)` if that existed. `and_then` is like optional chaining `a?.b?.c` where each step might fail."""}
        ]
    },
    {
        "category": "option_combinators",
        "conversations": [
            {"role": "user", "content": "What about Result combinators? I want to chain fallible operations cleanly."},
            {"role": "assistant", "content": """Result has the same combinator pattern as Option — and they interop nicely.

**Essential Result combinators**:
```rust
use std::fs;
use std::num::ParseIntError;

// map: transform Ok value
let content_len: Result<usize, _> = fs::read_to_string("passwords.txt")
    .map(|s| s.len());

// map_err: transform the error type
let port: Result<u16, String> = "8080".parse::<u16>()
    .map_err(|e| format!("bad port: {e}"));

// and_then: chain fallible operations (flatmap)
let count: Result<usize, std::io::Error> = fs::read_to_string("passwords.txt")
    .map(|content| content.lines().count());

// unwrap_or / unwrap_or_else: same as Option
let content = fs::read_to_string("pw.txt").unwrap_or_default();

// ok(): Result<T,E> → Option<T> (discards error)
let maybe_content: Option<String> = fs::read_to_string("pw.txt").ok();

// Convert Option → Result with ok_or / ok_or_else
let first: Result<&str, &str> = passwords.first()
    .ok_or("no passwords found")?;
```

**Chaining fallible operations**:
```rust
fn load_config(path: &str) -> anyhow::Result<Config> {
    fs::read_to_string(path)              // Result<String, io::Error>
        .map(|s| s.trim().to_string())     // Result<String, io::Error>
        .and_then(|s| {                    // chain another fallible op
            serde_json::from_str(&s)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
}

// But honestly, ? is usually cleaner for this:
fn load_config(path: &str) -> anyhow::Result<Config> {
    let content = fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(content.trim())?;
    Ok(config)
}
```

**When to use combinators vs `?`**:
- **`?` operator**: When steps are sequential and you want early return
- **Combinators**: When transforming data inline, especially in iterator chains

```rust
// Combinators shine in iterator chains:
let valid_ports: Vec<u16> = config_lines.iter()
    .filter_map(|line| line.strip_prefix("port="))
    .filter_map(|s| s.parse::<u16>().ok())  // .ok() drops parse errors
    .filter(|&p| p > 1024)
    .collect();
```

`filter_map` is especially useful — it maps and filters in one step, keeping only the `Some` values."""}
        ]
    },
    # ── TYPE CONVERSIONS ──
    {
        "category": "type_conversions",
        "conversations": [
            {"role": "user", "content": "I'm confused by From, Into, AsRef, and all the conversion traits. When do I use each?"},
            {"role": "assistant", "content": """These are Rust's standard conversion traits. They look intimidating, but each has a clear purpose. Think of it as a spectrum from cheap to expensive:

**`AsRef<T>` / `AsMut<T>` — cheap reference conversion (borrowing)**:
```rust
// "I can look at this as a &T without any allocation"
fn check_password(pw: impl AsRef<str>) -> bool {
    let pw = pw.as_ref();  // works with String, &str, Cow<str>, etc.
    pw.len() >= 8
}

check_password("literal");           // &str
check_password(String::from("owned")); // String
check_password(&some_cow_str);        // Cow<str>
```

**`From<T>` / `Into<T>` — owned conversion (possibly allocating)**:
```rust
// From: "I can create Self from T"
impl From<&str> for PasswordStrength {
    fn from(s: &str) -> Self {
        match s {
            "weak" => Self::Weak,
            "strong" => Self::Strong,
            _ => Self::Fair,
        }
    }
}

// Into is the reverse — you get it free when you impl From
let strength: PasswordStrength = "strong".into();

// From is also how ? converts error types!
impl From<io::Error> for AuditError {
    fn from(e: io::Error) -> Self {
        AuditError::IoError(e)
    }
}
```

**`TryFrom<T>` / `TryInto<T>` — fallible conversion**:
```rust
impl TryFrom<u32> for PasswordStrength {
    type Error = String;

    fn try_from(score: u32) -> Result<Self, String> {
        match score {
            0..=25  => Ok(Self::Weak),
            26..=50 => Ok(Self::Fair),
            51..=100 => Ok(Self::Strong),
            _ => Err(format!("score {score} out of range 0-100")),
        }
    }
}

let strength = PasswordStrength::try_from(150); // Err("score 150 out of range")
```

**`Deref<Target=T>` — transparent "smart pointer" access**:
```rust
// This is why &String auto-converts to &str
// String implements Deref<Target=str>
// Box<T> implements Deref<Target=T>
// Vec<T> implements Deref<Target=[T]>
```

**Quick reference**:
| Trait | Purpose | Cost | Example |
|-------|---------|------|---------|
| `AsRef<T>` | Borrow as &T | Free | `String` → `&str` |
| `From<T>` | Create from T | May allocate | `&str` → `String` |
| `TryFrom<T>` | Fallible create | May allocate | `&str` → `u32` |
| `Deref` | Auto-dereference | Free | `Box<T>` → `&T` |

**Rule of thumb**: Implement `From` for your types. Accept `impl AsRef<str>` in functions that just read strings. The compiler handles the rest."""}
        ]
    },
    # ── MACROS ──
    {
        "category": "macros",
        "conversations": [
            {"role": "user", "content": "What are macros in Rust? There's nothing like them in Go or Python."},
            {"role": "assistant", "content": """Macros are Rust's metaprogramming system — they generate code at compile time. You've already been using them! `println!`, `vec!`, `format!`, `assert!` — anything with `!` is a macro.

**Why macros exist**: Some patterns can't be expressed with functions alone. Macros fill the gap.

**`macro_rules!` — declarative macros (pattern matching on code)**:
```rust
// Create a simple macro for password validation rules
macro_rules! check_rule {
    ($password:expr, $rule:expr, $message:literal) => {
        if !$rule($password) {
            println!("❌ {}", $message);
            false
        } else {
            println!("✅ {}", $message);
            true
        }
    };
}

fn audit(pw: &str) {
    let all_passed = true
        & check_rule!(pw, |p: &str| p.len() >= 8, "Minimum length")
        & check_rule!(pw, |p: &str| p.chars().any(|c| c.is_uppercase()), "Has uppercase")
        & check_rule!(pw, |p: &str| p.chars().any(|c| c.is_numeric()), "Has digit");
}
```

**The `vec!` macro** — you use this constantly:
```rust
// vec! is a macro because functions can't take variable arguments
let passwords = vec!["hunter2", "P@ssw0rd!", "123456"];
// Expands to: { let mut v = Vec::new(); v.push("hunter2"); ... v }
```

**Derive macros (the ones you'll use most)**:
```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct AuditConfig {
    min_length: usize,
    require_uppercase: bool,
    require_digit: bool,
}
// The compiler generates Debug, Clone, PartialEq, etc. implementations
```

**Common derive macros**:
| Derive | What it does | Go equivalent |
|--------|-------------|---------------|
| `Debug` | `{:?}` formatting | `fmt.Sprintf("%+v")` |
| `Clone` | `.clone()` method | deep copy |
| `PartialEq` | `==` comparison | manual `Equal()` method |
| `Hash` | hashable for HashMap keys | — |
| `Default` | `::default()` constructor | zero value |
| `Serialize/Deserialize` | serde JSON/TOML/etc | `json.Marshal/Unmarshal` |

**When to use macros vs functions**:
- Functions: for logic and behavior (95% of the time)
- Derive macros: for auto-generating trait implementations
- `macro_rules!`: for reducing repetitive code patterns
- Procedural macros: for advanced code generation (you probably won't write these early on)

You'll mostly *use* macros rather than *write* them. Focus on getting comfortable with `derive` and the built-in macros first."""}
        ]
    },
    {
        "category": "macros",
        "conversations": [
            {"role": "user", "content": "How do derive macros work? I see #[derive(Debug, Clone)] everywhere."},
            {"role": "assistant", "content": """Derive macros auto-generate trait implementations at compile time. Think of them as Go's `go generate` but built into the compiler.

**Without derive** — you write this manually:
```rust
struct Password {
    value: String,
    strength: u32,
}

// Manual Debug implementation (like Go's String() method)
impl std::fmt::Debug for Password {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Password")
            .field("value", &self.value)
            .field("strength", &self.strength)
            .finish()
    }
}

// Manual Clone implementation
impl Clone for Password {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            strength: self.strength,
        }
    }
}
```

**With derive** — one line does it all:
```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Password {
    value: String,
    strength: u32,
}

// Now you can:
let pw = Password { value: "hunter2".into(), strength: 10 };
println!("{:?}", pw);              // Debug
let pw2 = pw.clone();             // Clone
assert_eq!(pw, pw2);              // PartialEq
```

**The essential derives and when to use them**:
```rust
#[derive(
    Debug,         // Almost always. Enables {:?} printing. Add to every struct.
    Clone,         // When you need .clone(). Most data types should have this.
    PartialEq,     // When you need == comparison.
    Eq,            // When equality is total (most types). Requires PartialEq.
    Hash,          // When used as HashMap/HashSet keys. Requires Eq.
    Default,       // When you want MyStruct::default() with zero/empty values.
    PartialOrd,    // When you need < > comparison.
    Ord,           // When ordering is total. Requires PartialOrd + Eq.
)]
struct MyType { /* ... */ }
```

**Serde derives** (JSON/TOML/YAML serialization):
```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct AuditReport {
    #[serde(rename = "passwordCount")]  // customize JSON field name
    password_count: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,

    #[serde(default)]  // use Default if missing in JSON
    verbose: bool,
}
```

**Rule of thumb**: Put `#[derive(Debug, Clone)]` on almost everything. Add `PartialEq` when you need comparisons. Add `Serialize, Deserialize` when you touch JSON."""}
        ]
    },
    # ── COLLECTIONS ──
    {
        "category": "collections",
        "conversations": [
            {"role": "user", "content": "What collections does Rust have? In Go I mainly use slices and maps."},
            {"role": "assistant", "content": """Rust's standard library has more collection types than Go, but you'll mainly use four:

**`Vec<T>` — growable array (Go's slice)**:
```rust
let mut passwords: Vec<String> = Vec::new();
passwords.push("hunter2".to_string());
passwords.push("P@ssw0rd!".to_string());

// Shorthand with vec! macro
let passwords = vec!["hunter2", "P@ssw0rd!", "123456"];

// Slicing (like Go slices)
let first_two = &passwords[..2];

// Common operations
passwords.len();                    // length
passwords.is_empty();              // empty check
passwords.contains(&"hunter2");    // search
passwords.sort();                  // in-place sort
passwords.dedup();                 // remove consecutive duplicates
passwords.retain(|pw| pw.len() >= 8); // filter in-place
```

**`HashMap<K, V>` — hash map (Go's map)**:
```rust
use std::collections::HashMap;

let mut scores: HashMap<String, u32> = HashMap::new();
scores.insert("hunter2".to_string(), 10);
scores.insert("P@ssw0rd!".to_string(), 75);

// Get (returns Option, not (value, ok) like Go)
if let Some(score) = scores.get("hunter2") {
    println!("Score: {score}");
}

// Entry API — Go doesn't have this! Update or insert in one call:
scores.entry("newpass".to_string())
    .or_insert(0);  // insert 0 if not present

// Count character frequencies:
let mut freq: HashMap<char, usize> = HashMap::new();
for c in password.chars() {
    *freq.entry(c).or_insert(0) += 1;
}
```

**`HashSet<T>` — set (Go doesn't have a built-in set)**:
```rust
use std::collections::HashSet;

let common_passwords: HashSet<&str> = [
    "password", "123456", "qwerty", "hunter2"
].into_iter().collect();

// O(1) membership check
if common_passwords.contains(user_password) {
    println!("❌ Common password detected!");
}

// Set operations
let set_a: HashSet<_> = vec![1, 2, 3].into_iter().collect();
let set_b: HashSet<_> = vec![2, 3, 4].into_iter().collect();
let intersection: HashSet<_> = set_a.intersection(&set_b).collect();
let union: HashSet<_> = set_a.union(&set_b).collect();
```

**`BTreeMap<K, V>` / `BTreeSet<T>` — sorted versions**:
```rust
use std::collections::BTreeMap;

// Keys are always sorted — useful for ordered output
let mut strength_groups: BTreeMap<u32, Vec<&str>> = BTreeMap::new();
strength_groups.entry(10).or_default().push("hunter2");
strength_groups.entry(75).or_default().push("P@ssw0rd!");

// Iterates in key order (HashMap doesn't guarantee order)
for (score, passwords) in &strength_groups {
    println!("Score {score}: {passwords:?}");
}
```

**`VecDeque<T>` — double-ended queue**:
```rust
use std::collections::VecDeque;

// Efficient push/pop from both ends (Vec is only efficient at the back)
let mut recent: VecDeque<String> = VecDeque::with_capacity(10);
recent.push_back("latest".to_string());
recent.push_front("earliest".to_string());
```

**Quick pick guide**:
| Need | Use | Go equivalent |
|------|-----|---------------|
| Ordered list | `Vec<T>` | `[]T` slice |
| Key-value | `HashMap<K,V>` | `map[K]V` |
| Unique items | `HashSet<T>` | `map[T]struct{}` |
| Sorted key-value | `BTreeMap<K,V>` | — (need library) |
| Queue/deque | `VecDeque<T>` | — (use slice tricks) |"""}
        ]
    },
    # ── SERDE & SERIALIZATION ──
    {
        "category": "serde",
        "conversations": [
            {"role": "user", "content": "How do I work with JSON in Rust? In Go I use encoding/json with struct tags."},
            {"role": "assistant", "content": """Serde is Rust's serialization framework — it's like Go's `encoding/json` but works with JSON, TOML, YAML, MessagePack, and more with the same derive macros.

```bash
cargo add serde --features derive
cargo add serde_json
```

**Basic JSON (Go struct tags → Serde attributes)**:
```rust
use serde::{Serialize, Deserialize};

// Go equivalent:
// type AuditResult struct {
//     Password  string  `json:"password"`
//     Score     int     `json:"score"`
//     Issues    []string `json:"issues,omitempty"`
// }

#[derive(Serialize, Deserialize, Debug)]
struct AuditResult {
    password: String,
    score: u32,

    #[serde(skip_serializing_if = "Vec::is_empty")]  // like omitempty
    issues: Vec<String>,

    #[serde(rename = "strengthLabel")]  // custom JSON key
    strength_label: String,

    #[serde(default)]  // use Default if missing during deserialization
    verbose: bool,

    #[serde(skip)]  // never serialize/deserialize this field
    internal_hash: u64,
}
```

**Serialize (like json.Marshal)**:
```rust
let result = AuditResult {
    password: "hunter2".into(),
    score: 10,
    issues: vec!["too short".into(), "common password".into()],
    strength_label: "WEAK".into(),
    verbose: false,
    internal_hash: 0,
};

// To JSON string
let json = serde_json::to_string(&result)?;
let json_pretty = serde_json::to_string_pretty(&result)?;

// To a file
let file = std::fs::File::create("report.json")?;
serde_json::to_writer_pretty(file, &result)?;
```

**Deserialize (like json.Unmarshal)**:
```rust
// From string
let json_str = r#"{"password": "hunter2", "score": 10, "strengthLabel": "WEAK"}"#;
let result: AuditResult = serde_json::from_str(json_str)?;

// From file
let file = std::fs::File::open("report.json")?;
let result: AuditResult = serde_json::from_reader(file)?;

// Dynamic JSON (like Go's map[string]interface{})
let dynamic: serde_json::Value = serde_json::from_str(json_str)?;
println!("{}", dynamic["password"]);  // "hunter2"
```

**Enum serialization** (Rust's killer feature over Go):
```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]  // adds a "type" field to JSON
enum CheckResult {
    #[serde(rename = "pass")]
    Pass { checker: String },
    #[serde(rename = "fail")]
    Fail { checker: String, reason: String },
}

// Serializes to: {"type": "fail", "checker": "length", "reason": "too short"}
```

**TOML** (for config files):
```bash
cargo add toml
```
```rust
let config: AuditConfig = toml::from_str(&fs::read_to_string("config.toml")?)?;
```

Serde + derive is one of the best parts of the Rust ecosystem. Once you derive `Serialize`/`Deserialize`, your types work with every data format."""}
        ]
    },
    # ── FILE I/O ──
    {
        "category": "file_io",
        "conversations": [
            {"role": "user", "content": "How do I read and write files in Rust? In Go I use os and bufio packages."},
            {"role": "assistant", "content": """Rust's file I/O is in `std::fs` and `std::io`. The patterns map closely to Go's `os` and `bufio` packages.

**Read entire file (Go's `os.ReadFile`)**:
```rust
use std::fs;

// Simple: read entire file to string
let content = fs::read_to_string("passwords.txt")?;

// Read to bytes
let bytes = fs::read("binary_file.dat")?;

// Process lines
let passwords: Vec<String> = fs::read_to_string("passwords.txt")?
    .lines()
    .map(|line| line.trim().to_string())
    .filter(|line| !line.is_empty())
    .collect();
```

**BufReader for large files (Go's `bufio.Scanner`)**:
```rust
use std::io::{BufRead, BufReader};
use std::fs::File;

// Go equivalent:
// file, _ := os.Open("passwords.txt")
// scanner := bufio.NewScanner(file)
// for scanner.Scan() { line := scanner.Text() }

let file = File::open("passwords.txt")?;
let reader = BufReader::new(file);

for line in reader.lines() {
    let line = line?;  // lines() returns Result<String> per line
    if line.len() < 8 {
        println!("Weak: {line}");
    }
}
```

**Write files (Go's `os.WriteFile`)**:
```rust
// Simple write
fs::write("output.txt", "Hello, Rust!")?;

// Write with more control
use std::io::Write;

let mut file = File::create("report.txt")?;
writeln!(file, "Password Audit Report")?;
writeln!(file, "=====================")?;
for pw in &weak_passwords {
    writeln!(file, "❌ {pw}")?;
}

// BufWriter for performance (like Go's bufio.Writer)
use std::io::BufWriter;
let file = File::create("large_output.txt")?;
let mut writer = BufWriter::new(file);
for i in 0..1_000_000 {
    writeln!(writer, "line {i}")?;
}
// BufWriter flushes automatically when dropped
```

**Path handling (like Go's `filepath` package)**:
```rust
use std::path::{Path, PathBuf};

let path = Path::new("data/passwords.txt");
println!("{}", path.display());
println!("{:?}", path.extension());       // Some("txt")
println!("{:?}", path.file_stem());       // Some("passwords")
println!("{:?}", path.parent());          // Some("data")

// Build paths
let mut output = PathBuf::from("results");
output.push("report.json");
// → "results/report.json"

// Check existence
if path.exists() && path.is_file() {
    let content = fs::read_to_string(path)?;
}
```

**Create directories (like `os.MkdirAll`)**:
```rust
fs::create_dir_all("data/processed")?;  // creates parent dirs too
```

**Practical passaudit pattern**:
```rust
fn audit_file(path: &str) -> anyhow::Result<Vec<AuditResult>> {
    let file = File::open(path)
        .with_context(|| format!("Cannot open {path}"))?;

    BufReader::new(file)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.with_context(|| format!("Error reading line {}", i + 1))?;
            Ok(audit_password(&line))
        })
        .collect()
}
```

Use `fs::read_to_string` for small files, `BufReader` for large files or when you want line-by-line processing without loading everything into memory."""}
        ]
    },
    # ── UNSAFE RUST ──
    {
        "category": "unsafe_rust",
        "conversations": [
            {"role": "user", "content": "What is unsafe in Rust? When would I ever need it?"},
            {"role": "assistant", "content": """`unsafe` lets you do things the compiler can't verify are safe. It's like Go's `unsafe` package — you rarely need it, but it exists for when you do.

**What `unsafe` unlocks (and only these things)**:
1. Dereference raw pointers
2. Call unsafe functions (including C FFI)
3. Access mutable static variables
4. Implement unsafe traits

```rust
// You'll rarely write this, but it's good to understand:
let mut value = 42;
let ptr: *mut i32 = &mut value;

unsafe {
    *ptr = 100;  // dereferencing raw pointer requires unsafe
}
println!("{value}"); // 100
```

**When you'll encounter it in practice**:

**1. FFI — calling C libraries**:
```rust
// Calling a C function
extern "C" {
    fn strlen(s: *const std::os::raw::c_char) -> usize;
}

fn c_string_length(s: &std::ffi::CStr) -> usize {
    unsafe { strlen(s.as_ptr()) }  // C can't guarantee safety
}
```

**2. Performance-critical code** (rare):
```rust
// get_unchecked skips bounds checking
let data = vec![1, 2, 3, 4, 5];
let val = unsafe { data.get_unchecked(2) };
// Only do this if profiling shows bounds checks are a bottleneck
// (they almost never are)
```

**3. Implementing data structures** (you won't do this often):
```rust
// Linked lists, custom allocators, lock-free structures
// require raw pointers internally
```

**What unsafe does NOT mean**:
- It doesn't disable the borrow checker
- It doesn't turn off all safety checks
- Code inside `unsafe` blocks can still be safe Rust
- It's a signal: "I, the programmer, have verified this is correct"

**The Go comparison**:
- Go's `unsafe.Pointer` ≈ Rust raw pointers inside `unsafe {}`
- Go's cgo ≈ Rust's FFI with `extern "C"`
- But Go's GC still protects memory; Rust's `unsafe` truly removes the safety net

**Rule of thumb for beginners**: If you think you need `unsafe`, you probably don't. Ask yourself:
1. Can I use a safe abstraction instead? (Almost always yes)
2. Am I wrapping a C library? (Use the `cc` or `bindgen` crate)
3. Am I doing low-level optimization? (Profile first — safe Rust is already fast)

For passaudit, you will never need `unsafe`. If you find yourself reaching for it, there's a safe pattern you're missing."""}
        ]
    },
    # ── CLOSURES IN DEPTH ──
    {
        "category": "closures",
        "conversations": [
            {"role": "user", "content": "I keep getting errors about Fn, FnMut, and FnOnce when using closures. What's the difference?"},
            {"role": "assistant", "content": """These are the three closure traits — they describe how a closure uses its captured variables. In Go and Python, closures "just work." In Rust, the compiler tracks what closures do with captured data.

**The three traits** (from most to least restrictive):

```rust
// Fn — borrows captured variables immutably (can be called many times)
let name = String::from("auditor");
let greet = || println!("Hello from {name}");  // captures &name
greet();  // ✅
greet();  // ✅ can call again
println!("{name}");  // ✅ name still valid

// FnMut — borrows captured variables mutably (can be called many times)
let mut count = 0;
let mut counter = || { count += 1; count };  // captures &mut count
counter();  // ✅
counter();  // ✅
// But can't use `count` while `counter` borrows it mutably

// FnOnce — takes ownership of captured variables (can only be called once)
let name = String::from("auditor");
let consume = move || {
    drop(name);  // takes ownership, destroys name
};
consume();  // ✅
// consume();  // ❌ can't call again — name is gone
```

**The hierarchy**: `Fn` is a subset of `FnMut`, which is a subset of `FnOnce`.
- Every `Fn` closure is also `FnMut` and `FnOnce`
- Every `FnMut` closure is also `FnOnce`
- `FnOnce` is the most general

**When you hit this in practice**:
```rust
// Functions that take closures specify which trait they need:

// Iterator methods use FnMut (need to call multiple times)
passwords.iter().filter(|pw| pw.len() >= 8);  // Fn works here too

// thread::spawn requires FnOnce + Send (closure moves to another thread)
let passwords = vec!["pw1".to_string()];
std::thread::spawn(move || {  // `move` makes it FnOnce
    println!("{:?}", passwords);
});

// Custom function accepting closures
fn apply_check<F: Fn(&str) -> bool>(password: &str, checker: F) -> bool {
    checker(password)  // Fn because we might want to reuse checker
}

fn apply_transform<F: FnMut(&mut String)>(passwords: &mut [String], mut f: F) {
    for pw in passwords {
        f(pw);  // FnMut because f might track state
    }
}
```

**The `move` keyword**: Forces the closure to take ownership of captured variables.
```rust
let threshold = 8;  // i32 is Copy, so move copies it
let checker = move |pw: &str| pw.len() >= threshold;
// threshold is still usable (i32 was copied, not moved)

let wordlist = vec!["password".to_string()];  // Vec is not Copy
let checker = move |pw: &str| wordlist.contains(&pw.to_string());
// wordlist was MOVED into the closure — can't use it anymore
```

**Rule of thumb**: Start with `Fn`. If the compiler complains, try `FnMut`. Use `FnOnce` when the closure needs to consume captured data (like spawning threads)."""}
        ]
    },
    # ── MODULES & VISIBILITY ──
    {
        "category": "modules_visibility",
        "conversations": [
            {"role": "user", "content": "How does visibility work in Rust? In Go it's just uppercase = public, lowercase = private."},
            {"role": "assistant", "content": """Rust has more granular visibility than Go's uppercase/lowercase convention. Everything is **private by default** — the opposite of Go where everything within a package is accessible.

**Visibility levels**:
```rust
mod checkers {
    // Private by default — only accessible within this module
    fn internal_helper() {}

    // pub — public to everyone
    pub fn check_password(pw: &str) -> bool {
        internal_helper();  // can use private items within same module
        pw.len() >= 8
    }

    // pub(crate) — public within this crate only (like Go's package-level)
    pub(crate) fn crate_only_helper() {}

    // pub(super) — public to parent module only
    pub(super) fn parent_only() {}

    pub struct Checker {
        pub name: String,          // public field
        min_length: usize,         // private field — even if struct is pub!
        pub(crate) cache: Vec<String>, // crate-visible field
    }

    impl Checker {
        // Public constructor needed because min_length is private
        pub fn new(name: &str, min: usize) -> Self {
            Self {
                name: name.to_string(),
                min_length: min,
                cache: Vec::new(),
            }
        }

        // Private method
        fn validate_internally(&self) {}

        // Public method
        pub fn check(&self, pw: &str) -> bool {
            self.validate_internally();
            pw.len() >= self.min_length
        }
    }
}
```

**Key differences from Go**:
| Go | Rust | Notes |
|----|------|-------|
| Uppercase = public | `pub` = public | Explicit keyword |
| Lowercase = package-private | Default = private | Private to the module |
| No sub-package privacy | `pub(crate)`, `pub(super)` | Granular control |
| All fields accessible in package | Fields are private by default | Even on pub structs |

**Module re-exports** (controlling your public API):
```rust
// src/lib.rs
mod checkers;      // private module
mod config;        // private module

// Re-export only what users need
pub use checkers::Checker;
pub use checkers::check_password;
pub use config::Config;
// internal helpers stay hidden
```

**The `pub use` pattern** is like Go's practice of having a package expose only certain types, but explicit.

**Practical guideline**:
- Start everything as private
- Add `pub` only to what your callers need
- Use `pub(crate)` for internal helpers shared between modules
- If a struct has private fields, provide a `pub fn new()` constructor"""}
        ]
    },
    # ── CARGO WORKSPACES ──
    {
        "category": "cargo_workspaces",
        "conversations": [
            {"role": "user", "content": "My project is getting big. How do I split it into multiple crates? In Go I'd use separate modules."},
            {"role": "assistant", "content": """Cargo workspaces let you split a project into multiple crates that share a single `Cargo.lock` and build directory. It's like Go's multi-module workspace (`go.work`).

**When to use a workspace**:
- Your project has a library + binary (or multiple binaries)
- You want to share code between packages without publishing to crates.io
- You have independent components that should compile separately

**Setting up a workspace**:
```
passaudit/
├── Cargo.toml              # workspace root
├── crates/
│   ├── passaudit-core/     # library crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs
│   ├── passaudit-cli/      # binary crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs
│   └── passaudit-web/      # another binary crate
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
```

**Root `Cargo.toml`**:
```toml
[workspace]
members = [
    "crates/passaudit-core",
    "crates/passaudit-cli",
    "crates/passaudit-web",
]

# Share dependency versions across the workspace
[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
anyhow = "1"
clap = { version = "4", features = ["derive"] }
```

**Crate `Cargo.toml`** (e.g., `crates/passaudit-cli/Cargo.toml`):
```toml
[package]
name = "passaudit-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
passaudit-core = { path = "../passaudit-core" }  # local dependency!
anyhow.workspace = true  # inherit from workspace
clap.workspace = true
```

**Using the library crate**:
```rust
// crates/passaudit-cli/src/main.rs
use passaudit_core::{Auditor, AuditConfig, check_password};

fn main() -> anyhow::Result<()> {
    let auditor = Auditor::new(AuditConfig::default());
    auditor.audit_file("passwords.txt")?;
    Ok(())
}
```

**Build and run**:
```bash
cargo build                      # builds all crates
cargo run -p passaudit-cli       # run specific binary
cargo test --workspace           # test all crates
cargo test -p passaudit-core     # test specific crate
```

**Go comparison**:
- `go.work` ≈ workspace `Cargo.toml`
- `go mod init` per module ≈ `Cargo.toml` per crate
- `replace` directive ≈ `path = "../local-crate"`

**When NOT to use workspaces**: If your project is small (< 5000 lines), a single crate with modules is simpler. Use workspaces when you genuinely need separate compilation units or want to publish parts independently."""}
        ]
    },
    # ── DOCUMENTATION ──
    {
        "category": "documentation",
        "conversations": [
            {"role": "user", "content": "How do I document my Rust code? In Go I use godoc comments."},
            {"role": "assistant", "content": """Rust has a built-in documentation system called **rustdoc** — it generates HTML docs from comments, and it can even run code examples as tests!

**Doc comments use `///` (not `//`)**:
```rust
/// Check if a password meets minimum security requirements.
///
/// Evaluates the password against length, complexity, and
/// common password database checks.
///
/// # Arguments
///
/// * `password` - The password to check
/// * `min_length` - Minimum required length
///
/// # Returns
///
/// `true` if the password passes all checks
///
/// # Examples
///
/// ```
/// use passaudit::check_password;
///
/// assert!(check_password("P@ssw0rd!123", 8));
/// assert!(!check_password("short", 8));
/// ```
///
/// # Panics
///
/// Panics if `min_length` is 0.
pub fn check_password(password: &str, min_length: usize) -> bool {
    assert!(min_length > 0, "min_length must be > 0");
    password.len() >= min_length
}
```

**Module and crate docs use `//!`**:
```rust
//! # PassAudit Core
//!
//! A library for auditing password security.
//!
//! ## Quick Start
//!
//! ```
//! use passaudit_core::Auditor;
//!
//! let auditor = Auditor::new(8);
//! let result = auditor.check("hunter2");
//! ```

pub struct Auditor { /* ... */ }
```

**Doc tests — the killer feature** (Go has `Example` functions, Rust has inline examples):
```bash
cargo test --doc  # runs all code examples in /// comments as tests!
```

Every ```` ```rust ```` block in your doc comments becomes a test. If the example doesn't compile or panics, `cargo test` catches it. This means your docs never go stale.

**Standard sections**:
| Section | Purpose |
|---------|---------|
| `# Examples` | Usage examples (tested!) |
| `# Panics` | When the function panics |
| `# Errors` | When it returns `Err` |
| `# Safety` | For `unsafe` functions |
| `# Arguments` | Parameter descriptions |

**Generate and view docs**:
```bash
cargo doc              # generate HTML docs
cargo doc --open       # generate and open in browser
cargo doc --no-deps    # skip dependency docs
```

**Linking to other items**:
```rust
/// Uses [`Auditor::check`] internally.
/// See [`AuditConfig`] for configuration options.
pub fn audit_file(path: &str) -> Vec<AuditResult> { /* ... */ }
```

**Go comparison**:
- `///` ≈ `//` before declarations (godoc)
- `# Examples` with code blocks ≈ `func ExampleFoo()` in `_test.go`
- `cargo doc` ≈ `godoc` / `go doc`
- But Rust tests the examples automatically — Go example tests are optional

**Rule**: At minimum, add `///` docs to all `pub` items. Add examples for functions that aren't obvious."""}
        ]
    },
    # ── TYPE ALIASES & NEWTYPE PATTERN ──
    {
        "category": "newtype",
        "conversations": [
            {"role": "user", "content": "What's the newtype pattern in Rust? In Go I use type aliases like `type UserID int`."},
            {"role": "assistant", "content": """Rust has two approaches: type aliases (like Go's type aliases) and the newtype pattern (like Go's defined types but stronger).

**Type aliases** — just a shorthand, no type safety:
```rust
type Password = String;
type Score = u32;

// These are exactly the same type — no safety!
let pw: Password = "hunter2".to_string();
let s: String = pw;  // works, they're the same type
```

**Newtype pattern** — wrapper struct, full type safety:
```rust
// Go: type UserID int64 — creates a distinct type
// Rust equivalent:
struct Password(String);
struct Score(u32);

let pw = Password("hunter2".to_string());
let s: String = pw.0;  // must explicitly unwrap with .0
// let s: String = pw;  // ❌ compile error — different types!
```

**Why newtype is powerful**:

**1. Prevent mixing up arguments**:
```rust
// ❌ Easy to mix up — both are just strings
fn create_user(email: String, password: String) {}

// ✅ Compiler catches mistakes
struct Email(String);
struct HashedPassword(String);

fn create_user(email: Email, password: HashedPassword) {}

// create_user(password, email) would be a compile error!
```

**2. Add methods to external types**:
```rust
// Can't impl methods on String directly (orphan rule)
// But you CAN on a newtype wrapper
struct Password(String);

impl Password {
    fn strength(&self) -> u32 {
        let mut score = 0;
        if self.0.len() >= 8 { score += 25; }
        if self.0.chars().any(|c| c.is_uppercase()) { score += 25; }
        score
    }

    fn is_common(&self) -> bool {
        ["password", "123456", "hunter2"].contains(&self.0.as_str())
    }
}
```

**3. Implement traits differently**:
```rust
struct Celsius(f64);
struct Fahrenheit(f64);

impl std::fmt::Display for Celsius {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:.1}°C", self.0)
    }
}
```

**Make newtypes ergonomic with Deref**:
```rust
use std::ops::Deref;

struct Password(String);

impl Deref for Password {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

let pw = Password("hunter2".to_string());
println!("Length: {}", pw.len());  // auto-derefs to &str methods
```

**Go comparison**:
- Go `type X = Y` (alias) ≈ Rust `type X = Y`
- Go `type X Y` (defined type) ≈ Rust `struct X(Y)` (newtype)
- But Rust newtypes are zero-cost at runtime — the wrapper is optimized away

**When to use**: Use newtypes when you want the compiler to prevent you from accidentally mixing up values that have the same underlying type but different meanings."""}
        ]
    },
    # ── ADDITIONAL OWNERSHIP: COPY & MOVE SEMANTICS ──
    {
        "category": "ownership",
        "conversations": [
            {"role": "user", "content": "Why do some types copy and others move? I don't understand the Copy trait."},
            {"role": "assistant", "content": """Great question — this is fundamental to how Rust thinks about values.

**Move semantics (default for heap types)**:
```rust
let a = String::from("hello");
let b = a;  // MOVE — a is invalid now
// println!("{a}");  // ❌ compile error
```

**Copy semantics (for simple stack types)**:
```rust
let a: i32 = 42;
let b = a;  // COPY — both a and b are valid
println!("{a}");  // ✅ still works
```

**The `Copy` trait** marks types that can be duplicated by just copying bits (like `memcpy`). Types are `Copy` if:
1. They live entirely on the stack
2. They don't own heap resources
3. They don't implement `Drop`

**What implements Copy**:
```rust
// ✅ Copy types — cheap bit copy
let x: i32 = 42;        // integers
let y: f64 = 3.14;      // floats
let z: bool = true;     // booleans
let c: char = 'a';      // characters
let t: (i32, bool) = (1, true);  // tuples of Copy types
let a: [i32; 5] = [0; 5];       // arrays of Copy types
let r: &str = "hello";           // references (Copy the pointer, not data)

// ❌ NOT Copy — own heap data
let s: String = String::from("hello");   // owns heap buffer
let v: Vec<i32> = vec![1, 2, 3];        // owns heap array
let b: Box<i32> = Box::new(42);         // owns heap allocation
```

**Making your own types Copy**:
```rust
// Small, stack-only types can derive Copy
#[derive(Debug, Clone, Copy)]
struct Score {
    value: u32,
    max: u32,
}

let s1 = Score { value: 75, max: 100 };
let s2 = s1;  // COPY — both valid
println!("{s1:?}");  // ✅

// Can't derive Copy if any field is non-Copy
#[derive(Debug, Clone)]  // Clone but NOT Copy
struct AuditResult {
    score: u32,       // Copy
    issues: Vec<String>,  // NOT Copy → whole struct can't be Copy
}
```

**The relationship**: `Copy` is a subtrait of `Clone`. All `Copy` types are `Clone`, but not all `Clone` types are `Copy`.
- `Copy`: implicit, bitwise duplication (automatic on assignment)
- `Clone`: explicit, potentially expensive (must call `.clone()`)

**Go comparison**: In Go, everything is either copied (value types) or shared (pointers). Rust makes this distinction explicit and enforced:
- Go value copy ≈ Rust Copy
- Go pointer sharing ≈ Rust borrowing (`&T`)
- No Go equivalent ≈ Rust move (Go has no concept of invalidating a variable)

**Practical rule**: Use `#[derive(Clone, Copy)]` for small value types (scores, coordinates, IDs). Don't fight the borrow checker by making everything Copy — use references instead."""}
        ]
    },
    # ── ADDITIONAL TRAITS: DISPLAY, DEFAULT, ITERATOR ──
    {
        "category": "traits",
        "conversations": [
            {"role": "user", "content": "What are the most important standard library traits I should know?"},
            {"role": "assistant", "content": """Good question — knowing the key traits is like knowing Go's key interfaces (`io.Reader`, `fmt.Stringer`, `error`). Here are the ones you'll use constantly:

**`Display` — for user-facing output (Go's `fmt.Stringer`)**:
```rust
use std::fmt;

struct AuditReport {
    total: usize,
    weak: usize,
    strong: usize,
}

impl fmt::Display for AuditReport {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Audited {} passwords: {} weak, {} strong",
            self.total, self.weak, self.strong)
    }
}

let report = AuditReport { total: 100, weak: 30, strong: 70 };
println!("{report}");  // uses Display
// vs {:?} which uses Debug (derived)
```

**`Default` — zero/empty values (Go's zero values)**:
```rust
#[derive(Default)]
struct AuditConfig {
    min_length: usize,      // defaults to 0
    require_digit: bool,    // defaults to false
    max_attempts: Option<u32>, // defaults to None
    checkers: Vec<String>,  // defaults to empty vec
}

// Useful with struct update syntax:
let config = AuditConfig {
    min_length: 8,
    ..AuditConfig::default()  // fill remaining with defaults
};
```

**`Iterator` — the engine behind for loops and chains**:
```rust
struct PasswordGenerator {
    length: usize,
    count: usize,
    current: usize,
}

impl Iterator for PasswordGenerator {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if self.current >= self.count {
            return None;
        }
        self.current += 1;
        Some(generate_random_password(self.length))
    }
}

// Now you get .map(), .filter(), .collect(), etc. for free!
let passwords: Vec<String> = PasswordGenerator { length: 16, count: 10, current: 0 }
    .filter(|pw| pw.chars().any(|c| c.is_uppercase()))
    .collect();
```

**`From`/`Into` — type conversions (covered earlier)**

**`Error` — for custom error types (Go's `error` interface)**:
```rust
use std::fmt;

#[derive(Debug)]
enum AuditError {
    FileNotFound(String),
    TooShort { length: usize },
}

impl fmt::Display for AuditError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "file not found: {p}"),
            Self::TooShort { length } => write!(f, "password too short: {length} chars"),
        }
    }
}

impl std::error::Error for AuditError {}
// Or just use #[derive(thiserror::Error)] to skip the boilerplate
```

**Quick reference**:
| Trait | Purpose | Go equivalent | Derive? |
|-------|---------|---------------|---------|
| `Debug` | `{:?}` formatting | `%+v` | Yes |
| `Display` | `{}` formatting | `Stringer` | No (manual) |
| `Clone` | `.clone()` | deep copy | Yes |
| `Copy` | Implicit copy | value semantics | Yes |
| `Default` | `::default()` | zero value | Yes |
| `PartialEq`/`Eq` | `==` operator | `==` | Yes |
| `Hash` | HashMap keys | — | Yes |
| `Iterator` | for loops, chains | range protocol | No |
| `Error` | error type | `error` interface | thiserror |
| `From`/`Into` | type conversion | — | No |

**Rule**: Derive `Debug`, `Clone`, `PartialEq` on almost everything. Implement `Display` for types users will see. Implement `Default` for config structs."""}
        ]
    },
    # ── ADDITIONAL ERROR HANDLING: if let & let-else ──
    {
        "category": "error_handling",
        "conversations": [
            {"role": "user", "content": "What's if-let and let-else? I see these a lot in Rust code."},
            {"role": "assistant", "content": """`if let` and `let-else` are shorthand for pattern matching when you only care about one variant. They reduce boilerplate compared to full `match` statements.

**`if let` — handle one pattern, ignore the rest**:
```rust
// Full match (verbose when you only care about Some)
match config.get("port") {
    Some(port) => println!("Port: {port}"),
    None => {}  // do nothing
}

// if let (concise)
if let Some(port) = config.get("port") {
    println!("Port: {port}");
}

// With else
if let Some(port) = config.get("port") {
    println!("Using port {port}");
} else {
    println!("Using default port 3000");
}

// Works with any pattern, not just Option
if let AuditResult::Fail { reason, .. } = &result {
    eprintln!("Failed: {reason}");
}
```

**`let-else` (Rust 1.65+) — unwrap or return early**:
```rust
// This is THE pattern for early returns — cleaner than match or if-let

// ❌ Nested if-let
fn process(input: Option<String>) -> Result<(), Error> {
    if let Some(value) = input {
        // all your logic indented one level
        do_stuff(&value)?;
        Ok(())
    } else {
        return Err(Error::new("no input"));
    }
}

// ✅ let-else — flat, readable
fn process(input: Option<String>) -> Result<(), Error> {
    let Some(value) = input else {
        return Err(Error::new("no input"));
    };

    // continue with value — no extra indentation!
    do_stuff(&value)?;
    Ok(())
}
```

**Practical passaudit examples**:
```rust
fn audit_password(pw: &str) -> AuditResult {
    // Guard clause with let-else
    let Some(first_char) = pw.chars().next() else {
        return AuditResult::Fail {
            reason: "empty password".into()
        };
    };

    // ... continue with first_char available
}

fn load_config(path: &str) -> anyhow::Result<Config> {
    // Chain with if-let
    let content = fs::read_to_string(path)?;

    if let Some(line) = content.lines().find(|l| l.starts_with("min_length=")) {
        let value = line.strip_prefix("min_length=").unwrap();
        println!("Found min_length: {value}");
    }

    // ...
}
```

**`while let` — loop while pattern matches**:
```rust
// Process items from a channel until it closes
while let Ok(password) = receiver.recv() {
    audit_password(&password);
}

// Pop items from a stack
let mut stack = vec![1, 2, 3];
while let Some(top) = stack.pop() {
    println!("{top}");
}
```

**Go comparison**:
- `if let Some(x) = expr` ≈ `if x, ok := expr; ok { ... }`
- `let Some(x) = expr else { return }` ≈ `x, ok := expr; if !ok { return }`
- `while let` ≈ `for msg := range channel { ... }`

**When to use what**:
- One variant: `if let` or `let-else`
- Two variants: `if let` + `else`
- Many variants: full `match`
- Unwrap-or-return: `let-else` (most readable)"""}
        ]
    },
    # ── ADDITIONAL CODE REVIEW ──
    {
        "category": "code_review",
        "conversations": [
            {"role": "user", "content": "```rust\nfn process_data(data: &Vec<String>) -> Vec<String> {\n    let mut results: Vec<String> = Vec::new();\n    for i in 0..data.len() {\n        let item = data[i].clone();\n        let processed = item.to_uppercase();\n        if processed.len() > 0 {\n            results.push(processed);\n        }\n    }\n    return results;\n}\n```\nCan you review this?"},
            {"role": "assistant", "content": """Several improvements here — let me walk through each:

**1. Take `&[String]` instead of `&Vec<String>`**:
```rust
// ❌ Overly specific — only accepts Vec
fn process_data(data: &Vec<String>) -> Vec<String>

// ✅ Accepts Vec, arrays, slices — anything contiguous
fn process_data(data: &[String]) -> Vec<String>
```
Clippy will warn about `&Vec<T>` — `&[T]` is almost always better because it accepts more input types via Deref coercion.

**2. Use iterators instead of index-based loop**:
```rust
// ❌ Index-based with unnecessary clone
for i in 0..data.len() {
    let item = data[i].clone();
}

// ✅ Iterator — no indexing, no clone needed
for item in data {
    let processed = item.to_uppercase();
}
```

**3. Use `!is_empty()` instead of `len() > 0`**:
```rust
// ❌ Verbose
if processed.len() > 0 { ... }

// ✅ Idiomatic
if !processed.is_empty() { ... }
```

**4. Drop explicit `return`**:
```rust
// ❌ C-style return
return results;

// ✅ Rust convention — last expression is the return value
results
```

**5. Use an iterator chain for the whole thing**:
```rust
fn process_data(data: &[String]) -> Vec<String> {
    data.iter()
        .map(|item| item.to_uppercase())
        .filter(|s| !s.is_empty())
        .collect()
}
```

That's it — 7 lines → 4 lines, more idiomatic, and no unnecessary allocations (removed the `.clone()`).

**Bonus: if the input might contain `&str`**:
```rust
fn process_data(data: &[impl AsRef<str>]) -> Vec<String> {
    data.iter()
        .map(|item| item.as_ref().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect()
}
// Now works with Vec<String>, Vec<&str>, &[String], &[&str], etc.
```

**What the borrow checker saved us**: The original cloned every string unnecessarily. By iterating with references, we only allocate for the `.to_uppercase()` results we actually keep."""}
        ]
    },
    # ── ADDITIONAL ITERATORS: ADVANCED PATTERNS ──
    {
        "category": "iterators",
        "conversations": [
            {"role": "user", "content": "What are some advanced iterator patterns in Rust? I want to write more idiomatic code."},
            {"role": "assistant", "content": """Here are the iterator methods that level up your Rust code — these are the ones experienced Rustaceans reach for:

**`filter_map` — filter and transform in one step**:
```rust
// ❌ Separate filter + map
let ports: Vec<u16> = config_lines.iter()
    .map(|line| line.parse::<u16>())
    .filter(|r| r.is_ok())
    .map(|r| r.unwrap())
    .collect();

// ✅ filter_map does both
let ports: Vec<u16> = config_lines.iter()
    .filter_map(|line| line.parse::<u16>().ok())
    .collect();
```

**`flat_map` — map then flatten nested iterators**:
```rust
// Get all characters from all passwords
let all_chars: Vec<char> = passwords.iter()
    .flat_map(|pw| pw.chars())
    .collect();

// Split each line into words and collect all words
let words: Vec<&str> = lines.iter()
    .flat_map(|line| line.split_whitespace())
    .collect();
```

**`chunks` / `windows` — process in groups**:
```rust
// Process passwords in batches of 100
for batch in passwords.chunks(100) {
    process_batch(batch);  // batch is &[String]
}

// Sliding window (pairs of consecutive items)
let data = vec![1, 5, 3, 8, 2];
for window in data.windows(2) {
    println!("{} → {} (diff: {})", window[0], window[1],
        (window[1] - window[0]).abs());
}
```

**`zip` — pair up two iterators (like Python's zip)**:
```rust
let passwords = vec!["hunter2", "P@ssw0rd!", "123456"];
let scores = vec![10, 85, 5];

for (pw, score) in passwords.iter().zip(scores.iter()) {
    println!("{pw}: {score}/100");
}

// Practical: number your results
let numbered: Vec<String> = passwords.iter()
    .zip(1..)  // infinite counter starting at 1
    .map(|(pw, n)| format!("{n}. {pw}"))
    .collect();
```

**`fold` / `reduce` — accumulate values**:
```rust
// Sum of all password lengths
let total_len: usize = passwords.iter()
    .map(|pw| pw.len())
    .sum();  // sugar for fold(0, |acc, x| acc + x)

// Custom accumulation
let summary = passwords.iter().fold(
    (0usize, 0usize),  // (weak_count, strong_count)
    |(weak, strong), pw| {
        if pw.len() < 8 { (weak + 1, strong) }
        else { (weak, strong + 1) }
    },
);
```

**`take` / `skip` — slice iterators**:
```rust
// First 10 weak passwords
let top_weak: Vec<&str> = passwords.iter()
    .filter(|pw| pw.len() < 8)
    .take(10)
    .copied()
    .collect();

// Skip header line
for line in content.lines().skip(1) {
    process_password(line);
}
```

**`any` / `all` — boolean checks**:
```rust
let has_weak = passwords.iter().any(|pw| pw.len() < 8);
let all_strong = passwords.iter().all(|pw| pw.len() >= 12);

// Short-circuits! Stops at first match/failure.
```

**`partition` — split into two collections**:
```rust
let (strong, weak): (Vec<&str>, Vec<&str>) = passwords.iter()
    .partition(|pw| pw.len() >= 8);
```

**`collect` into different types**:
```rust
// Into Vec
let v: Vec<_> = iter.collect();

// Into HashMap
let map: HashMap<String, usize> = passwords.iter()
    .map(|pw| (pw.clone(), pw.len()))
    .collect();

// Into String (joining)
let csv: String = passwords.iter()
    .map(|pw| pw.as_str())
    .collect::<Vec<_>>()
    .join(", ");
// Or simpler:
let csv = passwords.join(", ");
```

**Cheat sheet**:
| Want | Use |
|------|-----|
| Transform each item | `.map()` |
| Keep matching items | `.filter()` |
| Transform + filter | `.filter_map()` |
| Flatten nested | `.flat_map()` |
| Accumulate | `.fold()` / `.sum()` / `.count()` |
| Check condition | `.any()` / `.all()` |
| First match | `.find()` |
| Take/skip items | `.take(n)` / `.skip(n)` |
| Process in groups | `.chunks(n)` / `.windows(n)` |"""}
        ]
    },
    # ── ADDITIONAL TESTING: ASSERTIONS & PROPERTY TESTING ──
    {
        "category": "testing",
        "conversations": [
            {"role": "user", "content": "What assertion macros and testing patterns should I know beyond basic assert?"},
            {"role": "assistant", "content": """Rust has a rich testing toolkit. Here are the patterns beyond basic `assert!`:

**Assertion macros**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assertions() {
        // Basic
        assert!(check_strength("P@ss") < 50);

        // Equality with nice diff output
        assert_eq!(classify("hunter2"), Strength::Weak);
        assert_ne!(classify("P@ssw0rd!"), Strength::Weak);

        // Custom failure message
        assert!(
            score > 50,
            "Expected score > 50, got {score} for password '{pw}'"
        );

        // Float comparison (no assert_eq for floats!)
        let entropy = calculate_entropy("test");
        assert!((entropy - 2.0).abs() < 0.01, "entropy was {entropy}");
    }
}
```

**Testing errors and panics**:
```rust
#[test]
fn test_error_returned() {
    let result = load_passwords("nonexistent.txt");
    assert!(result.is_err());

    // Check specific error content
    let err = result.unwrap_err();
    assert!(err.to_string().contains("not found"));
}

#[test]
#[should_panic(expected = "empty password")]
fn test_panics_on_empty() {
    check_password("");  // should panic with "empty password"
}

// Return Result from tests (no unwrap noise)
#[test]
fn test_with_result() -> anyhow::Result<()> {
    let passwords = load_passwords("test_data.txt")?;
    assert_eq!(passwords.len(), 5);
    Ok(())
}
```

**Test fixtures and setup**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Shared test helper
    fn sample_passwords() -> Vec<String> {
        vec![
            "hunter2".into(),
            "P@ssw0rd!123".into(),
            "a".into(),
            "correcthorsebatterystaple".into(),
        ]
    }

    #[test]
    fn test_weak_detection() {
        let passwords = sample_passwords();
        let weak: Vec<_> = passwords.iter()
            .filter(|pw| pw.len() < 8)
            .collect();
        assert_eq!(weak.len(), 2);
    }
}
```

**Temporary files for I/O tests**:
```rust
use std::io::Write;

#[test]
fn test_file_audit() -> anyhow::Result<()> {
    // Create a temp file
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("test_passwords.txt");

    std::fs::write(&path, "hunter2\\nP@ssw0rd!\\n123\\n")?;

    let results = audit_file(path.to_str().unwrap())?;
    assert_eq!(results.len(), 3);
    assert!(results[0].is_weak());
    Ok(())
}
// tempdir is automatically cleaned up when `dir` is dropped
```

**Parameterized tests** (test multiple inputs):
```rust
#[test]
fn test_strength_cases() {
    let cases = vec![
        ("", Strength::Weak),
        ("abc", Strength::Weak),
        ("abcdefgh", Strength::Fair),
        ("P@ssw0rd!123", Strength::Strong),
    ];

    for (password, expected) in cases {
        assert_eq!(
            classify(password), expected,
            "Failed for password: '{password}'"
        );
    }
}
```

**Ignoring slow tests**:
```rust
#[test]
#[ignore]  // skipped by default, run with: cargo test -- --ignored
fn test_against_real_breach_api() {
    // slow integration test
}
```

**Go comparison**:
| Go | Rust |
|----|------|
| `t.Error()` / `t.Fatal()` | `assert!()` / `assert_eq!()` |
| `t.Run("subtest", ...)` | test functions (or loop inside test) |
| `TestMain(m)` | no direct equivalent (use `#[ctor]` crate) |
| `t.Skip()` | `#[ignore]` |
| `go test -run Foo` | `cargo test foo` |
| table-driven tests | vec of cases + loop |
| `testify/assert` | built-in macros are good enough |"""}
        ]
    },
]


def generate_rust_dataset(
    output_path: str,
    num_samples: int = 500,
    system_prompt: str = "",
) -> int:
    """
    Generate synthetic Rust tutor training data.

    Uses the seed examples above and optionally expands them.
    Returns the number of samples written.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    samples = []
    for example in RUST_TUTOR_CONVERSATIONS:
        # Format as chat messages with system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(example["conversations"])

        samples.append({
            "conversations": messages,
            "category": example.get("category", "general"),
        })

    # If we need more samples than seed data, create variations
    if num_samples > len(samples):
        # Duplicate with slight variations (in production, use an LLM API)
        import copy
        original_count = len(samples)
        while len(samples) < num_samples:
            base = copy.deepcopy(samples[len(samples) % original_count])
            samples.append(base)

    # Trim to requested count
    samples = samples[:num_samples]

    # Write JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return len(samples)


if __name__ == "__main__":
    output = "data/processed/rust_tutor_synthetic.jsonl"
    count = generate_rust_dataset(output, num_samples=500)
    print(f"Generated {count} samples → {output}")
