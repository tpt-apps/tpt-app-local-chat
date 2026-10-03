//! End-to-end tests against the real SmolLM2-135M-Instruct Q4_0 GGUF
//! (91.7 MB, gitignored at `test-data/`). Parse/tokenize tests run
//! everywhere; generation tests are `#[ignore]`d — run them explicitly in
//! release:
//!
//!   cargo test -p tpt-localchat-engine --release -- --ignored
//!
//! These are the true validation for the quant kernels: a wrong Q4_0/Q8_0
//! dequant turns generation into byte soup, instantly visible.

use tpt_localchat_engine::{
    ChatEngine, ChatHistory, ChatMessage, GenParams, LocalChatEngine,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test-data/SmolLM2-135M-Instruct.Q4_0.gguf"
);

fn load_engine() -> Option<&'static LocalChatEngine> {
    let bytes = std::fs::read(FIXTURE).ok()?;
    let engine = LocalChatEngine::load(Box::leak(bytes.into_boxed_slice())).ok()?;
    Some(Box::leak(Box::new(engine)))
}

#[test]
fn parses_real_gguf_metadata() {
    let Some(engine) = load_engine() else {
        eprintln!("fixture missing — skipping: {FIXTURE}");
        return;
    };
    let info = engine.model_info();
    assert!(info.contains("30 layers"), "{info}");
    assert!(info.contains("vocab 49152"), "{info}");
    assert!(info.contains("GGUF v3"), "{info}");
}

#[test]
fn tokenizer_round_trips_english_and_specials() {
    let Some(engine) = load_engine() else {
        eprintln!("fixture missing — skipping");
        return;
    };
    // Round-trip through prompt rendering + token count exercises the BPE
    // and special-token scan on real vocab data.
    let mut history = ChatHistory::new();
    history.push(ChatMessage::system("Be terse."));
    history.push(ChatMessage::user("What is 2 + 2?"));
    let count = engine.prompt_token_count(&history).expect("tokenize");
    // ChatML header + short question: well under 100 tokens.
    assert!(count > 10 && count < 100, "token count {count}");
}

/// The coherence gate: greedy decoding must produce readable English with
/// the right answer. Debug builds are ~30x slower — run in release.
#[test]
#[ignore]
fn greedy_generation_answers_arithmetic() {
    let Some(engine) = load_engine() else {
        eprintln!("fixture missing — skipping");
        return;
    };
    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("What is 2 + 2? Answer with just the number."));

    let started = std::time::Instant::now();
    let mut pieces = 0usize;
    let text = engine
        .chat(
            &history,
            &GenParams {
                max_tokens: 24,
                temperature: 0.0, // greedy: deterministic coherence check
                ..GenParams::default()
            },
            &mut |_| {
                pieces += 1;
                Ok(())
            },
        )
        .expect("generation runs");
    let elapsed = started.elapsed();
    println!("generated {pieces} pieces in {elapsed:.1?}:");
    println!("  {text}");

    assert!(pieces > 1, "streamed {pieces} pieces");
    assert!(text.contains('4'), "answer should contain \"4\": {text:?}");
    // Readability gate: mostly ASCII, mostly letters/digits/spaces.
    let printable = text
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_ascii_punctuation() || c.is_whitespace())
        .count();
    assert!(
        printable as f32 >= 0.9 * text.chars().count() as f32,
        "output looks like byte soup: {text:?}"
    );
}

/// Greedy output must be stable, and turn 2 must reuse the cached prefix
/// Turn 2 via the KV-cache prefix-reuse path must produce EXACTLY what a
/// fresh engine produces on the same history (the reuse is an optimization,
/// never a behavior change), and stay fluent English.
#[test]
#[ignore]
fn conversation_second_turn_reuses_cache() {
    let Some(engine) = load_engine() else {
        eprintln!("fixture missing — skipping");
        return;
    };
    let params = GenParams {
        max_tokens: 24,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenParams::default()
    };

    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("What is 2 + 2?"));
    let first = engine.chat(&history, &params, &mut |_| Ok(())).expect("turn 1");
    println!("turn 1: {first:?}");
    assert!(first.contains('4'), "turn 1 should contain a \"4\": {first:?}");

    history.push(ChatMessage::assistant(first.clone()));
    history.push(ChatMessage::user("What is 3 + 3?"));
    let reused = engine.chat(&history, &params, &mut |_| Ok(())).expect("turn 2 (reused cache)");
    println!("turn 2 (reused): {reused:?}");

    engine.reset_cache();
    let fresh = engine.chat(&history, &params, &mut |_| Ok(())).expect("turn 2 (fresh)");
    println!("turn 2 (fresh):  {fresh:?}");

    assert_eq!(reused, fresh, "cache prefix reuse must not change output");
    let printable = reused
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_ascii_punctuation() || c.is_whitespace())
        .count();
    assert!(
        printable as f32 >= 0.9 * reused.chars().count() as f32,
        "turn 2 output looks like byte soup: {reused:?}"
    );
}
