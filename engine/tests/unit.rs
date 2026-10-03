//! Unit tests: quant block layouts (against hand-computed llama.cpp
//! reference values), ChatML rendering, mock streaming plumbing, and a
//! hand-computable micro-model forward pass through a synthetic GGUF.

mod common;

use common::{byte_unicode, f16, q4_0, q4_k, q6_k, q8_0, GgufBuilder};
use tpt_localchat_engine::_dequant_test::{dequant_row, dot_row, f16_to_f32};
use tpt_localchat_engine::{
    ChatEngine, ChatHistory, ChatMessage, EngineError, GenParams, LocalChatEngine, MockChatEngine,
};

// ---------- f16 ----------

#[test]
fn f16_decoder_matches_known_bit_patterns() {
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x8000), -0.0);
    assert_eq!(f16_to_f32(0x3C00), 1.0);
    assert_eq!(f16_to_f32(0x4000), 2.0);
    assert_eq!(f16_to_f32(0x3800), 0.5);
    assert_eq!(f16_to_f32(0xC000), -2.0);
    assert_eq!(f16_to_f32(0x4900), 10.0);
    // Denormal: smallest positive subnormal 2^-24.
    assert!((f16_to_f32(0x0001) - 5.96e-8).abs() < 1e-10);
}

#[test]
fn f16_builder_and_decoder_agree() {
    for value in [0.25, 0.5, 1.0, 2.0, -2.0, 3.14159, 100.0] {
        let decoded = f16_to_f32(f16(value));
        assert!(
            (decoded - value).abs() < value.abs() * 1e-3 + 1e-6,
            "f16 round trip failed for {value}: {decoded}"
        );
    }
}

// ---------- quant block layouts ----------

#[test]
fn q4_0_layout_matches_llama_cpp_reference() {
    // w[j] = d * (nibble_j - 8); nibbles 0..15 (lows) are weights 0..15,
    // nibbles 16..31 (highs) are weights 16..31 (llama.cpp halves layout).
    let nibbles: Vec<u8> = (0..32).collect();
    let bytes = q4_0(2.0, &nibbles);
    let row = dequant_row(2, &bytes, 32);
    for (j, &w) in row.iter().enumerate() {
        let q = (nibbles[j] & 0xF) as f32;
        assert_eq!(w, 2.0 * (q - 8.0), "q4_0[{j}]");
    }
}

#[test]
fn q8_0_layout_matches_llama_cpp_reference() {
    let qs: Vec<i32> = (-16..16).collect();
    let bytes = q8_0(0.5, &qs);
    let row = dequant_row(8, &bytes, 32);
    for (j, &w) in row.iter().enumerate() {
        assert_eq!(w, 0.5 * qs[j] as f32, "q8_0[{j}]");
    }
}

#[test]
fn q4_k_layout_matches_llama_cpp_reference() {
    // Scales are 6-bit pairs unpacked by llama.cpp's get_scale_min_k4:
    //   j < 4: d = scales[j] & 63, m = scales[j+4] & 63
    // Sub-block 0 (weights 0..32):  sc=5, m=9
    // Sub-block 1 (weights 32..64): sc=3, m=2
    // Sub-block 2 (weights 128..160): sc=1, m=0
    // Builder packs byte k = (nib[2k] & 0xF) | ((nib[2k+1] & 0xF) << 4);
    // the reference reads q[l]&0xF for weights 0..32 of a chunk and
    // q[l]>>4 for weights 32..64 — i.e. nibbles[2l] / nibbles[2l+1].
    let mut scales = [0u8; 12];
    scales[0] = 5;
    scales[4] = 9;
    scales[1] = 3;
    scales[5] = 2;
    scales[2] = 1;
    scales[6] = 0;

    let nibbles: Vec<u8> = (0..256).map(|i| (i * 7 % 16) as u8).collect();
    let bytes = q4_k(1.5, 0.5, scales, &nibbles);
    let row = dequant_row(15, &bytes, 256);

    let low = |byte_idx: usize| (nibbles[byte_idx] & 0xF) as f32;
    for l in 0..32 {
        let expected = 1.5 * 5.0 * low(2 * l) - 0.5 * 9.0;
        assert_eq!(row[l], expected, "q4_k weight {l} (sub-block 0, low)");
        let expected = 1.5 * 3.0 * low(2 * l + 1) - 0.5 * 2.0;
        assert_eq!(row[32 + l], expected, "q4_k weight {} (sub-block 1, high)", 32 + l);
    }
    // Chunk 1 (weights 64..192) uses sub-blocks 2/3 via the direct
    // scales[j] read (j < 4).
    for l in 0..32 {
        let expected = 1.5 * 1.0 * low(64 + 2 * l) - 0.5 * 0.0;
        assert_eq!(row[64 + l], expected, "q4_k weight {} (sub-block 2, low)", 64 + l);
    }
    // Chunk 2 (weights 128..) uses sub-block 4 via the k4 nibble packing:
    // get_scale_min_k4(4) reads d = (scales[0]&0xF) | ((scales[1]&0xF)<<4)
    // = 5 | (3<<4) = 53, m = 0.
    for l in 0..32 {
        let expected = 1.5 * 53.0 * low(128 + 2 * l);
        assert_eq!(row[128 + l], expected, "q4_k weight {} (sub-block 4, k4-packed)", 128 + l);
    }
}

#[test]
fn q6_k_layout_matches_llama_cpp_reference() {
    let six: Vec<u8> = (0..256).map(|i| (i * 5 % 64) as u8).collect();
    let scales: [i8; 16] = [3, -1, 2, 0, 5, 1, -2, 4, 0, 0, 0, 0, 0, 0, 0, 0];
    let bytes = q6_k(0.25, &six, scales);
    let row = dequant_row(18, &bytes, 256);

    for l in 0..32 {
        let decode = |i: usize| (six[i] as i32) - 32;
        let expected = |chunk: usize, offset: usize, scale_idx: usize| -> f32 {
            0.25 * scales[scale_idx] as f32 * decode(chunk * 128 + offset) as f32
        };
        assert_eq!(row[l], expected(0, l, l / 16), "q6_k[{l}]");
        assert_eq!(row[32 + l], expected(0, 32 + l, l / 16 + 2), "q6_k[{}]", 32 + l);
        assert_eq!(row[64 + l], expected(0, 64 + l, l / 16 + 4), "q6_k[{}]", 64 + l);
        assert_eq!(row[96 + l], expected(0, 96 + l, l / 16 + 6), "q6_k[{}]", 96 + l);
    }
}

#[test]
fn dot_row_matches_dequant_dot_product() {
    let nibbles: Vec<u8> = (0..32).collect();
    let y: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();

    let q4_bytes = q4_0(2.0, &nibbles);
    let q4_row = dequant_row(2, &q4_bytes, 32);
    let fused = dot_row(2, &q4_bytes, &y, 32);
    let reference: f32 = q4_row.iter().zip(y.iter()).map(|(w, &v)| w * v).sum();
    assert!((fused - reference).abs() < 1e-4, "q4_0 dot {fused} vs {reference}");

    let qs: Vec<i32> = (-16..16).collect();
    let q8_bytes = q8_0(0.5, &qs);
    let q8_row = dequant_row(8, &q8_bytes, 32);
    let fused = dot_row(8, &q8_bytes, &y, 32);
    let reference: f32 = q8_row.iter().zip(y.iter()).map(|(w, &v)| w * v).sum();
    assert!((fused - reference).abs() < 1e-4, "q8_0 dot {fused} vs {reference}");

    let mut scales = [0u8; 12];
    scales[0] = 5;
    scales[4] = 9;
    let nibbles256: Vec<u8> = (0..256).map(|i| (i * 7 % 16) as u8).collect();
    let q4k_bytes = q4_k(1.5, 0.5, scales, &nibbles256);
    let q4k_row = dequant_row(15, &q4k_bytes, 256);
    let y256: Vec<f32> = (0..256).map(|i| ((i % 13) as f32) - 6.0).collect();
    let fused = dot_row(15, &q4k_bytes, &y256, 256);
    let reference: f32 = q4k_row.iter().zip(y256.iter()).map(|(w, &v)| w * v).sum();
    assert!((fused - reference).abs() < 1e-3, "q4_k dot {fused} vs {reference}");
}

// ---------- synthetic micro model ----------

/// Builds a 1-layer, 32-embd, 2-head/1-kv-head micro LLaMA whose output is
/// hand-computable: every weight is zero except the norms (1.0) and
/// one-hot token_embd rows. With zero attention/FFN weights the residuals
/// vanish, so the final hidden state is rmsnorm(one-hot) and the tied LM
/// head makes the argmax the smallest vocab id v with v % 32 == token % 32.
///
/// The vocab is a synthetic byte-level tokenizer: 3 special tokens then all
/// 256 byte-unicode chars, no merges — every prompt byte round-trips.
fn micro_model_gguf() -> Vec<u8> {
    let n_embd = 32usize;
    let n_specials = 3usize;
    let n_vocab = n_specials + 256;
    let zeros_q8 = q8_0(0.0, &[0i32; 32]); // one 32-elem zero block
    let ones_f32: Vec<u8> = [1.0f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .cycle()
        .take(n_embd * 4)
        .collect();

    let mut embd = Vec::new();
    for v in 0..n_vocab {
        let mut qs = [0i32; 32];
        qs[v % 32] = 1;
        embd.extend_from_slice(&q8_0(1.0, &qs));
    }

    // Synthetic tokenizer: specials + every byte-unicode char as its own
    // token; no merges (all single-char BPE).
    let byte_chars = byte_unicode();
    let mut tokens: Vec<String> = vec![
        "<|endoftext|>".to_string(),
        "<|im_start|>".to_string(),
        "<|im_end|>".to_string(),
    ];
    tokens.extend(byte_chars.iter().map(|c| c.to_string()));
    let types: Vec<i32> = std::iter::repeat(3) // CONTROL
        .take(n_specials)
        .chain(std::iter::repeat(6).take(256)) // BYTE
        .collect();
    let merges: Vec<String> = Vec::new();

    GgufBuilder::new()
        .str_kv("general.architecture", "llama")
        .u32_kv("llama.block_count", 1)
        .u32_kv("llama.embedding_length", n_embd as u32)
        .u32_kv("llama.feed_forward_length", 32)
        .u32_kv("llama.attention.head_count", 2)
        .u32_kv("llama.attention.head_count_kv", 1)
        .u32_kv("llama.rope.dimension_count", 16)
        .f32_kv("llama.rope.freq_base", 10000.0)
        .f32_kv("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .u32_kv("llama.context_length", 2048)
        .u32_kv("llama.vocab_size", n_vocab as u32)
        .u32_kv("tokenizer.ggml.eos_token_id", 2)
        .u32_kv("tokenizer.ggml.bos_token_id", 1)
        .str_array_kv("tokenizer.ggml.tokens", &tokens)
        .i32_array_kv("tokenizer.ggml.token_type", &types)
        .str_array_kv("tokenizer.ggml.merges", &merges)
        .tensor("token_embd.weight", &[n_embd as u64, n_vocab as u64], 8, &embd)
        .tensor("output_norm.weight", &[n_embd as u64], 0, &ones_f32)
        .tensor("blk.0.attn_norm.weight", &[n_embd as u64], 0, &ones_f32)
        .tensor(
            "blk.0.attn_q.weight",
            &[n_embd as u64, n_embd as u64],
            8,
            &zeros_q8.repeat(n_embd),
        )
        .tensor(
            "blk.0.attn_k.weight",
            &[n_embd as u64, 16],
            8,
            &zeros_q8.repeat(16),
        )
        .tensor(
            "blk.0.attn_v.weight",
            &[n_embd as u64, 16],
            8,
            &zeros_q8.repeat(16),
        )
        .tensor(
            "blk.0.attn_output.weight",
            &[n_embd as u64, n_embd as u64],
            8,
            &zeros_q8.repeat(n_embd),
        )
        .tensor("blk.0.ffn_norm.weight", &[n_embd as u64], 0, &ones_f32)
        .tensor(
            "blk.0.ffn_gate.weight",
            &[n_embd as u64, 32],
            8,
            &zeros_q8.repeat(32),
        )
        .tensor(
            "blk.0.ffn_up.weight",
            &[n_embd as u64, 32],
            8,
            &zeros_q8.repeat(32),
        )
        .tensor(
            "blk.0.ffn_down.weight",
            &[32, n_embd as u64],
            8,
            &zeros_q8.repeat(n_embd),
        )
        .build()
}

#[test]
fn micro_model_forward_runs_and_terminates() {
    let bytes: &'static [u8] = Box::leak(micro_model_gguf().into_boxed_slice());
    let engine = LocalChatEngine::load(bytes).expect("micro model loads");

    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("hi"));
    let mut events = 0;
    let text = engine
        .chat(
            &history,
            &GenParams {
                max_tokens: 3,
                temperature: 0.0, // greedy
                ..GenParams::default()
            },
            &mut |_| {
                events += 1;
                Ok(())
            },
        )
        .expect("micro chat runs");
    assert!(events >= 1);
    // Terminated within the token budget; decoded text is gibberish
    // (synthetic vocab) but bounded.
    assert!(text.chars().count() <= 12);
}

#[test]
fn micro_model_greedy_argmax_is_deterministic() {
    // The hand-computable property: with one-hot embeddings and identity
    // norms, greedy sampling must always produce tokens whose id % 32
    // equals the last prompt token's id % 32. We assert determinism and a
    // bounded sequence here (full argmax identity needs tokenizer access —
    // covered end-to-end by the real-model fixture test).
    let bytes: &'static [u8] = Box::leak(micro_model_gguf().into_boxed_slice());
    let engine = LocalChatEngine::load(bytes).expect("micro model loads");
    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("hi"));

    let mut runs: Vec<String> = Vec::new();
    for seed in 0..3 {
        let _ = seed;
        let text = engine
            .chat(
                &history,
                &GenParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    ..GenParams::default()
                },
                &mut |_| Ok(()),
            )
            .expect("greedy run");
        runs.push(text);
    }
    eprintln!("runs: {runs:?}");
    assert_eq!(runs[0], runs[1]);
    assert_eq!(runs[1], runs[2]);
}

#[test]
fn micro_model_context_overflow_is_reported() {
    let bytes: &'static [u8] = Box::leak(micro_model_gguf().into_boxed_slice());
    let engine = LocalChatEngine::load(bytes).expect("micro model loads");
    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("hi"));
    let err = engine
        .chat(
            &history,
            &GenParams {
                max_tokens: 10_000,
                ..GenParams::default()
            },
            &mut |_| Ok(()),
        )
        .expect_err("must overflow");
    assert!(matches!(err, EngineError::ContextOverflow { .. }));
}

#[test]
fn bad_magic_is_rejected() {
    let bytes = b"NOPEthis is not a gguf file___________";
    assert!(matches!(
        LocalChatEngine::load(Box::leak(bytes.to_vec().into_boxed_slice())),
        Err(EngineError::BadMagic)
    ));
}

// ---------- mock plumbing ----------

#[test]
fn mock_streams_words_and_reports_done() {
    let engine = MockChatEngine::new();
    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("weather?"));

    let mut pieces: Vec<(String, bool)> = Vec::new();
    let text = engine
        .chat(&history, &GenParams::default(), &mut |event| {
            pieces.push((event.token.clone(), event.done));
            Ok(())
        })
        .expect("mock chat");
    assert!(text.contains("weather?"));
    assert!(text.starts_with("[mock]"));
    assert_eq!(pieces.last().map(|p| p.1), Some(true));
    assert_eq!(pieces.iter().filter(|p| p.1).count(), 1);
}

#[test]
fn mock_honors_max_tokens_and_stop_sequences() {
    let engine = MockChatEngine::new();
    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("abc"));

    let text = engine
        .chat(
            &history,
            &GenParams {
                max_tokens: 2,
                ..GenParams::default()
            },
            &mut |_| Ok(()),
        )
        .expect("capped chat");
    assert_eq!(text.split_whitespace().count(), 2);

    let mut saw_done = false;
    engine
        .chat(
            &history,
            &GenParams {
                stop_sequences: vec!["said".to_string()],
                ..GenParams::default()
            },
            &mut |event| {
                if event.done {
                    saw_done = true;
                }
                Ok(())
            },
        )
        .expect("stop-seq chat");
    assert!(saw_done);
}

// ---------- ChatML rendering ----------

#[test]
fn chatml_renders_system_then_turns_then_header() {
    let engine = MockChatEngine::new();
    let mut history = ChatHistory::new();
    history.push(ChatMessage::system("Be terse."));
    history.push(ChatMessage::user("Hello"));
    history.push(ChatMessage::assistant("Hi!"));
    history.push(ChatMessage::user("Bye"));

    let prompt = engine.render_prompt(&history);
    assert!(prompt.starts_with("<|im_start|>system\nBe terse.<|im_end|>\n"));
    assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>\n"));
    assert!(prompt.contains("<|im_start|>assistant\nHi!<|im_end|>\n"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
    assert!(!prompt.contains("trained by Hugging Face"));
}

#[test]
fn chatml_inserts_default_system_when_absent() {
    let engine = MockChatEngine::new();
    let mut history = ChatHistory::new();
    history.push(ChatMessage::user("Hello"));
    let prompt = engine.render_prompt(&history);
    assert!(prompt.starts_with("<|im_start|>system\n"));
    assert!(prompt.contains(tpt_localchat_engine::DEFAULT_SYSTEM_PROMPT));
}
