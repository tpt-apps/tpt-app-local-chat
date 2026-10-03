//! GPT-2-style byte-level BPE built from the GGUF's own tokenizer metadata
//! (`tokenizer.ggml.tokens` / `.merges` / `.token_type`), so the model file
//! is the only artifact the engine ever needs.
//!
//! The GPT-2 pre-tokenization regex is hand-rolled as a scanner (std-only;
//! no regex crate): contractions, ` ?letters`, ` ?digits`, ` ?punct`,
//! trailing-whitespace runs — same segmentation, same order.

use std::collections::HashMap;

use crate::error::EngineError;
use crate::gguf::GgufFile;

/// GGUF token_type values (subset).
const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_BYTE: i32 = 6;

pub struct Tokenizer {
    /// id → token string (in GPT-2 byte-unicode space for normal tokens).
    vocab: Vec<String>,
    id_of: HashMap<String, u32>,
    types: Vec<i32>,
    /// BPE merge ranks: (left, right) → rank.
    ranks: HashMap<(String, String), u32>,
    /// UTF-8 byte → unicode char used by the byte-level mapping.
    byte_encoder: Vec<char>,
    /// Unicode char → original byte.
    byte_decoder: HashMap<char, u8>,
    /// Control/user-defined tokens treated as atomic during encoding,
    /// longest first.
    specials: Vec<(String, u32)>,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl Tokenizer {
    pub fn from_gguf(file: &GgufFile<'_>) -> Result<Self, EngineError> {
        let tokens = file
            .meta_strings("tokenizer.ggml.tokens")
            .ok_or(EngineError::BadTokenizer("missing tokenizer.ggml.tokens"))?;
        let types = file
            .meta_i32s("tokenizer.ggml.token_type")
            .ok_or(EngineError::BadTokenizer("missing tokenizer.ggml.token_type"))?;
        let merges = file
            .meta_strings("tokenizer.ggml.merges")
            .ok_or(EngineError::BadTokenizer("missing tokenizer.ggml.merges"))?;
        if tokens.len() != types.len() {
            return Err(EngineError::BadTokenizer("tokens/token_type length mismatch"));
        }

        let mut id_of = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            id_of.insert(token.clone(), id as u32);
        }

        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            // Merge strings are "left right" with a single ASCII space
            // separator (both sides are byte-unicode chars, never spaces).
            let Some(sep) = merge.find(' ') else {
                continue;
            };
            let left = &merge[..sep];
            let right = &merge[sep + 1..];
            ranks.insert((left.to_string(), right.to_string()), rank as u32);
        }

        let (byte_encoder, byte_decoder) = byte_level_maps();

        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| {
                let t = types[*id];
                t == TYPE_CONTROL || t == TYPE_USER_DEFINED
            })
            .map(|(id, tok)| (tok.clone(), id as u32))
            .filter(|(tok, _)| tok.chars().count() >= 2)
            .collect();
        specials.sort_by_key(|(tok, _)| std::cmp::Reverse(tok.chars().count()));

        let eos_token_id = file.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);
        let bos_token_id = file.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);

        Ok(Self {
            vocab: tokens.to_vec(),
            id_of,
            types: types.to_vec(),
            ranks,
            byte_encoder,
            byte_decoder,
            specials,
            eos_token_id,
            bos_token_id,
        })
    }

    /// Encodes text, honoring special tokens embedded in the string (the
    /// same behavior as llama.cpp's `parse_special = true`).
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (special, id) in &self.specials {
                if rest.starts_with(special.as_str()) {
                    ids.push(*id);
                    rest = &rest[special.len()..];
                    continue 'outer;
                }
            }
            // Longest run with no special-token prefix, BPE-encoded as one
            // segment (pre-tokenization splits it further).
            let mut end = rest.len();
            for (i, _) in rest.char_indices().skip(1) {
                if self.specials.iter().any(|(s, _)| rest[i..].starts_with(s.as_str())) {
                    end = i;
                    break;
                }
            }
            let segment = &rest[..end];
            if !segment.is_empty() {
                self.encode_segment(segment, &mut ids);
            }
            rest = &rest[end..];
        }
        Ok(ids)
    }

    /// Splits a plain-text segment into GPT-2 pre-tokens and BPE-encodes
    /// each into vocab ids.
    fn encode_segment(&self, text: &str, ids: &mut Vec<u32>) {
        for word in pre_tokenize(text) {
            let mapped: String = word
                .as_bytes()
                .iter()
                .map(|&b| self.byte_encoder[b as usize])
                .collect();
            for piece in self.bpe(&mapped) {
                let id = self
                    .id_of
                    .get(&piece)
                    .copied()
                    .or_else(|| self.id_of.get("\u{FFFD}").copied())
                    .unwrap_or(0);
                ids.push(id);
            }
        }
    }

    /// Classic rank-greedy BPE over one mapped word.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&rank) = self.ranks.get(&(
                    parts[i].as_str().to_string(),
                    parts[i + 1].as_str().to_string(),
                )) {
                    if best.map_or(true, |(best_rank, _)| rank < best_rank) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..=i + 1, [merged]);
        }
        parts
    }

    /// True when the id is a special/control token (never BPE-decoded).
    pub fn is_special(&self, id: u32) -> bool {
        self.types
            .get(id as usize)
            .map(|&t| t == TYPE_CONTROL || t == TYPE_USER_DEFINED || t == TYPE_UNKNOWN)
            .unwrap_or(true)
    }

    /// The token's literal string (control tokens) — used by the full-text
    /// decoder for specials.
    pub fn token_str(&self, id: u32) -> &str {
        self.vocab.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }

    /// Decodes ids to bytes: control tokens as their literal text, normal
    /// tokens through the byte-level reverse map.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend_from_slice(&self.token_bytes(id));
        }
        out
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(token) = self.vocab.get(id as usize) else {
            return Vec::new();
        };
        if self.is_special(id) {
            return token.as_bytes().to_vec();
        }
        token
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect()
    }

    /// Incremental decoder for streaming: buffers incomplete UTF-8 tails so
    /// multi-byte characters split across tokens never render as �.
    pub fn streaming_decoder(&self) -> StreamingDecoder<'_> {
        StreamingDecoder {
            tokenizer: self,
            pending: Vec::new(),
        }
    }
}

/// Buffers decoded bytes and only emits complete UTF-8.
pub struct StreamingDecoder<'a> {
    tokenizer: &'a Tokenizer,
    pending: Vec<u8>,
}

impl<'a> StreamingDecoder<'a> {
    /// Decodes one token; returns newly-completed text (possibly empty).
    pub fn push(&mut self, id: u32) -> String {
        self.pending.extend_from_slice(&self.tokenizer.token_bytes(id));
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let out = text.to_string();
                self.pending.clear();
                out
            }
            Err(e) if e.valid_up_to() > 0 => {
                let split = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..split]).to_string();
                self.pending.drain(..split);
                out
            }
            Err(_) => {
                // Invalid prefix that isn't a truncated sequence (huge token
                // runs of garbage): drop one byte so the stream can recover.
                self.pending.remove(0);
                String::new()
            }
        }
    }

    /// Flushes any buffered tail at end of generation (lossy).
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).to_string();
        self.pending.clear();
        out
    }
}

/// GPT-2 `bytes_to_unicode`: printable ranges map to themselves; the rest
/// shift up into U+0100+ so every byte is one BMP char.
fn byte_level_maps() -> (Vec<char>, HashMap<char, u8>) {
    let mut encoder = vec!['\0'; 256];
    let mut decoder = HashMap::with_capacity(256);
    let mut mapped: Vec<u32> = Vec::with_capacity(256);
    for b in 0..256u32 {
        let printable = (0x21..=0x7E).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        mapped.push(if printable { b } else { u32::MAX });
    }
    let mut n = 0u32;
    for slot in mapped.iter_mut() {
        if *slot == u32::MAX {
            *slot = 256 + n;
            n += 1;
        }
    }
    for b in 0..256usize {
        let c = char::from_u32(mapped[b]).unwrap_or('\u{FFFD}');
        encoder[b] = c;
        decoder.insert(c, b as u8);
    }
    (encoder, decoder)
}

/// Hand-rolled GPT-2 pre-tokenization:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        // Contractions (lowercase only, as in the original pattern).
        if chars[i] == '\'' {
            let rest: String = chars[i..].iter().take(3).collect();
            let mut matched = false;
            for c in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
                if rest.starts_with(c) {
                    words.push(c.to_string());
                    i += c.chars().count();
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
        }

        let start = i;
        // ` ?X+` groups: one optional leading space.
        let mut j = i;
        if chars[j] == ' ' {
            j += 1;
        }
        if j < chars.len() {
            let class = char_class(chars[j]);
            if class != 0 {
                let mut k = j;
                while k < chars.len() && char_class(chars[k]) == class {
                    k += 1;
                }
                words.push(chars[i..k].iter().collect());
                i = k;
                continue;
            }
        }

        // Whitespace groups. `\s+(?!\S)` matches all but the final space of
        // a run when that run is followed by a word start — i.e. a run of
        // length n before a non-space keeps n-1 chars; a run at end-of-text
        // (or before more-of-the-run) keeps everything.
        if chars[i].is_whitespace() {
            let mut k = i;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            let followed_by_nonspace = k < chars.len();
            let take = if followed_by_nonspace && k - i > 1 && chars[k - 1] == ' ' {
                k - i - 1
            } else if followed_by_nonspace && k - i == 1 {
                // Single space before a word is consumed by the ` ?X+` arm —
                // reaching here means the next char is punctuation/EOF-ish;
                // the regex's `\s+(?!\S)` fails for one space, and `\s+`
                // also fails, so the space joins the NEXT match instead.
                // Fall through: emit it as its own word only if it is not a
                // plain ' ' (which the ` ?` arms above would have eaten when
                // followed by a class-0 char... it can't be: class-0 chars
                // are whitespace). This arm is unreachable for ' '.
                1
            } else {
                k - i
            };
            if take > 0 {
                words.push(chars[i..i + take].iter().collect());
            }
            i += take.max(1);
            continue;
        }

        if i == start {
            // Unknown single char (shouldn't happen): emit to guarantee
            // progress.
            words.push(chars[i].to_string());
            i += 1;
        }
    }
    words
}

/// 1 = letter, 2 = number, 3 = other non-space, 0 = whitespace.
fn char_class(c: char) -> u8 {
    if c.is_whitespace() {
        0
    } else if c.is_alphabetic() {
        1
    } else if c.is_numeric() {
        2
    } else {
        3
    }
}
