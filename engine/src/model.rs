//! LLaMA-architecture transformer over GGUF tensors: RMSNorm, RoPE, GQA
//! attention with a persistent f32 KV cache, SwiGLU FFN, tied embeddings.
//!
//! The KV cache survives across `chat` calls: the engine finds the longest
//! common token prefix with the cached sequence and only prefills the
//! delta, so turn 2+ of a conversation doesn't reprocess the whole history
//! (that reuse is what makes an in-browser model tolerable).
//!
//! The hot-path kernels are free functions over disjoint fields — Rust's
//! borrow checker otherwise conflicts `&self` methods with the per-step
//! `&mut` scratch buffers.

use std::collections::HashMap;

use crate::dequant;
use crate::error::EngineError;
use crate::gguf::{GgufFile, Tensor};

pub struct Config {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub rope_base: f32,
    pub rope_dims: usize,
    pub rms_eps: f32,
    /// Logical max sequence length the cache may hold.
    pub max_seq: usize,
}

/// One transformer block's tensor indices into the file.
struct Block {
    attn_norm: usize,
    attn_q: usize,
    attn_k: usize,
    attn_v: usize,
    attn_output: usize,
    ffn_norm: usize,
    ffn_gate: usize,
    ffn_up: usize,
    ffn_down: usize,
}

pub struct Model<'a> {
    file: GgufFile<'a>,
    cfg: Config,
    inv_freq: Vec<f32>,
    embd_idx: usize,
    output_norm_idx: usize,
    blocks: Vec<Block>,

    /// f32 KV cache: `[layer][k_or_v][kv_head][slot][head_dim]`, flattened.
    cache: Vec<f32>,
    /// Tokens currently represented in the cache.
    cache_len: usize,

    // Reusable scratch (kept alive across calls to avoid reallocating).
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    norm_w: Vec<f32>,
    attn_out: Vec<f32>,
    up: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
}

impl<'a> Model<'a> {
    pub fn from_gguf(file: GgufFile<'a>) -> Result<Self, EngineError> {
        let arch = file
            .architecture()
            .ok_or_else(|| EngineError::MissingMeta("general.architecture".to_string()))?
            .to_string();
        if arch != "llama" {
            return Err(EngineError::UnsupportedArchitecture(arch));
        }
        let meta_u32 = |key: &str| {
            file.meta_u32(key)
                .ok_or_else(|| EngineError::MissingMeta(key.to_string()))
        };
        let n_layers = meta_u32("llama.block_count")? as usize;
        let n_embd = meta_u32("llama.embedding_length")? as usize;
        let n_ff = meta_u32("llama.feed_forward_length")? as usize;
        let n_heads = meta_u32("llama.attention.head_count")? as usize;
        let n_kv_heads = meta_u32("llama.attention.head_count_kv")? as usize;
        if n_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(EngineError::BadTensorShape { tensor: "attention".into() });
        }
        let head_dim = n_embd / n_heads;
        let rope_dims = file
            .meta_u32("llama.rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let n_vocab_meta = file.meta_u32("llama.vocab_size").map(|v| v as usize);
        let rms_eps = file
            .meta_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1.0e-5);
        let rope_base = file.meta_f32("llama.rope.freq_base").unwrap_or(10000.0);
        let trained_ctx = file.meta_u32("llama.context_length").unwrap_or(2048) as usize;

        // Resolve tensors by name once.
        let mut by_name: HashMap<&str, usize> = HashMap::with_capacity(file.tensors.len());
        for (i, t) in file.tensors.iter().enumerate() {
            by_name.insert(t.name.as_str(), i);
        }
        let get = |name: &str| -> Result<usize, EngineError> {
            by_name
                .get(name)
                .copied()
                .ok_or_else(|| EngineError::MissingMeta(format!("tensor {name}")))
        };

        let embd_idx = get("token_embd.weight")?;
        // Tied embeddings: no separate output.weight (the vocabulary head
        // is token_embd itself).
        let output_norm_idx = get("output_norm.weight")?;
        let mut blocks = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            blocks.push(Block {
                attn_norm: get(&format!("blk.{layer}.attn_norm.weight"))?,
                attn_q: get(&format!("blk.{layer}.attn_q.weight"))?,
                attn_k: get(&format!("blk.{layer}.attn_k.weight"))?,
                attn_v: get(&format!("blk.{layer}.attn_v.weight"))?,
                attn_output: get(&format!("blk.{layer}.attn_output.weight"))?,
                ffn_norm: get(&format!("blk.{layer}.ffn_norm.weight"))?,
                ffn_gate: get(&format!("blk.{layer}.ffn_gate.weight"))?,
                ffn_up: get(&format!("blk.{layer}.ffn_up.weight"))?,
                ffn_down: get(&format!("blk.{layer}.ffn_down.weight"))?,
            });
        }

        let n_vocab = n_vocab_meta.unwrap_or(file.tensors[embd_idx].dims[1] as usize);
        // Trained context is 8192 for SmolLM2; the f32 KV cache at 2048
        // slots costs ~47 MB for this model — well-bounded for a browser.
        let max_seq = trained_ctx.min(2048);
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| rope_base.powf(-(2.0 * i as f32) / rope_dims as f32))
            .collect();

        let cache = vec![0.0; 2 * n_layers * n_kv_heads * max_seq * head_dim];

        Ok(Self {
            file,
            cfg: Config {
                n_layers,
                n_embd,
                n_ff,
                n_heads,
                n_kv_heads,
                head_dim,
                n_vocab,
                rope_base,
                rope_dims,
                rms_eps,
                max_seq,
            },
            inv_freq,
            embd_idx,
            output_norm_idx,
            blocks,
            cache,
            cache_len: 0,
            x: vec![0.0; n_embd],
            xb: vec![0.0; n_embd],
            q: vec![0.0; n_heads * head_dim],
            k: vec![0.0; n_kv_heads * head_dim],
            v: vec![0.0; n_kv_heads * head_dim],
            norm_w: vec![0.0; n_embd],
            attn_out: vec![0.0; n_embd.max(n_ff)],
            up: vec![0.0; n_ff],
            scores: vec![0.0; max_seq],
            logits: vec![0.0; n_vocab],
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    pub fn info(&self) -> String {
        format!(
            "{} layers, {} embd, {} heads ({} kv), vocab {} — GGUF v{}",
            self.cfg.n_layers,
            self.cfg.n_embd,
            self.cfg.n_heads,
            self.cfg.n_kv_heads,
            self.cfg.n_vocab,
            self.file.version
        )
    }

    /// Drops the cache back to `len` tokens (cheap: logical only).
    pub fn truncate_cache(&mut self, len: usize) {
        self.cache_len = len.min(self.cache_len);
    }

    /// Processes one token at the cache's next position, appending K/V to
    /// the cache. Returns the logits when `want_logits` (only the caller's
    /// last prefill token and each generated token ask for them), else an
    /// empty vec.
    pub fn forward(&mut self, token: u32, want_logits: bool) -> Result<Vec<f32>, EngineError> {
        let pos = self.cache_len;
        let cfg = &self.cfg;
        if pos >= cfg.max_seq {
            return Err(EngineError::ContextOverflow {
                needed: pos + 1,
                cap: cfg.max_seq,
            });
        }
        // Embedding lookup (dequant one row).
        {
            let t = &self.file.tensors[self.embd_idx];
            if token as usize >= t.dims[1] as usize {
                return Err(EngineError::BadTokenizer("token id out of range"));
            }
            let row = embd_row_bytes(&self.file, t, token as usize, cfg.n_embd);
            dequant::dequant_row_into(t.quant, row, cfg.n_embd, &mut self.x);
        }

        for (li, block) in self.blocks.iter().enumerate() {
            // ---- attention ----
            self.xb.copy_from_slice(&self.x);
            copy_norm_weights(&self.file, &self.file.tensors[block.attn_norm], &mut self.norm_w);
            rmsnorm(&mut self.xb, &self.norm_w, cfg.rms_eps);

            let layer_kv_base = li * 2 * cfg.n_kv_heads * cfg.max_seq * cfg.head_dim;

            {
                let (file, tensors) = (&self.file, &self.file.tensors);
                let xb = self.xb.clone();
                matvec(file, &tensors[block.attn_q], &xb, &mut self.q);
                matvec(file, &tensors[block.attn_k], &xb, &mut self.k);
                matvec(file, &tensors[block.attn_v], &xb, &mut self.v);
            }
            rope(&mut self.q, pos, &self.inv_freq);
            rope(&mut self.k, pos, &self.inv_freq);

            // Write K/V into the cache ([layer][head][slot][dim] layout —
            // each kv head's 64 dims go to its own contiguous slot range).
            for kvh in 0..cfg.n_kv_heads {
                let k_dst = layer_kv_base + kvh * cfg.max_seq * cfg.head_dim + pos * cfg.head_dim;
                let v_dst = layer_kv_base
                    + cfg.n_kv_heads * cfg.max_seq * cfg.head_dim
                    + kvh * cfg.max_seq * cfg.head_dim
                    + pos * cfg.head_dim;
                self.cache[k_dst..k_dst + cfg.head_dim]
                    .copy_from_slice(&self.k[kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim]);
                self.cache[v_dst..v_dst + cfg.head_dim]
                    .copy_from_slice(&self.v[kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim]);
            }

            // GQA attention: query head h uses kv head h / group.
            let group = cfg.n_heads / cfg.n_kv_heads;
            let scale = 1.0 / (cfg.head_dim as f32).sqrt();
            for h in 0..cfg.n_heads {
                let kvh = h / group;
                let q_head = &self.q[h * cfg.head_dim..(h + 1) * cfg.head_dim];
                let kv_head_base = layer_kv_base + kvh * cfg.max_seq * cfg.head_dim;
                for s in 0..=pos {
                    let k_head = &self.cache[kv_head_base + s * cfg.head_dim..][..cfg.head_dim];
                    let mut dot = 0.0f32;
                    for i in 0..cfg.head_dim {
                        dot += q_head[i] * k_head[i];
                    }
                    self.scores[s] = dot * scale;
                }
                softmax(&mut self.scores[..=pos]);

                let mut head_acc = vec![0.0f32; cfg.head_dim];
                for s in 0..=pos {
                    let p = self.scores[s];
                    if p == 0.0 {
                        continue;
                    }
                    let v_head_base = layer_kv_base
                        + cfg.n_kv_heads * cfg.max_seq * cfg.head_dim
                        + kvh * cfg.max_seq * cfg.head_dim
                        + s * cfg.head_dim;
                    let v_head = &self.cache[v_head_base..v_head_base + cfg.head_dim];
                    for i in 0..cfg.head_dim {
                        head_acc[i] += p * v_head[i];
                    }
                }
                self.xb[h * cfg.head_dim..(h + 1) * cfg.head_dim].copy_from_slice(&head_acc);
            }

            {
                let xb = self.xb.clone();
                matvec(
                    &self.file,
                    &self.file.tensors[block.attn_output],
                    &xb,
                    &mut self.attn_out[..cfg.n_embd],
                );
            }
            for (x, a) in self.x.iter_mut().zip(self.attn_out.iter()) {
                *x += a;
            }

            // ---- FFN (SwiGLU) ----
            self.xb.copy_from_slice(&self.x);
            copy_norm_weights(&self.file, &self.file.tensors[block.ffn_norm], &mut self.norm_w);
            rmsnorm(&mut self.xb, &self.norm_w, cfg.rms_eps);

            {
                let xb = self.xb.clone();
                matvec(
                    &self.file,
                    &self.file.tensors[block.ffn_gate],
                    &xb,
                    &mut self.attn_out[..cfg.n_ff],
                );
            }
            for g in self.attn_out[..cfg.n_ff].iter_mut() {
                // silu(g) = g * sigmoid(g)
                *g /= 1.0 + (-*g).exp();
            }
            {
                let xb = self.xb.clone();
                matvec(&self.file, &self.file.tensors[block.ffn_up], &xb, &mut self.up);
            }
            for (g, u) in self.attn_out[..cfg.n_ff].iter_mut().zip(self.up.iter_mut()) {
                *g *= *u;
            }
            {
                let gated = self.attn_out[..cfg.n_ff].to_vec();
                matvec(
                    &self.file,
                    &self.file.tensors[block.ffn_down],
                    &gated,
                    &mut self.xb,
                );
            }
            for (x, a) in self.x.iter_mut().zip(self.xb.iter()) {
                *x += a;
            }
        }

        self.cache_len = pos + 1;

        if !want_logits {
            return Ok(Vec::new());
        }

        copy_norm_weights(&self.file, &self.file.tensors[self.output_norm_idx], &mut self.norm_w);
        rmsnorm(&mut self.x, &self.norm_w, cfg.rms_eps);

        // Tied LM head: logits[v] = token_embd row v · x.
        {
            let t = &self.file.tensors[self.embd_idx];
            let (block_elems, block_bytes) = crate::gguf::quant_block(t.quant)
                .expect("quant validated at parse");
            let row_bytes = (cfg.n_embd / block_elems as usize) * block_bytes as usize;
            let data = self
                .file
                .tensor_data(t)
                .expect("tensor bounds validated at parse");
            for (v, slot) in self.logits.iter_mut().enumerate() {
                let row = &data[v * row_bytes..(v + 1) * row_bytes];
                *slot = dequant::dot_row(t.quant, row, &self.x, cfg.n_embd);
            }
        }
        Ok(self.logits.clone())
    }
}

/// The byte slice of one `token_embd` row (validated at parse).
fn embd_row_bytes<'b>(
    file: &'b GgufFile<'b>,
    t: &'b Tensor,
    row: usize,
    n_embd: usize,
) -> &'b [u8] {
    let (block_elems, block_bytes) =
        crate::gguf::quant_block(t.quant).expect("quant validated at parse");
    let row_bytes = (n_embd / block_elems as usize) * block_bytes as usize;
    let data = file.tensor_data(t).expect("tensor bounds validated at parse");
    &data[row * row_bytes..(row + 1) * row_bytes]
}

/// Reads an F32 norm tensor into `out`.
fn copy_norm_weights(file: &GgufFile<'_>, t: &Tensor, out: &mut [f32]) {
    let data = file.tensor_data(t).expect("tensor bounds validated at parse");
    for (i, slot) in out.iter_mut().enumerate() {
        let b = &data[4 * i..4 * i + 4];
        *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
}

/// `x ← rmsnorm(x) * w` in place.
fn rmsnorm(x: &mut [f32], weights: &[f32], eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for (v, &w) in x.iter_mut().zip(weights.iter()) {
        *v *= scale * w;
    }
}

/// RoPE (LLaMA "normal" style: rotate adjacent pairs), in place.
fn rope(x: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let head_dim = inv_freq.len() * 2;
    for base in (0..x.len()).step_by(head_dim) {
        for (i, &inv) in inv_freq.iter().enumerate() {
            let freq = pos as f32 * inv;
            let (sin, cos) = freq.sin_cos();
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            x[base + 2 * i] = x0 * cos - x1 * sin;
            x[base + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }
}

/// matvec: `out[i] = tensor row i · x` for a `[in_dim, out_dim]` tensor.
fn matvec(file: &GgufFile<'_>, t: &Tensor, x: &[f32], out: &mut [f32]) {
    let in_dim = t.dims[0] as usize;
    let out_dim = t.dims[1] as usize;
    let (block_elems, block_bytes) =
        crate::gguf::quant_block(t.quant).expect("quant validated at parse");
    let row_bytes = (in_dim / block_elems as usize) * block_bytes as usize;
    let data = file.tensor_data(t).expect("tensor bounds validated by tensor_byte_len");
    for (i, slot) in out.iter_mut().take(out_dim).enumerate() {
        let row = &data[i * row_bytes..(i + 1) * row_bytes];
        *slot = dequant::dot_row(t.quant, row, x, in_dim);
    }
}

/// In-place softmax over the whole (already-filled) slice.
fn softmax(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}
