//! Shared test helper: builds synthetic GGUF byte buffers so unit tests
//! can exercise the parser, tokenizer, and transformer without the 92 MB
//! real-model fixture.

#![allow(dead_code)]

use std::io::Write;

pub struct GgufBuilder {
    buf: Vec<u8>,
    kv_count: u64,
    tensors: Vec<(String, Vec<u64>, u32)>, // name, dims, quant
    tensor_data: Vec<u8>,
    tensor_count: u64,
}

impl GgufBuilder {
    pub fn new() -> Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count (patched)
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv count (patched)
        Self {
            buf,
            kv_count: 0,
            tensors: Vec::new(),
            tensor_data: Vec::new(),
            tensor_count: 0,
        }
    }

    pub fn u32_kv(mut self, key: &str, value: u32) -> Self {
        self.key(key);
        self.buf.extend_from_slice(&4u32.to_le_bytes());
        self.buf.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    pub fn i32_kv(mut self, key: &str, value: i32) -> Self {
        self.key(key);
        self.buf.extend_from_slice(&5u32.to_le_bytes());
        self.buf.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    pub fn f32_kv(mut self, key: &str, value: f32) -> Self {
        self.key(key);
        self.buf.extend_from_slice(&6u32.to_le_bytes());
        self.buf.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    pub fn str_kv(mut self, key: &str, value: &str) -> Self {
        self.key(key);
        self.buf.extend_from_slice(&8u32.to_le_bytes());
        self.buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(value.as_bytes());
        self.kv_count += 1;
        self
    }

    pub fn str_array_kv(mut self, key: &str, values: &[String]) -> Self {
        self.key(key);
        self.buf.extend_from_slice(&9u32.to_le_bytes()); // array
        self.buf.extend_from_slice(&8u32.to_le_bytes()); // of string
        self.buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            self.buf
                .extend_from_slice(&(v.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(v.as_bytes());
        }
        self.kv_count += 1;
        self
    }

    pub fn i32_array_kv(mut self, key: &str, values: &[i32]) -> Self {
        self.key(key);
        self.buf.extend_from_slice(&9u32.to_le_bytes()); // array
        self.buf.extend_from_slice(&5u32.to_le_bytes()); // of i32
        self.buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
        self
    }

    fn key(&mut self, key: &str) {
        self.buf
            .extend_from_slice(&(key.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(key.as_bytes());
    }

    /// Adds a tensor; `data` must be `tensor_bytes(dims, quant)` long.
    pub fn tensor(mut self, name: &str, dims: &[u64], quant: u32, data: &[u8]) -> Self {
        self.buf
            .extend_from_slice(&(name.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(name.as_bytes());
        self.buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            self.buf.extend_from_slice(&d.to_le_bytes());
        }
        self.buf.extend_from_slice(&quant.to_le_bytes());
        self.buf
            .extend_from_slice(&(self.tensor_data.len() as u64).to_le_bytes());
        self.tensors.push((name.to_string(), dims.to_vec(), quant));
        self.tensor_data.extend_from_slice(data);
        self.tensor_count += 1;
        self
    }

    /// Finalizes: patches the counts and pads the info section to 32 bytes
    /// (the parser aligns the data section off the END OF INFOS, so the pad
    /// must be computed before the tensor data is appended).
    pub fn build(mut self) -> Vec<u8> {
        let pad = (32 - (self.buf.len() as u64 % 32)) % 32;
        self.buf
            .extend_from_slice(&vec![0u8; pad as usize + self.tensor_data.len()]);
        // Overwrite the padding+data region with the real bytes.
        let at = self.buf.len() - self.tensor_data.len();
        self.buf[at..].copy_from_slice(&self.tensor_data);
        // Patch counts.
        self.buf[8..16].copy_from_slice(&self.tensor_count.to_le_bytes());
        self.buf[16..24].copy_from_slice(&self.kv_count.to_le_bytes());
        self.buf
    }
}

/// f16 encoding (round-trip of the engine's decoder).
pub fn f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        0
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u16) << 10) | frac as u16
    }
}

/// Q8_0 block: f16 scale + 32 i8s.
pub fn q8_0(scale: f32, qs: &[i32]) -> Vec<u8> {
    assert_eq!(qs.len(), 32);
    let mut out = Vec::with_capacity(34);
    out.extend_from_slice(&f16(scale).to_le_bytes());
    for &q in qs {
        out.push(q as i8 as u8);
    }
    out
}

/// Q4_0 block: f16 scale + 16 bytes. llama.cpp halves layout: the first 16
/// nibbles are the LOW half (weights 0..15), the next 16 are the HIGH half
/// (weights 16..31).
pub fn q4_0(scale: f32, nibbles: &[u8]) -> Vec<u8> {
    assert_eq!(nibbles.len(), 32);
    let mut out = Vec::with_capacity(18);
    out.extend_from_slice(&f16(scale).to_le_bytes());
    for j in 0..16 {
        out.push((nibbles[j] & 0xF) | ((nibbles[16 + j] & 0xF) << 4));
    }
    out
}

/// Q4_K block: f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes.
pub fn q4_k(d: f32, dmin: f32, scales: [u8; 12], nibbles: &[u8]) -> Vec<u8> {
    assert_eq!(nibbles.len(), 256);
    let mut out = Vec::with_capacity(144);
    out.extend_from_slice(&f16(d).to_le_bytes());
    out.extend_from_slice(&f16(dmin).to_le_bytes());
    out.extend_from_slice(&scales);
    for pair in nibbles.chunks(2) {
        out.push((pair[0] & 0xF) | ((pair[1] & 0xF) << 4));
    }
    out
}

/// Q6_K block (llama.cpp `block_q6_K` layout): ql[128] low nibbles,
/// qh[64] high 2 bits, scales[16], f16 d LAST — built from 256 6-bit codes.
pub fn q6_k(d: f32, six_bits: &[u8], scales: [i8; 16]) -> Vec<u8> {
    assert_eq!(six_bits.len(), 256);
    let mut out = Vec::with_capacity(210);

    // ql[128]: byte l of each 64-byte half packs weight l's low nibble and
    // weight l+64's low nibble; the second half covers weights +128.
    let mut ql = vec![0u8; 128];
    for l in 0..32 {
        ql[l] = (six_bits[l] & 0xF) | ((six_bits[l + 64] & 0xF) << 4);
        ql[l + 32] = (six_bits[l + 32] & 0xF) | ((six_bits[l + 96] & 0xF) << 4);
        ql[64 + l] = (six_bits[128 + l] & 0xF) | ((six_bits[128 + l + 64] & 0xF) << 4);
        ql[64 + l + 32] = (six_bits[128 + l + 32] & 0xF) | ((six_bits[128 + l + 96] & 0xF) << 4);
    }
    out.extend_from_slice(&ql);

    // qh[64]: byte l packs 2 high bits of weights l, l+32, l+64, l+96.
    let mut qh = vec![0u8; 64];
    for l in 0..32 {
        qh[l] = (((six_bits[l] >> 4) & 3) as u8)
            | ((((six_bits[l + 32] >> 4) & 3) as u8) << 2)
            | ((((six_bits[l + 64] >> 4) & 3) as u8) << 4)
            | ((((six_bits[l + 96] >> 4) & 3) as u8) << 6);
        qh[32 + l] = (((six_bits[128 + l] >> 4) & 3) as u8)
            | ((((six_bits[128 + l + 32] >> 4) & 3) as u8) << 2)
            | ((((six_bits[128 + l + 64] >> 4) & 3) as u8) << 4)
            | ((((six_bits[128 + l + 96] >> 4) & 3) as u8) << 6);
    }
    out.extend_from_slice(&qh);

    for &s in &scales {
        out.push(s as u8);
    }
    out.extend_from_slice(&f16(d).to_le_bytes());
    assert_eq!(out.len(), 210);
    out
}

/// GPT-2 bytes_to_unicode map, replicated for test vocab construction.
pub fn byte_unicode() -> Vec<char> {
    let mut out = vec!['\0'; 256];
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
        out[b] = char::from_u32(mapped[b]).unwrap();
    }
    out
}

/// A stdout flush helper for long-running fixture tests.
pub fn flush() {
    let _ = std::io::stdout().flush();
}
