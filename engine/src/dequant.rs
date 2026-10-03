//! Quantized-weight dequantization and fused dequant-dot kernels.
//!
//! Block layouts match llama.cpp exactly (validated against the reference
//! algorithms in `tests/unit.rs`): F32, F16, Q8_0, Q4_0, Q4_K, Q6_K.
//! `dequant_row` materializes a row as f32 (tests, small tensors); the
//! `dot_*` kernels accumulate straight from the quantized bytes — no f32
//! copy of the weights ever exists, which keeps a 135M model at ~92 MB of
//! RAM instead of ~540 MB.

use crate::gguf::quant_block;

pub const QK8_0: usize = 32;
pub const QK4_0: usize = 32;
pub const QK_K: usize = 256;

/// f16 bits → f32 (std-only; the `half` crate would work but this is all
/// the engine needs).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let frac = u32::from(bits & 0x03FF);
    match exp {
        0 => {
            // Denormal: scale up the subnormal mantissa.
            if frac == 0 {
                f32::from_bits(sign)
            } else {
                let mut e: i32 = -1;
                let mut f = frac;
                while f & 0x0400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x03FF;
                // Denormal: value = frac·2^-24; after normalizing to 1.f
                // with `shifts` left shifts, the exponent is -14 - shifts
                // (== 114 + e in the loop's terms), biased +127.
                f32::from_bits(sign | (((114 + e) as u32) << 23) | (f << 13))
            }
        }
        0x1F => f32::from_bits(sign | 0x7F80_0000 | (frac << 13)),
        _ => f32::from_bits(sign | ((u32::from(exp) + 127 - 15) << 23) | (frac << 13)),
    }
}

fn f16_at(bytes: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

/// Quant type ids (GGUF `ggml_type`), mirrored from gguf.rs for ergonomics.
pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q4_0: u32 = 2;
pub const Q8_0: u32 = 8;
pub const Q4K_S: u32 = 14;
pub const Q4K_M: u32 = 15;
pub const Q6_K: u32 = 18;

/// Materializes one row (length must be a whole number of blocks) as f32.
pub fn dequant_row(quant: u32, bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    dequant_row_into(quant, bytes, n, &mut out);
    out
}

/// Writes one quantized row's f32 values into `out` (length `n`).
pub fn dequant_row_into(quant: u32, bytes: &[u8], n: usize, out: &mut [f32]) {
    match quant {
        F32 => {
            for (i, slot) in out.iter_mut().enumerate().take(n) {
                let b = &bytes[4 * i..4 * i + 4];
                *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        F16 => {
            for (i, slot) in out.iter_mut().enumerate().take(n) {
                *slot = f16_at(bytes, i);
            }
        }
        Q4_0 => {
            // llama.cpp dequantize_row_q4_0: the 16 bytes' LOW nibbles are
            // weights 0..15, the HIGH nibbles are weights 16..31 (halves,
            // NOT interleaved).
            let nblocks = n / QK4_0;
            for b in 0..nblocks {
                let block = &bytes[b * 18..b * 18 + 18];
                let d = f16_at(block, 0);
                for j in 0..16 {
                    let lo = (block[2 + j] & 0xF) as i32 - 8;
                    let hi = (block[2 + j] >> 4) as i32 - 8;
                    out[b * QK4_0 + j] = d * lo as f32;
                    out[b * QK4_0 + 16 + j] = d * hi as f32;
                }
            }
        }
        Q8_0 => {
            let nblocks = n / QK8_0;
            for b in 0..nblocks {
                let block = &bytes[b * 34..b * 34 + 34];
                let d = f16_at(block, 0);
                for j in 0..QK8_0 {
                    out[b * QK8_0 + j] = d * block[2 + j] as i8 as f32;
                }
            }
        }
        Q4K_S | Q4K_M => dequant_row_q4k(bytes, n, out),
        Q6_K => dequant_row_q6k(bytes, n, out),
        _ => unreachable!("unsupported quant reached dequant: {quant}"),
    }
}

/// llama.cpp `get_scale_min_k4`: unpacks the j-th 6-bit scale/min pair from
/// the 12-byte Q4_K scale field.
fn scale_min_k4(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j - 4] & 0xF) | ((scales[j - 3] & 0xF) << 4),
            (scales[j - 4] >> 4) | ((scales[j - 3] >> 4) << 4),
        )
    }
}

fn dequant_row_q4k(bytes: &[u8], n: usize, out: &mut [f32]) {
    let nblocks = n / QK_K;
    for b in 0..nblocks {
        let block = &bytes[b * 144..b * 144 + 144];
        let d = f16_at(block, 0);
        let dmin = f16_at(block, 1);
        let scales = &block[4..16];
        let mut q = &block[16..];
        let mut is = 0usize;
        let mut written = 0usize;
        let mut j = 0usize;
        while j < QK_K {
            let (sc1, m1) = scale_min_k4(scales, is);
            let (sc2, m2) = scale_min_k4(scales, is + 1);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                out[b * QK_K + written + l] = d1 * (q[l] & 0xF) as f32 - m1;
            }
            for l in 0..32 {
                out[b * QK_K + written + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
            }
            written += 64;
            j += 64;
            q = &q[32..];
            is += 2;
        }
    }
}

/// NOTE: unlike Q4_K/Q8_0/Q4_0 (whose f16 scale leads the block), the Q6_K
/// block serializes as ql[128], qh[64], scales[16], d — the f16 super-block
/// scale is LAST (llama.cpp `block_q6_K`, 210 bytes).
fn dequant_row_q6k(bytes: &[u8], n: usize, out: &mut [f32]) {
    let nblocks = n / QK_K;
    for b in 0..nblocks {
        let block = &bytes[b * 210..b * 210 + 210];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let mut y = &mut out[b * QK_K..b * QK_K + QK_K];
        let (mut ql_off, mut qh_off, mut sc_off) = (0usize, 0usize, 0usize);
        while !y.is_empty() {
            for l in 0..32 {
                let is = l / 16;
                // Q6_K scales are SIGNED bytes (unlike Q4_K's unsigned
                // 6-bit pairs).
                let s1 = scales[sc_off + is] as i8 as f32;
                let s2 = scales[sc_off + is + 2] as i8 as f32;
                let s3 = scales[sc_off + is + 4] as i8 as f32;
                let s4 = scales[sc_off + is + 6] as i8 as f32;
                let q1 = ((ql[ql_off + l] & 0xF) as i32 | (((qh[qh_off + l] >> 0) & 3) as i32) << 4) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) as i32 | (((qh[qh_off + l] >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i32 | (((qh[qh_off + l] >> 6) & 3) as i32) << 4) - 32;
                y[l] = d * s1 * q1 as f32;
                y[l + 32] = d * s2 * q2 as f32;
                y[l + 64] = d * s3 * q3 as f32;
                y[l + 96] = d * s4 * q4 as f32;
            }
            y = &mut y[128..];
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

/// Fused dot product: quantized weight row · f32 input vector.
///
/// `bytes` holds `n` elements in the given quant format (whole blocks),
/// `y` holds the matching f32 row. This is the hot kernel of the whole
/// engine — plain loops so LLVM autovectorizes them under `+simd128`.
pub fn dot_row(quant: u32, bytes: &[u8], y: &[f32], n: usize) -> f32 {
    match quant {
        F32 => {
            let mut acc = 0.0f32;
            for i in 0..n {
                let b = &bytes[4 * i..4 * i + 4];
                acc += f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * y[i];
            }
            acc
        }
        F16 => {
            let mut acc = 0.0f32;
            for i in 0..n {
                acc += f16_at(bytes, i) * y[i];
            }
            acc
        }
        Q4_0 => {
            // w = d*(q-8), halves layout: lows are weights 0..15, highs are
            // 16..31 (see dequant_row_into).
            let nblocks = n / QK4_0;
            let mut acc = 0.0f32;
            for b in 0..nblocks {
                let block = &bytes[b * 18..b * 18 + 18];
                let d = f16_at(block, 0);
                let ylo = &y[b * QK4_0..b * QK4_0 + 16];
                let yhi = &y[b * QK4_0 + 16..b * QK4_0 + 32];
                let mut qy = 0.0f32;
                let mut sy = 0.0f32;
                for j in 0..16 {
                    let q0 = (block[2 + j] & 0xF) as f32;
                    let q1 = (block[2 + j] >> 4) as f32;
                    qy += q0 * ylo[j] + q1 * yhi[j];
                    sy += ylo[j] + yhi[j];
                }
                acc += d * (qy - 8.0 * sy);
            }
            acc
        }
        Q8_0 => {
            let nblocks = n / QK8_0;
            let mut acc = 0.0f32;
            for b in 0..nblocks {
                let block = &bytes[b * 34..b * 34 + 34];
                let d = f16_at(block, 0);
                let yb = &y[b * QK8_0..b * QK8_0 + QK8_0];
                let mut qy = 0.0f32;
                for j in 0..QK8_0 {
                    qy += block[2 + j] as i8 as f32 * yb[j];
                }
                acc += d * qy;
            }
            acc
        }
        Q4K_S | Q4K_M => {
            let nblocks = n / QK_K;
            let mut acc = 0.0f32;
            for b in 0..nblocks {
                let block = &bytes[b * 144..b * 144 + 144];
                let d = f16_at(block, 0);
                let dmin = f16_at(block, 1);
                let scales = &block[4..16];
                let mut q = &block[16..];
                let yb = &y[b * QK_K..b * QK_K + QK_K];
                let mut is = 0usize;
                let mut yoff = 0usize;
                let mut j = 0usize;
                while j < QK_K {
                    let (sc1, m1) = scale_min_k4(scales, is);
                    let (sc2, m2) = scale_min_k4(scales, is + 1);
                    let d1 = d * sc1 as f32;
                    let r1 = dmin * m1 as f32;
                    let d2 = d * sc2 as f32;
                    let r2 = dmin * m2 as f32;
                    let (mut qy1, mut sy1) = (0.0f32, 0.0f32);
                    for l in 0..32 {
                        let qv = (q[l] & 0xF) as f32;
                        qy1 += qv * yb[yoff + l];
                        sy1 += yb[yoff + l];
                    }
                    let (mut qy2, mut sy2) = (0.0f32, 0.0f32);
                    for l in 0..32 {
                        let qv = (q[l] >> 4) as f32;
                        qy2 += qv * yb[yoff + 32 + l];
                        sy2 += yb[yoff + 32 + l];
                    }
                    acc += d1 * qy1 - r1 * sy1 + d2 * qy2 - r2 * sy2;
                    j += 64;
                    yoff += 64;
                    q = &q[32..];
                    is += 2;
                }
            }
            acc
        }
        Q6_K => {
            let nblocks = n / QK_K;
            let mut acc = 0.0f32;
            for b in 0..nblocks {
                let block = &bytes[b * 210..b * 210 + 210];
                // d comes last in the Q6_K block (see dequant_row_q6k).
                let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
                let ql = &block[0..128];
                let qh = &block[128..192];
                let scales = &block[192..208];
                let yb = &y[b * QK_K..b * QK_K + QK_K];
                let (mut ql_off, mut qh_off, mut sc_off, mut yoff) = (0usize, 0usize, 0usize, 0usize);
                let mut chunk = 0usize;
                while chunk < QK_K {
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[ql_off + l] & 0xF) as i32 | (((qh[qh_off + l] >> 0) & 3) as i32) << 4) - 32;
                        let q2 = ((ql[ql_off + l + 32] & 0xF) as i32 | (((qh[qh_off + l] >> 2) & 3) as i32) << 4) - 32;
                        let q3 = ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 3) as i32) << 4) - 32;
                        let q4 = ((ql[ql_off + l + 32] >> 4) as i32 | (((qh[qh_off + l] >> 6) & 3) as i32) << 4) - 32;
                        // Q6_K scales are SIGNED bytes.
                        acc += d
                            * (scales[sc_off + is] as i8 as f32 * q1 as f32 * yb[yoff + l]
                                + scales[sc_off + is + 2] as i8 as f32 * q2 as f32 * yb[yoff + 32 + l]
                                + scales[sc_off + is + 4] as i8 as f32 * q3 as f32 * yb[yoff + 64 + l]
                                + scales[sc_off + is + 6] as i8 as f32 * q4 as f32 * yb[yoff + 96 + l]);
                    }
                    chunk += 128;
                    ql_off += 64;
                    qh_off += 32;
                    sc_off += 8;
                    yoff += 128;
                }
            }
            acc
        }
        _ => unreachable!("unsupported quant reached dot: {quant}"),
    }
}

/// Bytes per element for unquantized rows (used by size sanity checks).
pub fn is_supported(quant: u32) -> bool {
    quant_block(quant).is_some()
}
