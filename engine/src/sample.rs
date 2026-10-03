//! Sampling: temperature / top-k / top-p / min-p with llama.cpp-style
//! penalties, plus a std-only SplitMix64 PRNG (no `rand` — its OS-entropy
//! source doesn't exist on wasm).

use crate::GenParams;
use crate::tokenizer::Tokenizer;

pub struct Sampler {
    params: GenParams,
    rng: SplitMix64,
    eos_token_id: u32,
}

impl Sampler {
    pub fn new(params: GenParams, tokenizer: &Tokenizer) -> Self {
        let eos_token_id = tokenizer.eos_token_id;
        let seed = params.seed;
        Self {
            params,
            rng: SplitMix64::new(seed),
            eos_token_id,
        }
    }

    /// True when the id ends generation (model EOS).
    pub fn hit_stop_token(&self, id: u32) -> bool {
        id == self.eos_token_id
    }

    /// Picks the next token from the raw logits, applying penalties over
    /// `generated`, then the sampling filters.
    pub fn pick(
        &mut self,
        logits: &[f32],
        generated: &[u32],
    ) -> Result<u32, crate::error::EngineError> {
        let mut work = logits.to_vec();
        apply_penalties(
            &mut work,
            generated,
            self.params.repeat_penalty,
            self.params.frequency_penalty,
            self.params.presence_penalty,
        );

        if self.params.temperature <= 0.01 {
            return Ok(argmax(&work));
        }

        for logit in work.iter_mut() {
            *logit /= self.params.temperature;
        }

        // Softmax → candidate filtering (min-p / top-k / top-p) → sample.
        let mut probs = softmax(&work);
        let max_p = probs.iter().cloned().fold(0.0f32, f32::max);

        // min-p: keep p >= min_p * max_p.
        if self.params.min_p > 0.0 {
            let floor = self.params.min_p * max_p;
            for p in probs.iter_mut() {
                if *p < floor {
                    *p = 0.0;
                }
            }
        }

        // top-k: zero everything past the k-th largest.
        if self.params.top_k > 0 {
            let k = (self.params.top_k as usize).min(probs.len());
            let mut order: Vec<u32> = (0..probs.len() as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                probs[b as usize].total_cmp(&probs[a as usize])
            });
            for &i in order.iter().skip(k) {
                probs[i as usize] = 0.0;
            }
        }

        // top-p: keep the smallest prefix of (sorted) tokens whose mass
        // reaches top_p, zero the rest.
        if self.params.top_p < 1.0 {
            let mut order: Vec<u32> = (0..probs.len() as u32).collect();
            order.sort_unstable_by(|&a, &b| {
                probs[b as usize].total_cmp(&probs[a as usize])
            });
            let mut cum = 0.0f32;
            let mut cutoff = order.len();
            for (rank, &i) in order.iter().enumerate() {
                cum += probs[i as usize];
                if cum >= self.params.top_p {
                    cutoff = rank + 1;
                    break;
                }
            }
            for &i in order.iter().skip(cutoff) {
                probs[i as usize] = 0.0;
            }
        }

        let total: f32 = probs.iter().sum();
        if total <= 0.0 {
            return Ok(argmax(&work));
        }
        let dart = self.rng.next_f32() * total;
        let mut cum = 0.0f32;
        for (id, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= dart {
                return Ok(id as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for p in out.iter_mut() {
            *p /= sum;
        }
    }
    out
}

/// llama.cpp-style penalties, applied over the generated tokens only â
/// penalizing prompt tokens too makes the model dodge its own question
/// ("What is 2 + 2?" â "I would need more information").
fn apply_penalties(
    logits: &mut [f32],
    generated: &[u32],
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    let n = logits.len() as u32;
    if repeat_penalty != 1.0 {
        for &id in generated {
            let Some(l) = logits.get_mut(id as usize) else {
                continue;
            };
            if id < n {
                *l = if *l <= 0.0 { *l * repeat_penalty } else { *l / repeat_penalty };
            }
        }
    }
    if frequency_penalty != 0.0 || presence_penalty != 0.0 {
        let mut counts = vec![0u32; logits.len()];
        for &id in generated {
            if let Some(c) = counts.get_mut(id as usize) {
                *c += 1;
            }
        }
        for (id, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let l = &mut logits[id];
            *l -= frequency_penalty * count as f32;
            if presence_penalty != 0.0 {
                *l -= presence_penalty;
            }
        }
    }
}

/// SplitMix64: tiny, well-mixed, no dependencies.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
