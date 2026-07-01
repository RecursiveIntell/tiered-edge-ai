#![no_std]
#![no_main]

//! Minimal on-device inference smoke test using spec-core
//! - exercises int4 matmul / layer norm / gelu / cache trait path / forward() once
//! - runs a short fixed number of inference steps with synthetic weights

use defmt::info;
use esp_backtrace as _;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_println as _;

esp_bootloader_esp_idf::esp_app_desc!();

use spec_core::transformer::{AttentionCache, LayerWeights, ModelConfig, ModelWeights, forward};

/// Simple bounded attention cache with deterministic fallback behavior.
struct SmokeCache {
    keys: [[f32; 16]; 8],
    values: [[f32; 16]; 8],
    len: usize,
}

impl SmokeCache {
    const fn new() -> Self {
        Self {
            keys: [[0.0; 16]; 8],
            values: [[0.0; 16]; 8],
            len: 0,
        }
    }
}

impl AttentionCache for SmokeCache {
    fn push(&mut self, key: &[f32], value: &[f32]) -> bool {
        if self.len >= self.keys.len() {
            return false;
        }

        let idx = self.len;
        let n = core::cmp::min(16, key.len());
        let mut i = 0;
        while i < n {
            self.keys[idx][i] = key[i];
            self.values[idx][i] = value[i];
            i += 1;
        }

        let n = core::cmp::min(16, value.len());
        let mut j = 0;
        while j < n {
            self.values[idx][j] = value[j];
            j += 1;
        }

        self.len += 1;
        true
    }

    fn attend(&self, query: &[f32]) -> [f32; 48] {
        let mut out = [0.0f32; 48];

        if self.len == 0 {
            return out;
        }

        let mut sums = [0.0f32; 3];
        let mut k = 0;
        while k < self.len {
            let mut h = 0;
            while h < 3 {
                let mut dot = 0.0f32;
                let mut i = 0;
                while i < 16 {
                    let q = if i < query.len() { query[i] } else { 0.0 };
                    let v = self.keys[k][i];
                    dot += q * v;
                    i += 1;
                }
                sums[h] += dot;
                let mut out_i = 0;
                while out_i < 16 {
                    out[(h * 16) + out_i] += dot * self.values[k][out_i];
                    out_i += 1;
                }
                h += 1;
            }
            k += 1;
        }

        let mut h = 0;
        while h < 48 {
            out[h] /= sums[(h / 16).min(2)] + 0.0001;
            h += 1;
        }
        out
    }

    fn len(&self) -> usize {
        self.len
    }

    fn rollback(&mut self, new_len: usize) {
        self.len = new_len.min(self.keys.len());
    }
}

const EMBED_TABLE: [u8; 6144] = [0x11u8; 6144];
const EMBED_SCALES: [f32; 256] = [0.015625; 256];

const LAY_QKV_WEIGHTS: [u8; 3456] = [0x11u8; 3456];
const LAY_QKV_SCALES: [f32; 144] = [0.015625; 144];
const LAY_QKV_BIAS: [f32; 144] = [0.0; 144];

const LAY_O_WEIGHTS: [u8; 1152] = [0x11u8; 1152];
const LAY_O_SCALES: [f32; 48] = [0.015625; 48];
const LAY_O_BIAS: [f32; 48] = [0.0; 48];

const LAY_LN1_GAMMA: [f32; 48] = [1.0; 48];
const LAY_LN1_BETA: [f32; 48] = [0.0; 48];

const LAY_FFN_UP_WEIGHTS: [u8; 2304] = [0x11u8; 2304];
const LAY_FFN_UP_SCALES: [f32; 96] = [0.015625; 96];
const LAY_FFN_UP_BIAS: [f32; 96] = [0.0; 96];

const LAY_FFN_DOWN_WEIGHTS: [u8; 2304] = [0x11u8; 2304];
const LAY_FFN_DOWN_SCALES: [f32; 48] = [0.015625; 48];
const LAY_FFN_DOWN_BIAS: [f32; 48] = [0.0; 48];

const LAY_LN2_GAMMA: [f32; 48] = [1.0; 48];
const LAY_LN2_BETA: [f32; 48] = [0.0; 48];

const LAYERS: [LayerWeights; 2] = [
    LayerWeights {
        qkv_weights: &LAY_QKV_WEIGHTS,
        qkv_scales: &LAY_QKV_SCALES,
        qkv_bias: &LAY_QKV_BIAS,
        o_weights: &LAY_O_WEIGHTS,
        o_scales: &LAY_O_SCALES,
        o_bias: &LAY_O_BIAS,
        ln1_gamma: &LAY_LN1_GAMMA,
        ln1_beta: &LAY_LN1_BETA,
        ffn_up_weights: &LAY_FFN_UP_WEIGHTS,
        ffn_up_scales: &LAY_FFN_UP_SCALES,
        ffn_up_bias: &LAY_FFN_UP_BIAS,
        ffn_down_weights: &LAY_FFN_DOWN_WEIGHTS,
        ffn_down_scales: &LAY_FFN_DOWN_SCALES,
        ffn_down_bias: &LAY_FFN_DOWN_BIAS,
        ln2_gamma: &LAY_LN2_GAMMA,
        ln2_beta: &LAY_LN2_BETA,
    },
    LayerWeights {
        qkv_weights: &LAY_QKV_WEIGHTS,
        qkv_scales: &LAY_QKV_SCALES,
        qkv_bias: &LAY_QKV_BIAS,
        o_weights: &LAY_O_WEIGHTS,
        o_scales: &LAY_O_SCALES,
        o_bias: &LAY_O_BIAS,
        ln1_gamma: &LAY_LN1_GAMMA,
        ln1_beta: &LAY_LN1_BETA,
        ffn_up_weights: &LAY_FFN_UP_WEIGHTS,
        ffn_up_scales: &LAY_FFN_UP_SCALES,
        ffn_up_bias: &LAY_FFN_UP_BIAS,
        ffn_down_weights: &LAY_FFN_DOWN_WEIGHTS,
        ffn_down_scales: &LAY_FFN_DOWN_SCALES,
        ffn_down_bias: &LAY_FFN_DOWN_BIAS,
        ln2_gamma: &LAY_LN2_GAMMA,
        ln2_beta: &LAY_LN2_BETA,
    },
];

const MODEL_WEIGHTS: ModelWeights = ModelWeights {
    embed_table: &EMBED_TABLE,
    embed_scales: &EMBED_SCALES,
    layers: &LAYERS,
    output_weights: &[0x11u8; 6144],
    output_scales: &[0.015625; 256],
    final_gamma: &[1.0; 48],
    final_beta: &[0.0; 48],
};

const CONFIG: ModelConfig = ModelConfig {
    vocab_size: 256,
    embed_dim: 48,
    num_heads: 3,
    head_dim: 16,
    num_layers: 2,
    ffn_dim: 96,
    max_seq_len: 1024,
};

#[main]
fn main() -> ! {
    let _peripherals = esp_hal::init(esp_hal::Config::default());

    let mut cache0 = SmokeCache::new();
    let mut cache1 = SmokeCache::new();
    let mut caches: [&mut dyn AttentionCache; 2] = [&mut cache0, &mut cache1];

    let mut logits = [0.0f32; 256];
    let mut hidden = [0.0f32; 48];

    let mut token = 3usize;
    let mut step = 0usize;

    loop {
        forward(
            token,
            step,
            &MODEL_WEIGHTS,
            CONFIG,
            &mut caches,
            &mut logits,
            &mut hidden,
        );

        let mut best_idx = 0usize;
        let mut best_val = logits[0];
        let mut i = 1usize;
        while i < 16 {
            if logits[i] > best_val {
                best_idx = i;
                best_val = logits[i];
            }
            i += 1;
        }

        info!(
            "inference-smoke step={} token_in={} best_idx={} best_logit={} cache0_len={} cache1_len={}",
            step,
            token,
            best_idx,
            best_val,
            caches[0].len(),
            caches[1].len(),
        );

        token = (best_idx + 1) % CONFIG.vocab_size;
        step += 1;

        if step >= 64 {
            break;
        }

        let mut z = 0usize;
        while z < hidden.len() {
            hidden[z] = 0.0;
            z += 1;
        }

        let mut r = 0;
        while r < caches.len() {
            info!("cache rollback step={}", r);
            caches[r].rollback(1);
            r += 1;
        }
    }

    info!("inference-smoke completed at step={}", step);

    loop {
        let idle_start = Instant::now();
        while idle_start.elapsed() < Duration::from_millis(5000) {}
        info!("inference-smoke idle after completion");
    }
}
