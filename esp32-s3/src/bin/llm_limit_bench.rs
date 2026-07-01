#![no_std]
#![no_main]

//! On-board tiny-GPT limit benchmark.
//!
//! This is a real transformer inference path benchmark: token embedding,
//! QKV projection, attention cache, FFN, final logits, sampling loop.
//! It compares exact f32 KV, int4 decode-score KV, and int4 compressed-key-score KV on the ESP32-S3.
//!
//! NOTE: weights are deterministic fixture weights, not a trained checkpoint.
//! This measures what the inference engine makes possible on-board: KV memory
//! capacity and runtime envelope. It is not a language-quality benchmark.

use defmt::info;
use esp_backtrace as _;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_println as _;
use spec_core::sample::argmax;
use spec_core::transformer::{forward, AttentionCache, LayerWeights, ModelConfig, ModelWeights};

esp_bootloader_esp_idf::esp_app_desc!();

const VOCAB: usize = 256;
const EMBED: usize = 48;
const HEAD: usize = 16;
const LAYERS_N: usize = 2;
const FFN: usize = 96;
const EXACT_TOKENS: usize = 64;
const COMPRESSED_TOKENS: usize = 256;
const PSRAM_BUDGET_BYTES: usize = 8 * 1024 * 1024;

const EMBED_TABLE: [u8; 6144] = [0x21u8; 6144];
const EMBED_SCALES: [f32; VOCAB] = [0.03125; VOCAB];

const LAY_QKV_WEIGHTS: [u8; 3456] = [0x1fu8; 3456];
const LAY_QKV_SCALES: [f32; 144] = [0.0078125; 144];
const LAY_QKV_BIAS: [f32; 144] = [0.0; 144];

const LAY_O_WEIGHTS: [u8; 1152] = [0x12u8; 1152];
const LAY_O_SCALES: [f32; EMBED] = [0.0078125; EMBED];
const LAY_O_BIAS: [f32; EMBED] = [0.0; EMBED];

const LAY_LN1_GAMMA: [f32; EMBED] = [1.0; EMBED];
const LAY_LN1_BETA: [f32; EMBED] = [0.0; EMBED];

const LAY_FFN_UP_WEIGHTS: [u8; 2304] = [0x2fu8; 2304];
const LAY_FFN_UP_SCALES: [f32; FFN] = [0.0078125; FFN];
const LAY_FFN_UP_BIAS: [f32; FFN] = [0.0; FFN];

const LAY_FFN_DOWN_WEIGHTS: [u8; 2304] = [0x11u8; 2304];
const LAY_FFN_DOWN_SCALES: [f32; EMBED] = [0.0078125; EMBED];
const LAY_FFN_DOWN_BIAS: [f32; EMBED] = [0.0; EMBED];

const LAY_LN2_GAMMA: [f32; EMBED] = [1.0; EMBED];
const LAY_LN2_BETA: [f32; EMBED] = [0.0; EMBED];

const LAYERS: [LayerWeights; LAYERS_N] = [
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
    output_weights: &[0x12u8; 6144],
    output_scales: &[0.0078125; VOCAB],
    final_gamma: &[1.0; EMBED],
    final_beta: &[0.0; EMBED],
};

const CONFIG: ModelConfig = ModelConfig {
    vocab_size: VOCAB,
    embed_dim: EMBED,
    num_heads: 3,
    head_dim: HEAD,
    num_layers: LAYERS_N,
    ffn_dim: FFN,
    max_seq_len: 1024,
};

struct ExactCache<const N: usize> {
    keys: [[f32; EMBED]; N],
    values: [[f32; EMBED]; N],
    len: usize,
}

impl<const N: usize> ExactCache<N> {
    const fn new() -> Self {
        Self { keys: [[0.0; EMBED]; N], values: [[0.0; EMBED]; N], len: 0 }
    }
}

impl<const N: usize> AttentionCache for ExactCache<N> {
    fn push(&mut self, key: &[f32], value: &[f32]) -> bool {
        if self.len >= N { return false; }
        let idx = self.len;
        let mut i = 0;
        while i < EMBED {
            self.keys[idx][i] = if i < key.len() { key[i] } else { 0.0 };
            self.values[idx][i] = if i < value.len() { value[i] } else { 0.0 };
            i += 1;
        }
        self.len += 1;
        true
    }

    fn attend(&self, query: &[f32]) -> [f32; EMBED] {
        let mut out = [0.0f32; EMBED];
        if self.len == 0 { return out; }
        let mut t = 0;
        while t < self.len {
            let mut dot = 0.0f32;
            let mut i = 0;
            while i < HEAD {
                dot += query[i] * self.keys[t][i];
                i += 1;
            }
            let weight = dot / ((self.len as f32) + 0.0001);
            let mut j = 0;
            while j < EMBED {
                out[j] += weight * self.values[t][j];
                j += 1;
            }
            t += 1;
        }
        out
    }

    fn len(&self) -> usize { self.len }
    fn rollback(&mut self, new_len: usize) { self.len = new_len.min(N); }
}

struct Int4Cache<const N: usize> {
    keys: [[u8; EMBED / 2]; N],
    values: [[u8; EMBED / 2]; N],
    key_scales: [f32; N],
    value_scales: [f32; N],
    len: usize,
}

impl<const N: usize> Int4Cache<N> {
    const fn new() -> Self {
        Self {
            keys: [[0; EMBED / 2]; N],
            values: [[0; EMBED / 2]; N],
            key_scales: [1.0; N],
            value_scales: [1.0; N],
            len: 0,
        }
    }

    fn pack_vec(src: &[f32], packed: &mut [u8; EMBED / 2]) -> f32 {
        let mut max_abs = 0.0f32;
        let mut i = 0;
        while i < EMBED {
            let v = if i < src.len() { src[i] } else { 0.0 };
            let a = if v < 0.0 { -v } else { v };
            if a > max_abs { max_abs = a; }
            i += 1;
        }
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        let mut j = 0;
        while j < EMBED {
            let v = if j < src.len() { src[j] } else { 0.0 };
            let scaled = v / scale;
            let q: i8 = if scaled >= 7.0 { 7 } else if scaled <= -8.0 { -8 } else { scaled as i8 };
            let nibble = (q as u8) & 0x0f;
            if j % 2 == 0 { packed[j / 2] = nibble; } else { packed[j / 2] |= nibble << 4; }
            j += 1;
        }
        scale
    }

    fn unpack_at(packed: &[u8; EMBED / 2], scale: f32, idx: usize) -> f32 {
        let n = if idx % 2 == 0 { packed[idx / 2] & 0x0f } else { (packed[idx / 2] >> 4) & 0x0f };
        let signed = if n & 0x08 != 0 { ((n as i8) | (0xf0u8 as i8)) as i8 } else { n as i8 };
        signed as f32 * scale
    }
}

impl<const N: usize> AttentionCache for Int4Cache<N> {
    fn push(&mut self, key: &[f32], value: &[f32]) -> bool {
        if self.len >= N { return false; }
        let idx = self.len;
        self.keys[idx] = [0; EMBED / 2];
        self.values[idx] = [0; EMBED / 2];
        self.key_scales[idx] = Self::pack_vec(key, &mut self.keys[idx]);
        self.value_scales[idx] = Self::pack_vec(value, &mut self.values[idx]);
        self.len += 1;
        true
    }

    fn attend(&self, query: &[f32]) -> [f32; EMBED] {
        let mut out = [0.0f32; EMBED];
        if self.len == 0 { return out; }
        let mut t = 0;
        while t < self.len {
            let mut dot = 0.0f32;
            let mut i = 0;
            while i < HEAD {
                let k = Self::unpack_at(&self.keys[t], self.key_scales[t], i);
                dot += query[i] * k;
                i += 1;
            }
            let weight = dot / ((self.len as f32) + 0.0001);
            let mut j = 0;
            while j < EMBED {
                let v = Self::unpack_at(&self.values[t], self.value_scales[t], j);
                out[j] += weight * v;
                j += 1;
            }
            t += 1;
        }
        out
    }

    fn len(&self) -> usize { self.len }
    fn rollback(&mut self, new_len: usize) { self.len = new_len.min(N); }
}


/// Same int4 storage as Int4Cache, but scores keys in the compressed integer domain.
/// Values still decode for aggregation. This isolates whether avoiding key decode in
/// QK scoring improves the current ESP32-S3 runtime envelope.
struct CompressedScoreCache<const N: usize> {
    keys: [[u8; EMBED / 2]; N],
    values: [[u8; EMBED / 2]; N],
    key_scales: [f32; N],
    value_scales: [f32; N],
    len: usize,
}

impl<const N: usize> CompressedScoreCache<N> {
    const fn new() -> Self {
        Self {
            keys: [[0; EMBED / 2]; N],
            values: [[0; EMBED / 2]; N],
            key_scales: [1.0; N],
            value_scales: [1.0; N],
            len: 0,
        }
    }

    fn unpack_i4(packed: &[u8; EMBED / 2], idx: usize) -> i8 {
        let n = if idx % 2 == 0 {
            packed[idx / 2] & 0x0f
        } else {
            (packed[idx / 2] >> 4) & 0x0f
        };
        if n & 0x08 != 0 { (n as i8) | (0xf0u8 as i8) } else { n as i8 }
    }

    fn quantize_query(query: &[f32], q: &mut [i8; HEAD]) -> f32 {
        let mut max_abs = 0.0f32;
        let mut i = 0;
        while i < HEAD {
            let v = if i < query.len() { query[i] } else { 0.0 };
            let a = if v < 0.0 { -v } else { v };
            if a > max_abs { max_abs = a; }
            i += 1;
        }
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        let mut j = 0;
        while j < HEAD {
            let v = if j < query.len() { query[j] } else { 0.0 };
            let scaled = v / scale;
            q[j] = if scaled >= 7.0 { 7 } else if scaled <= -8.0 { -8 } else { scaled as i8 };
            j += 1;
        }
        scale
    }
}

impl<const N: usize> AttentionCache for CompressedScoreCache<N> {
    fn push(&mut self, key: &[f32], value: &[f32]) -> bool {
        if self.len >= N { return false; }
        let idx = self.len;
        self.keys[idx] = [0; EMBED / 2];
        self.values[idx] = [0; EMBED / 2];
        self.key_scales[idx] = Int4Cache::<N>::pack_vec(key, &mut self.keys[idx]);
        self.value_scales[idx] = Int4Cache::<N>::pack_vec(value, &mut self.values[idx]);
        self.len += 1;
        true
    }

    fn attend(&self, query: &[f32]) -> [f32; EMBED] {
        let mut out = [0.0f32; EMBED];
        if self.len == 0 { return out; }

        let mut q = [0i8; HEAD];
        let q_scale = Self::quantize_query(query, &mut q);

        let mut t = 0;
        while t < self.len {
            let mut dot_i32 = 0i32;
            let mut i = 0;
            while i < HEAD {
                let k = Self::unpack_i4(&self.keys[t], i);
                dot_i32 += (q[i] as i32) * (k as i32);
                i += 1;
            }
            let dot = (dot_i32 as f32) * q_scale * self.key_scales[t];
            let weight = dot / ((self.len as f32) + 0.0001);

            let mut j = 0;
            while j < EMBED {
                let v = Int4Cache::<N>::unpack_at(&self.values[t], self.value_scales[t], j);
                out[j] += weight * v;
                j += 1;
            }
            t += 1;
        }
        out
    }

    fn len(&self) -> usize { self.len }
    fn rollback(&mut self, new_len: usize) { self.len = new_len.min(N); }
}

fn run_loop(caches: &mut [&mut dyn AttentionCache], max_tokens: usize) -> (usize, u64, u16) {
    let mut logits = [0.0f32; VOCAB];
    let mut hidden = [0.0f32; EMBED];
    let mut token = 32usize;
    let mut generated = 0usize;
    let start = Instant::now();

    while generated < max_tokens {
        forward(token, generated, &MODEL_WEIGHTS, CONFIG, caches, &mut logits, &mut hidden);
        token = (argmax(&logits) as usize + 1) % VOCAB;
        generated += 1;

        let mut i = 0;
        while i < EMBED { hidden[i] = 0.0; i += 1; }

        if generated == 1 || generated == 16 || generated == 64 || generated == 128 || generated == 256 {
            info!("bench progress generated={} token={}", generated, token);
        }
    }

    (generated, start.elapsed().as_micros(), token as u16)
}

#[main]
fn main() -> ! {
    let _peripherals = esp_hal::init(esp_hal::Config::default());

    static EXACT0: static_cell::StaticCell<ExactCache<EXACT_TOKENS>> = static_cell::StaticCell::new();
    static EXACT1: static_cell::StaticCell<ExactCache<EXACT_TOKENS>> = static_cell::StaticCell::new();
    static COMP0: static_cell::StaticCell<Int4Cache<COMPRESSED_TOKENS>> = static_cell::StaticCell::new();
    static COMP1: static_cell::StaticCell<Int4Cache<COMPRESSED_TOKENS>> = static_cell::StaticCell::new();
    static CS0: static_cell::StaticCell<CompressedScoreCache<COMPRESSED_TOKENS>> = static_cell::StaticCell::new();
    static CS1: static_cell::StaticCell<CompressedScoreCache<COMPRESSED_TOKENS>> = static_cell::StaticCell::new();

    let exact_bytes_per_token_per_layer = 2 * EMBED * 4;
    let int4_bytes_per_token_per_layer = 2 * (EMBED / 2);
    let exact_bytes_per_token = exact_bytes_per_token_per_layer * LAYERS_N;
    let int4_bytes_per_token = int4_bytes_per_token_per_layer * LAYERS_N;
    let ratio_x100 = (exact_bytes_per_token * 100) / int4_bytes_per_token;
    let exact_capacity = PSRAM_BUDGET_BYTES / exact_bytes_per_token;
    let int4_capacity = PSRAM_BUDGET_BYTES / int4_bytes_per_token;

    info!("llm-limit-bench start vocab={} embed={} layers={} heads={} ffn={}", VOCAB, EMBED, LAYERS_N, 3, FFN);
    info!("kv exact bytes/token={} int4 bytes/token={} ratio_x100={}", exact_bytes_per_token, int4_bytes_per_token, ratio_x100);
    info!("theoretical 8MiB kv capacity exact_tokens={} int4_tokens={}", exact_capacity, int4_capacity);

    let exact0 = EXACT0.init(ExactCache::new());
    let exact1 = EXACT1.init(ExactCache::new());
    let mut exact_caches: [&mut dyn AttentionCache; LAYERS_N] = [exact0, exact1];
    let (e_tokens, e_us, e_last) = run_loop(&mut exact_caches, EXACT_TOKENS);
    let e_tps_x100 = (e_tokens as u64 * 100_000_000) / e_us.max(1);
    info!("RESULT exact tokens={} micros={} tps_x100={} last_token={}", e_tokens, e_us, e_tps_x100, e_last);

    let comp0 = COMP0.init(Int4Cache::new());
    let comp1 = COMP1.init(Int4Cache::new());
    let mut comp_caches: [&mut dyn AttentionCache; LAYERS_N] = [comp0, comp1];
    let (c_tokens, c_us, c_last) = run_loop(&mut comp_caches, COMPRESSED_TOKENS);
    let c_tps_x100 = (c_tokens as u64 * 100_000_000) / c_us.max(1);
    info!("RESULT int4 tokens={} micros={} tps_x100={} last_token={}", c_tokens, c_us, c_tps_x100, c_last);

    let cs0 = CS0.init(CompressedScoreCache::new());
    let cs1 = CS1.init(CompressedScoreCache::new());
    let mut cs_caches: [&mut dyn AttentionCache; LAYERS_N] = [cs0, cs1];
    let (cs_tokens, cs_us, cs_last) = run_loop(&mut cs_caches, COMPRESSED_TOKENS);
    let cs_tps_x100 = (cs_tokens as u64 * 100_000_000) / cs_us.max(1);
    info!("RESULT compressed_score tokens={} micros={} tps_x100={} last_token={}", cs_tokens, cs_us, cs_tps_x100, cs_last);

    let cs_vs_int4_speed_x100 = (cs_tps_x100 * 100) / c_tps_x100.max(1);
    let cs_vs_exact_speed_x100 = (cs_tps_x100 * 100) / e_tps_x100.max(1);
    let measured_capacity_x100 = (c_tokens * 100) / e_tokens.max(1);
    let theoretical_capacity_x100 = (int4_capacity * 100) / exact_capacity.max(1);
    info!("RESULT improvement measured_capacity_x100={} theoretical_capacity_x100={} compressed_score_vs_int4_speed_x100={} compressed_score_vs_exact_speed_x100={}", measured_capacity_x100, theoretical_capacity_x100, cs_vs_int4_speed_x100, cs_vs_exact_speed_x100);
    info!("llm-limit-bench completed");

    loop {
        let idle_start = Instant::now();
        while idle_start.elapsed() < Duration::from_millis(5000) {}
        info!("llm-limit-bench idle after completion");
    }
}
