#![no_std]
#![no_main]

//! Real trained char-level LM on ESP32-S3.
//! Uses q8 weights trained from /tmp/train_tiny_char_rnn.py and embedded in flash.

use defmt::info;
use esp_backtrace as _;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_println as _;

#[path = "../trained_char_lm_weights.rs"]
mod weights;

esp_bootloader_esp_idf::esp_app_desc!();

const OUT_LEN: usize = 220;

fn deq(w: i8, scale: f32) -> f32 {
    w as f32 * scale
}

fn tanh_approx(x: f32) -> f32 {
    libm::tanhf(x)
}

fn vocab_index(byte: u8) -> usize {
    let mut i = 0usize;
    while i < weights::VOCAB_SIZE {
        if weights::VOCAB[i] == byte {
            return i;
        }
        i += 1;
    }
    0
}

fn argmax(logits: &[f32; weights::VOCAB_SIZE]) -> usize {
    let mut best = 0usize;
    let mut val = logits[0];
    let mut i = 1usize;
    while i < weights::VOCAB_SIZE {
        if logits[i] > val {
            best = i;
            val = logits[i];
        }
        i += 1;
    }
    best
}

fn step(ix: usize, h: &mut [f32; weights::HIDDEN], logits: &mut [f32; weights::VOCAB_SIZE]) -> usize {
    let mut new_h = [0.0f32; weights::HIDDEN];
    let mut i = 0usize;
    while i < weights::HIDDEN {
        let mut sum = deq(weights::BH[i], weights::BH_SCALE);
        sum += deq(weights::WXH[i * weights::VOCAB_SIZE + ix], weights::WXH_SCALE);
        let mut j = 0usize;
        while j < weights::HIDDEN {
            sum += deq(weights::WHH[i * weights::HIDDEN + j], weights::WHH_SCALE) * h[j];
            j += 1;
        }
        new_h[i] = tanh_approx(sum);
        i += 1;
    }

    i = 0;
    while i < weights::HIDDEN {
        h[i] = new_h[i];
        i += 1;
    }

    let mut v = 0usize;
    while v < weights::VOCAB_SIZE {
        let mut sum = deq(weights::BY[v], weights::BY_SCALE);
        let mut j = 0usize;
        while j < weights::HIDDEN {
            sum += deq(weights::WHY[v * weights::HIDDEN + j], weights::WHY_SCALE) * h[j];
            j += 1;
        }
        logits[v] = sum;
        v += 1;
    }
    argmax(logits)
}

#[main]
fn main() -> ! {
    let _peripherals = esp_hal::init(esp_hal::Config::default());

    info!("trained-char-lm start vocab={} hidden={} out_len={}", weights::VOCAB_SIZE, weights::HIDDEN, OUT_LEN);
    info!("weights bytes q8={} exact_f32_bytes={} compression_x100={}",
        weights::WXH.len() + weights::WHH.len() + weights::WHY.len() + weights::BH.len() + weights::BY.len(),
        (weights::WXH.len() + weights::WHH.len() + weights::WHY.len() + weights::BH.len() + weights::BY.len()) * 4,
        400
    );

    let mut h = [0.0f32; weights::HIDDEN];
    let mut logits = [0.0f32; weights::VOCAB_SIZE];
    let mut out = [0u8; OUT_LEN + 2];
    out[0] = b'i';
    out[1] = b' ';
    let mut ix = vocab_index(b' ');
    let start = Instant::now();

    let mut pos = 2usize;
    while pos < OUT_LEN {
        ix = step(ix, &mut h, &mut logits);
        out[pos] = weights::VOCAB[ix];
        if pos == 16 || pos == 64 || pos == 128 || pos == OUT_LEN - 1 {
            info!("trained-char-lm progress chars={}", pos + 1);
        }
        pos += 1;
    }

    let elapsed_us = start.elapsed().as_micros();
    let cps_x100 = (OUT_LEN as u64 * 100_000_000) / elapsed_us.max(1);
    let text = core::str::from_utf8(&out[..OUT_LEN]).unwrap_or("<utf8-error>");
    info!("RESULT trained_char_lm chars={} micros={} chars_per_sec_x100={}", OUT_LEN, elapsed_us, cps_x100);
    info!("RESULT generated_text={}", text);
    info!("trained-char-lm completed");

    loop {
        let idle_start = Instant::now();
        while idle_start.elapsed() < Duration::from_millis(5000) {}
        info!("trained-char-lm idle after completion");
    }
}
