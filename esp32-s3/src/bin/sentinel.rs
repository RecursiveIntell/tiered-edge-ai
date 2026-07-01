#![no_std]
#![no_main]
//! Tiered Edge AI — ESP32-S3 Sentinel Firmware.
//!
//! Runs a tiny quantized model continuously (Tier 0). When confidence drops
//! below threshold, logs a wake event.

use defmt::{info, warn};
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_println as _;

use microflow::model;
use nalgebra::matrix;

esp_bootloader_esp_idf::esp_app_desc!();

mod tiered_config {
    include!(concat!(env!("OUT_DIR"), "/tiered_config.rs"));
}

const UNO_Q_IP: [u8; 4] = tiered_config::UNO_Q_IP;
const UNO_Q_PORT: u16 = tiered_config::UNO_Q_PORT;
const CONFIDENCE_THRESHOLD: f32 = 0.75;
const INFERENCE_INTERVAL_MS: u64 = 1000;

#[model("models/sine.tflite")]
struct Sentinel;

fn assess_confidence(prediction: f32) -> f32 {
    if prediction >= 0.0 && prediction <= 1.0 {
        1.0 - (prediction - 0.5).abs()
    } else {
        0.0
    }
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    info!("=== Tiered Edge AI Sentinel ===");
    info!("ESP32-S3 Tier 0 (sentinel) starting up");
    info!(
        "UNO Q target: {}.{}.{}.{}:{}",
        UNO_Q_IP[0], UNO_Q_IP[1], UNO_Q_IP[2], UNO_Q_IP[3], UNO_Q_PORT
    );
    info!("Confidence threshold: {}", CONFIDENCE_THRESHOLD);
    info!("Inference interval: {}ms", INFERENCE_INTERVAL_MS);

    let mut cycle: u32 = 0;

    loop {
        let sensor_input = if cycle % 10 == 0 {
            2.5
        } else {
            (cycle as f32 * 0.1) % 6.28
        };

        let start = Instant::now();
        let prediction = Sentinel::predict(matrix![sensor_input])[0];
        let inference_time = start.elapsed();

        let confidence = assess_confidence(prediction);

        info!(
            "cycle {} | input: {} | pred: {} | conf: {} | time: {}us",
            cycle,
            sensor_input,
            prediction,
            confidence,
            inference_time.as_micros()
        );

        if confidence < CONFIDENCE_THRESHOLD {
            warn!(
                "LOW CONFIDENCE ({}) — would wake UNO Q for input {}",
                confidence, sensor_input
            );
        }

        cycle += 1;

        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(INFERENCE_INTERVAL_MS) {}
    }
}
