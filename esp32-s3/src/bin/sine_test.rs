#![no_std]
#![no_main]
//! Sine model test — verifies MicroFlow works on the ESP32-S3.
//! Runs a quantized TFLite sine predictor and compares against libm.

use defmt::info;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_println as _;
use libm::sinf;
use microflow::model;
use nalgebra::matrix;

esp_bootloader_esp_idf::esp_app_desc!();

#[model("models/sine.tflite")]
struct Sine;

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    let test_values: [f32; 5] = [0.0, 0.5, 1.0, 1.5, 3.14159];

    for &x in &test_values {
        let start = Instant::now();
        let y_predicted = Sine::predict(matrix![x])[0];
        let elapsed = start.elapsed();
        let y_exact = sinf(x);

        info!(
            "sin({}) = predicted: {}, exact: {}, error: {}, time: {} us",
            x,
            y_predicted,
            y_exact,
            y_exact - y_predicted,
            elapsed.as_micros()
        );
    }

    info!("=== Sentinel sine model test complete ===");

    loop {
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(5000) {}
    }
}
