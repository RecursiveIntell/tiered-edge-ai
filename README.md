# Tiered Edge AI — Adaptive Quantization Inference Pipeline

A hierarchical edge AI system where an ultra-low-power ESP32-S3 runs a tiny
quantized sentinel model continuously, and wakes a powerful Arduino UNO Q 4GB
(quad-core ARM A53 + GPU) for heavy inference only when needed.

The novel contribution: the same neural network model is exported at two
different quantization levels using custom vector quantization (fib-quant /
turbo-quant), creating a confidence bridge between tiers.

## Architecture

```
                          TIERED INFERENCE PIPELINE

  ┌─────────────────────────────────────────────────────────────────┐
  │                     ESP32-S3 (TIER 0 — SENTINEL)                 │
  │                                                                 │
  │  Xtensa LX7 dual-core @ 240MHz                                  │
  │  512KB SRAM + 8MB PSRAM                                         │
  │  Always-on, ultra-low-power (~50mA)                             │
  │                                                                 │
  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
  │  │  Sensor/Cam  │───▶│  MicroFlow    │───▶│ Confidence      │    │
  │  │  Input       │    │  int4/int8    │    │ Assessor        │    │
  │  │              │    │  Sentinel     │    │                 │    │
  │  └─────────────┘    │  Model        │    │ if conf < θ:    │    │
  │                      └──────────────┘    │   wake UNO Q    │    │
  │                                           └────────┬────────┘    │
  │                                                    │ WiFi        │
  └────────────────────────────────────────────────────┼─────────────┘
                                                       │
                          Wake signal + feature vector │
                                                       │
  ┌────────────────────────────────────────────────────┼─────────────┐
  │                  ARDUINO UNO Q 4GB (TIER 1 — HEAVY)              │
  │                    │                                             │
  │  Qualcomm QRB2210 quad-core A53 @ 2.0GHz + Adreno GPU           │
  │  4GB LPDDR4 RAM, 32GB eMMC, Debian Linux                         │
  │  Boots on demand (~3s), runs full ML framework                   │
  │                    │                                             │
  │  ┌─────────────────▼──────────┐    ┌──────────────────────┐      │
  │  │  Inference Server          │    │  Heavy Model          │      │
  │  │  (Python / Rust)           │───▶│  int8 / float16       │      │
  │  │  Receives wake + features  │    │  YOLO / MobileNet /   │      │
  │  │  Returns result to ESP32   │    │  Custom model          │      │
  │  └────────────────────────────┘    └──────────────────────┘      │
  │                                                                 │
  │  STM32U585 Cortex-M33 handles real-time I/O                     │
  └─────────────────────────────────────────────────────────────────┘
```

## Components

### esp32-s3/ — Tier 0: Sentinel Firmware (Rust, no_std)

- MicroFlow TinyML inference engine with int8 quantized TFLite models
- WiFi connectivity to communicate with UNO Q
- Confidence assessment + wake signal protocol
- Sensor input (camera, microphone, environmental sensors)

### uno-q/ — Tier 1: Heavy Inference Server (Python + Rust)

- HTTP server receiving wake signals from ESP32-S3
- Full ML framework (TensorFlow Lite / ONNX Runtime / PyTorch)
- Runs higher-precision model (int8/float16) with full feature set
- GPU acceleration via Adreno
- Returns rich inference results to ESP32-S3 or cloud

### bridge/ — Quantization Bridge (Rust)

The core novelty: exports the same model at two quantization levels.

- Uses fib-quant for radial-angular vector quantization (int4-equivalent)
- Uses turbo-quant for PolarQuant/TurboQuant compression
- Model export pipeline: PyTorch → dual quantization → both tier models
- Confidence calibration: maps tier-0 confidence to tier-1 expected accuracy

## Hardware

| Board | Role | CPU | RAM | Storage |
|-------|------|-----|-----|---------|
| ESP32-S3 | Tier 0 (sentinel) | Xtensa LX7 dual @ 240MHz | 512KB SRAM + 8MB PSRAM | 8MB+ flash |
| Arduino UNO Q 4GB | Tier 1 (heavy) | Qualcomm QRB2210 quad A53 @ 2GHz + Adreno GPU | 4GB LPDDR4 | 32GB eMMC |

## Project Structure

```
tiered-edge-ai/
├── esp32-s3/          # Tier 0 firmware (Rust, no_std, esp-hal)
│   ├── Cargo.toml
│   ├── .cargo/config.toml
│   ├── models/        # Quantized TFLite models (int4/int8)
│   └── src/bin/       # Firmware binaries
├── uno-q/             # Tier 1 server (Python + Rust)
│   ├── server/        # Inference server
│   └── models/        # Higher-precision models (int8/float16)
├── bridge/            # Quantization bridge (Rust)
│   ├── Cargo.toml
│   └── src/           # Dual-quantization export pipeline
├── scripts/           # Model training + export scripts
└── docs/              # Architecture docs, benchmarks
```

## Status

- [x] ESP32-S3 toolchain installed (Xtensa Rust 1.95.0.0, esp-hal v1.1.0)
- [x] MicroFlow verified on ESP32-S3 target (sine model compiles + runs)
- [x] Project structure created
- [ ] ESP32-S3 sentinel firmware with WiFi trigger
- [ ] UNO Q inference server
- [ ] Quantization bridge (fib-quant/turbo-quant integration)
- [ ] Model export pipeline (PyTorch → dual quantization)
- [ ] End-to-end demo

## License

MIT OR Apache-2.0