# Tiered Edge AI

An experimental split between ESP32-S3 firmware, a host inference receiver, and typed quantization/feature-transfer contracts.

The source contains runnable demonstration pieces and proposed integration work. It does not yet implement a complete same-model dual-quantization deployment or prove automatic power management, GPU acceleration, or end-to-end hardware reliability.

## What is implemented

- [esp32-s3/src/bin](esp32-s3/src/bin/): no_std firmware examples for sine inference, sentinel logic, WiFi, and additional local inference experiments
- [uno-q/server/server.py](uno-q/server/server.py): a framed TCP receiver for `TEAI` messages; it attempts TFLite or ONNX loading and falls back to a mock path when no usable model/backend is available
- [uno-q/server/mock_client.py](uno-q/server/mock_client.py): host-side protocol client
- [bridge/src/lib.rs](bridge/src/lib.rs): tier and quantization configuration types, feature-transfer serialization, size estimates, and unit tests

The bridge's FibQuant/TurboQuant dependencies are commented out in [bridge/Cargo.toml](bridge/Cargo.toml). Naming those schemes in an enum does not implement model conversion, confidence calibration, or codec integration.

## Protocol boundary

The firmware/receiver path uses TCP with a `TEAI` magic value, a little-endian payload length, and a typed feature payload. Responses are length-prefixed JSON. It is not an HTTP API. A running host receiver is required; a network message does not itself power on an off gateway.

## Host checks

```bash
python3 -m py_compile uno-q/server/server.py uno-q/server/mock_client.py
cargo test --manifest-path bridge/Cargo.toml
```

The Python check validates syntax. The Rust tests cover the local contract helpers. Neither exercises an ESP32 or establishes real-model inference.

For a controlled local protocol experiment, inspect the receiver's bind behavior before running it:

```bash
python3 uno-q/server/server.py --port 8080
# In another terminal:
python3 uno-q/server/mock_client.py --host 127.0.0.1 --port 8080
```

Without a valid model/backend, this tests mock behavior. Check the receiver logs for the selected backend and keep the service on a trusted test network; this example does not supply a production authentication or transport-security boundary.

## ESP32-S3 builds

Use an appropriate board and the custom Xtensa Rust toolchain documented by [esp-rs/espup](https://github.com/esp-rs/espup). The firmware has its own [Cargo manifest](esp32-s3/Cargo.toml), target configuration, build script, and environment requirements.

```bash
cd esp32-s3
cargo +esp build --release --bin sentinel-wifi
```

Review the chosen binary before flashing. Several examples use synthetic inputs; their presence does not prove camera, microphone, physical sensor, or display integration. Board flash/PSRAM capacity, available pins, power draw, and timing must be established on the exact hardware.

## Still to establish

- Actual dual-quantization export with the selected codec owners
- Measured confidence calibration between a defined pair of models
- Real sensor input and the chosen hardware's complete inference path
- Device/network failure and recovery behavior
- Power-management and accelerator measurements where those are intended

The original architecture is a design direction. Source-present scripts, mock responses, and estimated tensor sizes are not interchangeable with deployment receipts.

## License

The bridge manifest declares `MIT OR Apache-2.0`; inspect each component's manifest and notices for its own terms. No repository-wide license text is supplied in this checkout.
