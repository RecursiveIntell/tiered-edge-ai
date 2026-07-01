# ESP32-S3 Compressed-Domain Attention Bench Implementation Plan

> For Hermes: execute immediately; do not stop at planning.

Goal: add a third on-board transformer KV benchmark path that scores compressed int4 keys directly instead of dequantizing keys for every attention dot product.

Architecture: keep the existing llm-limit-bench exact f32 KV and int4 decode-score KV baselines unchanged. Add a compressed-score cache variant with the same int4 storage budget as the int4 baseline, but quantize the query once per attention call and compute key dot products directly from packed int4 nibbles. Values still decode for aggregation, so this is not a full fib-quant/turbo-quant port; it is the smallest honest hardware gate for the next bottleneck: compressed-key scoring.

Tech stack: no_std Rust, esp-hal, spec-core AttentionCache, espflash, defmt logs, JSON receipt.

Evidence-backed current state:
- Repo path: /home/sikmindz/projects/tiered-edge-ai/esp32-s3
- Existing firmware: src/bin/llm_limit_bench.rs
- Existing benchmark paths: ExactCache<64> and Int4Cache<256>
- Existing measured board result: exact f32 64 tokens at ~34.62 tok/s; int4 decode-score 256 tokens at ~21.70 tok/s; int4 KV capacity 8x exact f32.
- Current gap: compressed KV improves capacity, but current int4 path decodes keys while scoring. That pays decode cost every attention step.
- Local fib/turbo crates have compressed attention concepts, but are not no_std ESP32-ready dependencies. Directly importing them into this firmware would widen scope and likely pull std/serde/rand dependencies. The correct near-term hardware gate is a no_std compressed-key-score variant inside the existing bench.

Tasks:

1. Add CompressedScoreCache to src/bin/llm_limit_bench.rs
- Store keys and values with same packed int4 layout as Int4Cache.
- Add query quantization per attend() call.
- Compute dot product with int4 nibble integers directly: sum(q_i * k_i) * query_scale * key_scale.
- Keep value aggregation identical to Int4Cache so only key-scoring path changes.
- Implement AttentionCache for this type.

2. Extend main() benchmark sequence
- Add two static CompressedScoreCache<COMPRESSED_TOKENS> layers.
- Run run_loop after int4 decode-score run.
- Log RESULT compressed_score tokens/micros/tps_x100/last_token.
- Log RESULT compressed_score_vs_int4_speed_x100 and compressed_score_vs_exact_speed_x100.
- Keep existing exact/int4 capacity math unchanged.

3. Build and flash on ESP32-S3
- Command: source ~/export-esp.sh && cargo +esp build --target xtensa-esp32s3-none-elf --bin llm-limit-bench --release -Z build-std=core,alloc
- Command: espflash flash target/xtensa-esp32s3-none-elf/release/llm-limit-bench --port /dev/ttyACM0 --non-interactive --chip esp32s3 --monitor --log-format defmt --baud 115200 --monitor-baud 115200 --after hard-reset
- Capture log to /tmp/llm-limit-bench-compressed-score.log.

4. Write receipt
- Path: docs/llm-limit-bench-compressed-score-receipt.json
- Include board, model shape, bytes/token, exact/int4/compressed-score timings, speed ratios, capacity ratios, panic/watchdog status, and claim boundary.

Acceptance criteria:
- Build passes.
- Flash passes.
- Runtime reaches llm-limit-bench completed.
- Log has exact, int4, and compressed_score RESULT lines.
- No panic after run start.
- No watchdog reset after run start.
- Receipt states whether compressed scoring improved speed, worsened speed, or was neutral.
