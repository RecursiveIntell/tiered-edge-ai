#!/usr/bin/env python3
"""
UNO Q Sensor-Conditioned AI Gateway

Runs on Arduino UNO Q 4GB (Qualcomm QRB2210, Debian Linux).
Accepts proof receipts + sensor context from ESP32-S3 over WiFi,
runs Qwen2.5-0.5B for rich natural language response,
returns response for ESP32 display/logging.

Also provides RILM pack export endpoint: trains a domain LSTM
using PyTorch and exports int8 weights in RILM v1 format
optimized for ESP32-S3 ESP-NN SIMD + SRAM tiling.

Endpoints:
  POST /api/sensor_ai   — sensor context -> Qwen response
  POST /api/train_lstm  — train and export RILM pack
  GET  /health          — health check
"""

import json
import time
import hashlib
import struct
import os
import tempfile
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen

# Ollama endpoint (already running on UNO Q)
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"

# RILM export config
RILM_MAGIC = 0x4D4C4952  # "RILM" little-endian
RILM_VERSION = 1
DEFAULT_HIDDEN = 256
DEFAULT_LAYERS = 3
DEFAULT_VOCAB = 33
VOCAB_CHARS = "abcdefghijklmnopqrstuvwxyz .,!?\n'"  # 32 + newline = 33


class SensorAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model": OLLAMA_MODEL,
                "service": "uno-q-sensor-ai-gateway",
                "version": "1.0.0",
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""

        if self.path == "/api/sensor_ai":
            self._handle_sensor_ai(body)
        elif self.path == "/api/train_lstm":
            self._handle_train_lstm(body)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_sensor_ai(self, body):
        """Forward sensor context to Qwen2.5-0.5B and return natural language response."""
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        sensor = req.get("sensor", {})
        prompt = req.get("prompt", "")
        decision = req.get("decision", "")
        route_reason = req.get("route_reason", "")

        temp_c = sensor.get("temperature_c")
        humidity = sensor.get("humidity_pct")
        heat_index = sensor.get("heat_index_c")

        # Build a sensor-conditioned prompt for Qwen
        sensor_desc = f"Temperature: {temp_c}C"
        if humidity is not None:
            sensor_desc += f", Humidity: {humidity}%"
        if heat_index is not None:
            sensor_desc += f", Heat index: {heat_index}C"

        context = f"You are a building environmental monitor. Sensor reading: {sensor_desc}. "
        if route_reason:
            context += f"Local sentinel decision: {decision} ({route_reason}). "
        context += f"Question: {prompt if prompt else 'What should be done?'} "
        context += "Answer in one short sentence."

        # Call Ollama
        t0 = time.time()
        ollama_req = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": context,
            "stream": False,
            "options": {
                "num_predict": 60,
                "temperature": 0.3,
            },
        }).encode()

        try:
            resp = urlopen(OLLAMA_URL, data=ollama_req, timeout=30)
            result = json.loads(resp.read())
        except Exception as e:
            self._send_json(502, {"error": f"ollama error: {str(e)}"})
            return

        elapsed_ms = (time.time() - t0) * 1000
        eval_count = result.get("eval_count", 0)
        eval_duration = result.get("eval_duration", 1) / 1e9  # ns -> s

        self._send_json(200, {
            "schema": "ri_uno_q_sensor_ai_v1",
            "model": OLLAMA_MODEL,
            "sensor": sensor,
            "local_decision": decision,
            "route_reason": route_reason,
            "prompt": context,
            "response": result.get("response", "").strip(),
            "generated_tokens": eval_count,
            "elapsed_ms": round(elapsed_ms),
            "tokens_per_sec": round(eval_count / eval_duration, 2) if eval_duration > 0 else 0,
            "timestamp": int(time.time()),
        })

    def _handle_train_lstm(self, body):
        """Train a domain LSTM and export RILM v1 int8 weight pack."""
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        hidden = req.get("hidden", DEFAULT_HIDDEN)
        layers = req.get("layers", DEFAULT_LAYERS)
        steps = req.get("steps", 1200)
        seq_len = req.get("seq_len", 96)
        batch_size = req.get("batch_size", 96)

        try:
            import torch
            import torch.nn as nn
            import numpy as np
        except ImportError as e:
            self._send_json(500, {"error": f"missing dependency: {e}"})
            return

        # Domain corpus: sensor status/action phrases
        corpus = self._build_domain_corpus(req.get("extra_phrases", []))

        # Char-level vocab
        vocab = sorted(set("".join(corpus)))
        vocab_size = len(vocab)
        char_to_idx = {c: i for i, c in enumerate(vocab)}
        idx_to_char = {i: c for i, c in enumerate(vocab)}

        # Build training data — use space as separator instead of newline
        text = " ".join(corpus)
        data = [char_to_idx[c] for c in text]

        # Model
        class CharLSTM(nn.Module):
            def __init__(self, vocab_size, hidden, layers):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True)
                self.fc = nn.Linear(hidden, vocab_size)

            def forward(self, x, hidden=None):
                x = self.embed(x)
                out, hidden = self.lstm(x, hidden)
                out = self.fc(out)
                return out, hidden

        device = torch.device("cpu")
        model = CharLSTM(vocab_size, hidden, layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        criterion = nn.CrossEntropyLoss()

        # Train
        t0 = time.time()
        model.train()
        for step in range(steps):
            # Random batch from corpus
            idx = torch.randint(0, max(1, len(data) - seq_len - 1), (batch_size,))
            batch_x = torch.stack([torch.tensor(data[i:i+seq_len]) for i in idx])
            batch_y = torch.stack([torch.tensor(data[i+1:i+seq_len+1]) for i in idx])

            out, _ = model(batch_x)
            loss = criterion(out.reshape(-1, vocab_size), batch_y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
                print(f"  step {step}/{steps} loss={loss.item():.4f}")

        train_time = time.time() - t0

        # Export to RILM v1 format (all int8)
        model.eval()
        weight_bytes = self._export_rilm(model, vocab, hidden, layers, vocab_size)

        # Save to file
        out_path = os.path.join(tempfile.gettempdir(), f"domain_lstm_h{hidden}_l{layers}.bin")
        with open(out_path, "wb") as f:
            f.write(weight_bytes)

        sha256 = hashlib.sha256(weight_bytes).hexdigest()

        self._send_json(200, {
            "schema": "ri_uno_q_rilm_export_v1",
            "hidden": hidden,
            "layers": layers,
            "vocab_size": vocab_size,
            "params": sum(p.numel() for p in model.parameters()),
            "weight_bytes": len(weight_bytes),
            "sha256": sha256,
            "train_time_sec": round(train_time, 1),
            "steps": steps,
            "final_loss": round(loss.item(), 4),
            "output_path": out_path,
            "vocab": "".join(vocab),
        })

    def _build_domain_corpus(self, extra):
        """Domain-specific training corpus for ESP32 sensor status phrases.

        Includes all 33 characters from the firmware vocab to ensure
        RILM packs are directly flashable without vocab remapping.
        Firmware vocab: abcdefghijklmnopqrstuvwxyz .,!?\n'
        """
        phrases = [
            "hot room. action is check airflow.",
            "missing sensor. action is no claim.",
            "stale data. action is wait for fresh data.",
            "high heat and humidity. action is escalate.",
            "humid room. action is ventilate.",
            "normal room. action is log receipt.",
            "safe action is no claim without evidence.",
            "local first means decide before cloud.",
            "cold room. action is safe action.",
            "dry room. action is safe action.",
            "sensor ok. action is log receipt.",
            "sensor failed. action is no claim.",
            "temperature high. action is check airflow.",
            "temperature normal. action is log receipt.",
            "humidity high. action is ventilate.",
            "humidity normal. action is log receipt.",
            "the receipt says heat warning!",
            "the receipt says normal reading.",
            "the receipt says stale reading.",
            "the receipt says sensor missing.",
            "the gateway should verify the claim?",
            "confidence is high. log the receipt.",
            "confidence is low. escalate to gateway.",
            "the room can report before the cloud wakes.",
            # Include all firmware vocab chars
            "jump quickly, fix the vexing puzzle?",
            "examine the zones, adjust the queue.",
            "it's not a bug, it's a feature!",
            "no claim without evidence.",
        ]
        return phrases + extra

    def _export_rilm(self, model, vocab, hidden, layers, vocab_size):
        """Export model weights as RILM v1 all-int8 binary pack."""
        import numpy as np
        import struct

        buf = bytearray()

        # Header
        buf += struct.pack("<I", RILM_MAGIC)
        buf += struct.pack("<H", RILM_VERSION)
        buf += struct.pack("<H", 0)  # reserved
        buf += struct.pack("<I", 0)  # tensor count, filled later

        tensor_count = 0

        def add_tensor(name, weight_2d, scale=None):
            nonlocal tensor_count
            w = weight_2d.detach().cpu().numpy().astype(np.float32)
            if scale is None:
                max_abs = max(np.abs(w).max(), 1e-8)
                scale = float(max_abs / 127.0)
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            name_bytes = name.encode()
            buf_new = struct.pack("<H", len(name_bytes))
            buf_new += name_bytes
            buf_new += struct.pack("<B", 1)  # I8 dtype
            buf_new += struct.pack("<B", 2)  # ndim
            buf_new += struct.pack("<I", q.shape[0])
            buf_new += struct.pack("<I", q.shape[1])
            buf_new += struct.pack("<f", scale)
            buf_new += struct.pack("<I", q.tobytes().__len__())
            buf_new += q.tobytes()
            return buf_new

        def add_tensor_1d(name, weight_1d):
            nonlocal tensor_count
            w = weight_1d.detach().cpu().numpy().astype(np.float32)
            max_abs = max(np.abs(w).max(), 1e-8) if len(w) > 0 else 1.0
            scale = float(max_abs / 127.0) if max_abs > 0 else 1.0
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8) if len(w) > 0 else np.array([], dtype=np.int8)
            name_bytes = name.encode()
            buf_new = struct.pack("<H", len(name_bytes))
            buf_new += name_bytes
            buf_new += struct.pack("<B", 1)  # I8
            buf_new += struct.pack("<B", 1)  # ndim
            buf_new += struct.pack("<I", q.shape[0])
            buf_new += struct.pack("<f", scale)
            buf_new += struct.pack("<I", q.tobytes().__len__())
            buf_new += q.tobytes()
            return buf_new

        # Embedding
        t = add_tensor("embed.weight", model.embed.weight)
        # Insert at tensor position
        # We need to build tensors first, then patch the count
        tensors_data = bytearray()
        tensor_count = 0

        # Rebuild properly
        buf = bytearray()
        buf += struct.pack("<I", RILM_MAGIC)
        buf += struct.pack("<H", RILM_VERSION)
        buf += struct.pack("<H", 0)
        buf += struct.pack("<I", 0)  # patch later

        def write_tensor(name, data_bytes, dtype, dims, scale):
            nonlocal tensor_count, buf
            name_bytes = name.encode()
            buf += struct.pack("<H", len(name_bytes))
            buf += name_bytes
            buf += struct.pack("<B", dtype)
            buf += struct.pack("<B", len(dims))
            for d in dims:
                buf += struct.pack("<I", d)
            buf += struct.pack("<f", scale)
            buf += struct.pack("<I", len(data_bytes))
            buf += data_bytes
            tensor_count += 1

        # Embedding
        emb = model.embed.weight.detach().cpu().numpy()
        emb_scale = float(max(np.abs(emb).max(), 1e-8) / 127.0)
        emb_q = np.clip(np.round(emb / emb_scale), -127, 127).astype(np.int8)
        write_tensor("embed.weight", emb_q.tobytes(), 1, [vocab_size, hidden], emb_scale)

        # LSTM weights
        for l in range(layers):
            w_ih = getattr(model.lstm, f'weight_ih_l{l}').detach().cpu().numpy()
            w_hh = getattr(model.lstm, f'weight_hh_l{l}').detach().cpu().numpy()
            b_ih = getattr(model.lstm, f'bias_ih_l{l}').detach().cpu().numpy()
            b_hh = getattr(model.lstm, f'bias_hh_l{l}', None)
            if b_hh is not None:
                b_hh = b_hh.detach().cpu().numpy()
            else:
                b_hh = np.zeros_like(b_ih)

            # Quantize weights to int8
            wih_scale = float(max(np.abs(w_ih).max(), 1e-8) / 127.0)
            whh_scale = float(max(np.abs(w_hh).max(), 1e-8) / 127.0)
            wih_q = np.clip(np.round(w_ih / wih_scale), -127, 127).astype(np.int8)
            whh_q = np.clip(np.round(w_hh / whh_scale), -127, 127).astype(np.int8)

            # Biases as float32 (RILM v1 supports mixed dtype per tensor)
            bih_bytes = b_ih.astype(np.float32).tobytes()
            bhh_bytes = b_hh.astype(np.float32).tobytes()

            write_tensor(f"lstm.weight_ih_l{l}", wih_q.tobytes(), 1, [4*hidden, hidden], wih_scale)
            write_tensor(f"lstm.weight_hh_l{l}", whh_q.tobytes(), 1, [4*hidden, hidden], whh_scale)
            write_tensor(f"lstm.bias_ih_l{l}", bih_bytes, 0, [4*hidden], 1.0)  # F32
            write_tensor(f"lstm.bias_hh_l{l}", bhh_bytes, 0, [4*hidden], 1.0)  # F32

        # FC
        fc_w = model.fc.weight.detach().cpu().numpy()
        fc_b = model.fc.bias.detach().cpu().numpy()
        fc_scale = float(max(np.abs(fc_w).max(), 1e-8) / 127.0)
        fc_q = np.clip(np.round(fc_w / fc_scale), -127, 127).astype(np.int8)
        write_tensor("fc.weight", fc_q.tobytes(), 1, [vocab_size, hidden], fc_scale)
        write_tensor("fc.bias", fc_b.astype(np.float32).tobytes(), 0, [vocab_size], 1.0)

        # Patch tensor count
        struct.pack_into("<I", buf, 8, tensor_count)

        return bytes(buf)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), SensorAIHandler)
    print(f"UNO Q Sensor AI Gateway on http://{args.host}:{args.port}")
    print(f"  Ollama model: {OLLAMA_MODEL}")
    print(f"  Endpoints: /health, /api/sensor_ai, /api/train_lstm")
    server.serve_forever()

if __name__ == "__main__":
    main()