#!/usr/bin/env python3
"""
Tiered Edge AI — UNO Q Inference Server (Tier 1)

Runs on the Arduino UNO Q 4GB (Qualcomm QRB2210, Debian Linux).
Receives framed wake signals from the ESP32-S3 sentinel over WiFi,
runs the heavy model at higher precision, returns JSON results.

Protocol:
  - Listens on port 8080
  - Receives: 4-byte magic "TEAI" + 4-byte little-endian length + payload
  - Payload is a FeatureTransfer binary message
  - Runs inference using TensorFlow Lite / ONNX Runtime / mock
  - Returns inference result as JSON

Usage:
  python3 server.py [--model path/to/model.tflite] [--port 8080]
"""

import socket
import struct
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("tiered-ai-uno-q")

# Magic bytes for the wake signal protocol
MAGIC = b"TEAI"

@dataclass
class FeatureTransfer:
    """Feature vector received from the ESP32-S3 sentinel"""
    sentinel_prediction: list
    sentinel_confidence: float
    features: list
    timestamp_ms: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "FeatureTransfer":
        """Deserialize the binary wake signal from ESP32-S3"""
        if len(data) < 8:
            raise ValueError(f"Data too short for FeatureTransfer: {len(data)} bytes")

        magic = data[0:4]
        if magic != MAGIC:
            raise ValueError(f"Bad magic bytes: {magic!r} (expected {MAGIC!r})")

        pos = 4
        pred_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        prediction = []
        for _ in range(pred_len):
            prediction.append(struct.unpack_from("<f", data, pos)[0])
            pos += 4

        confidence = struct.unpack_from("<f", data, pos)[0]
        pos += 4

        feat_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        features = []
        for _ in range(feat_len):
            features.append(struct.unpack_from("<f", data, pos)[0])
            pos += 4

        timestamp_ms = struct.unpack_from("<Q", data, pos)[0]

        return cls(
            sentinel_prediction=prediction,
            sentinel_confidence=confidence,
            features=features,
            timestamp_ms=timestamp_ms,
        )

    def to_bytes(self) -> bytes:
        """Serialize back to bytes for round-trip testing"""
        b = bytearray()
        b.extend(MAGIC)
        b.extend(struct.pack("<I", len(self.sentinel_prediction)))
        for v in self.sentinel_prediction:
            b.extend(struct.pack("<f", v))
        b.extend(struct.pack("<f", self.sentinel_confidence))
        b.extend(struct.pack("<I", len(self.features)))
        for v in self.features:
            b.extend(struct.pack("<f", v))
        b.extend(struct.pack("<Q", self.timestamp_ms))
        return bytes(b)


@dataclass
class InferenceResult:
    """Result sent back to ESP32-S3"""
    prediction: list
    confidence: float
    inference_time_ms: float
    model_name: str
    tier: str = "tier1"

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")


class InferenceEngine:
    """Abstract inference engine — swap implementations here"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._backend = "mock"
        self._load_model()

    def _load_model(self):
        """Load the model. Tries TFLite first, then ONNX, then mock."""
        model_path = Path(self.model_path)

        if not model_path.exists():
            log.warning(f"Model file not found: {model_path}")
            log.info("Running in mock mode — will return sine(predictions)")
            return

        # Try TensorFlow Lite (new Google package name first)
        for tflite_pkg in ["ai_edge_litert.interpreter", "tflite_runtime.interpreter"]:
            try:
                tflite = __import__(tflite_pkg, fromlist=["Interpreter"])
                log.info(f"Loading TFLite model: {model_path}")
                self.model = tflite.Interpreter(model_path=str(model_path))
                self.model.allocate_tensors()
                self._backend = "tflite"
                log.info("TFLite model loaded successfully")
                return
            except ImportError:
                log.debug(f"{tflite_pkg} not available")
            except Exception as e:
                log.warning(f"TFLite load failed via {tflite_pkg}: {e}")

        # Try ONNX Runtime
        try:
            import onnxruntime as ort
            log.info(f"Loading ONNX model: {model_path}")
            self.model = ort.InferenceSession(str(model_path))
            self._backend = "onnx"
            log.info("ONNX model loaded successfully")
            return
        except ImportError:
            log.debug("onnxruntime not available")
        except Exception as e:
            log.warning(f"ONNX load failed: {e}")

        log.warning("No ML backend available — running in mock mode")

    def infer(self, features: list) -> InferenceResult:
        """Run inference on the feature vector"""
        start = time.monotonic()

        if self.model is None:
            # Mock mode: compute sine as placeholder
            import math
            pred = [math.sin(f) for f in features]
            confidence = 0.95
        elif self._backend == "tflite":
            # TFLite inference
            import numpy as np
            import math
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            input_data = np.array(features, dtype=np.float32).reshape(input_details[0]["shape"])
            self.model.set_tensor(input_details[0]["index"], input_data)
            self.model.invoke()
            output = self.model.get_tensor(output_details[0]["index"])
            pred = output.flatten().tolist()
            if len(pred) > 1:
                exp_pred = [math.exp(p) for p in pred]
                total = sum(exp_pred)
                confidence = max(p / total for p in exp_pred)
            else:
                confidence = 0.95
        elif self._backend == "onnx":
            # ONNX inference
            import numpy as np
            input_name = self.model.get_inputs()[0].name
            input_data = np.array(features, dtype=np.float32).reshape(1, -1)
            output = self.model.run(None, {input_name: input_data})
            pred = output[0].flatten().tolist()
            confidence = 0.95
        else:
            raise RuntimeError("Unknown backend")

        elapsed_ms = (time.monotonic() - start) * 1000

        return InferenceResult(
            prediction=pred,
            confidence=confidence,
            inference_time_ms=elapsed_ms,
            model_name=Path(self.model_path).stem,
        )


def recv_exact(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or return what we got before close"""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def handle_connection(conn: socket.socket, engine: InferenceEngine):
    """Handle one framed request/response cycle"""
    with conn:
        # Read 4-byte magic + 4-byte length
        header = recv_exact(conn, 8)
        if len(header) < 4:
            log.warning(f"No data received (only {len(header)} bytes)")
            return

        if header[:4] != MAGIC:
            log.error(f"Bad magic bytes: {header[:4]!r}")
            error_response = json.dumps({"error": f"bad magic: {header[:4]!r}"}).encode("utf-8")
            conn.sendall(error_response)
            return

        if len(header) < 8:
            log.warning("Connection closed before length header")
            return

        payload_len = struct.unpack_from("<I", header, 4)[0]
        if payload_len > 4096 or payload_len < 16:
            log.error(f"Invalid payload length: {payload_len}")
            error_response = json.dumps({"error": f"invalid payload length {payload_len}"}).encode("utf-8")
            conn.sendall(error_response)
            return

        payload = recv_exact(conn, payload_len)
        if len(payload) != payload_len:
            log.warning(f"Incomplete payload: got {len(payload)}/{payload_len}")
            return

        log.debug(f"raw payload ({len(payload)} bytes): {payload!r}")

        try:
            transfer = FeatureTransfer.from_bytes(payload)
            log.info(
                f"Wake signal: confidence={transfer.sentinel_confidence:.3f}, "
                f"features={transfer.features}, "
                f"sentinel_pred={transfer.sentinel_prediction}"
            )

            result = engine.infer(transfer.features)
            log.info(
                f"Heavy inference: pred={result.prediction}, "
                f"conf={result.confidence:.3f}, "
                f"time={result.inference_time_ms:.1f}ms"
            )

            response = result.to_json()
            # Prefix response with length so client can read it cleanly
            framed = struct.pack("<I", len(response)) + response
            conn.sendall(framed)

        except Exception as e:
            log.error(f"Error processing wake signal: {e}")
            error_response = json.dumps({"error": str(e)}).encode("utf-8")
            framed = struct.pack("<I", len(error_response)) + error_response
            conn.sendall(framed)


def run_server(port: int, model_path: str):
    """Run the inference server"""
    engine = InferenceEngine(model_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(5)

    log.info("=== Tiered Edge AI — UNO Q Inference Server ===")
    log.info(f"Listening on 0.0.0.0:{port}")
    log.info(f"Model: {model_path}")
    log.info(f"Backend: {engine._backend}")

    try:
        while True:
            try:
                conn, addr = sock.accept()
                log.info(f"Connection from {addr[0]}:{addr[1]}")
                handle_connection(conn, engine)
            except KeyboardInterrupt:
                log.info("Shutting down server...")
                break
            except Exception as e:
                log.error(f"Server error: {e}")
    finally:
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiered Edge AI UNO Q Server")
    parser.add_argument(
        "--model", "-m",
        default="models/heavy_model.tflite",
        help="Path to the model file (TFLite or ONNX)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int, default=8080,
        help="Port to listen on (default: 8080)"
    )
    args = parser.parse_args()

    run_server(args.port, args.model)
