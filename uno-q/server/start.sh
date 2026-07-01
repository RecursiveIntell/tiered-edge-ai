#!/usr/bin/env bash
# Run this on the UNO Q (or here for host test) to start the inference server.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt --break-system-packages 2>/dev/null || pip install -q -r requirements.txt
exec python3 server.py --model models/heavy_model.tflite --port 8080
