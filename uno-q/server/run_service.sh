#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs models

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

exec 9>/tmp/tiered-ai-server.lock
flock -n 9 || exit 0

exec "$python_bin" server.py --model models/heavy_model.tflite --port 8080

