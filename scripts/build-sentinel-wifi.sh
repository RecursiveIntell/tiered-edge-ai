#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="$repo_root/esp32-s3/.env.local"

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

cd "$repo_root/esp32-s3"
exec cargo +esp build --target xtensa-esp32s3-none-elf --bin sentinel-wifi --release -Z build-std=core,alloc

