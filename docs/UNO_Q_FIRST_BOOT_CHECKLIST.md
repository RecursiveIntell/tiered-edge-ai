# UNO Q 4GB First Boot + Test Checklist

## What you need before plugging in
1. Arduino UNO Q 4GB board
2. USB-C cable with data+power (not charge-only)
3. Your WiFi credentials ready in `esp32-s3/.env.local` (copy from `esp32-s3/.env.example`; do not commit the local file)
4. A computer or the UNO Q itself to run the server

## Step 1: Power on and identify the board
- Plug USB-C into the UNO Q power/data port.
- Wait ~30 seconds for Debian Linux to boot.
- The board should appear as a USB Ethernet/RNDIS gadget or serial device on your PC.
- If using the UNO Q standalone, connect a keyboard + HDMI/USB-C display, or access via the Arduino Cloud / IDE serial monitor.

## Step 2: Get the UNO Q IP address
Option A: connect a display and run `ip addr` in a terminal.
Option B: from your router admin page, find the device named like `uno-q`, `arduino`, or the MAC address.
Option C: scan your LAN:
  nmap -sn 192.168.1.0/24
Note the IP and set `TIERED_UNO_Q_IP` in:
  /home/sikmindz/projects/tiered-edge-ai/esp32-s3/.env.local

For this board, ADB is available over USB and unprivileged SSH is configured on port 2222:
  ssh -i ~/.ssh/roko -p 2222 arduino@<UNO_Q_IP>

Port 22 is controlled by the system ssh service and needs sudo/password access to enable.

## Step 3: Copy server files to the UNO Q
From your PC, scp the server directory to the UNO Q:
  scp -r /home/sikmindz/projects/tiered-edge-ai/uno-q/server user@<UNO_Q_IP>:~/tiered-ai-server
Or, if the UNO Q runs headless and you have a shared NFS/USB, place the files at a known path.

## Step 4: Install Python deps and start the server
SSH into the UNO Q:
  ssh user@<UNO_Q_IP>
  cd ~/tiered-ai-server
  pip3 install -r requirements.txt --break-system-packages   # or use the venv in start.sh
  python3 server.py --model models/heavy_model.tflite --port 8080

You should see:
  === Tiered Edge AI — UNO Q Inference Server ===
  Listening on 0.0.0.0:8080
  Backend: mock

(Mock mode is fine for the first loop test; we have no heavy model yet.)

For persistent startup, install `tiered-ai-server.service` as a user service when a user systemd bus is available:
  mkdir -p ~/.config/systemd/user
  cp ~/tiered-ai-server/tiered-ai-server.service ~/.config/systemd/user/
  systemctl --user daemon-reload
  systemctl --user enable --now tiered-ai-server.service

If the user systemd bus is not available over ADB, use the unprivileged cron fallback:
  (crontab -l 2>/dev/null | grep -v tiered-ai-server; echo '@reboot /home/arduino/tiered-ai-server/run_service.sh >> /home/arduino/tiered-ai-server/logs/server.log 2>&1') | crontab -

This board is currently using the cron fallback.

## Step 5: Verify the server responds from your PC
On your PC, run the mock client:
  python3 /home/sikmindz/projects/tiered-edge-ai/uno-q/server/mock_client.py --host <UNO_Q_IP> --port 8080

Expected output:
  sent 44 bytes
  received {"prediction": [...], "confidence": 0.95, ...}

If this works, the UNO Q server path is alive.

## Step 6: Flash the ESP32-S3 sentinel with WiFi wake logic
Create `esp32-s3/.env.local` from `esp32-s3/.env.example`, then build + flash:
  ./scripts/build-sentinel-wifi.sh
  espflash flash target/xtensa-esp32s3-none-elf/release/sentinel-wifi --port /dev/ttyACM0 --non-interactive --chip esp32s3 --monitor --log-format defmt --baud 115200

## Step 7: End-to-end smoke test
With both devices on the same WiFi and the server running, power cycle the ESP32-S3.
Watch its serial log. Every ~10 cycles it feeds a hard input to trigger low confidence.
You should see:
  LOW CONFIDENCE (...) — waking UNO Q
  Wake signal sent
  UNO Q response: {"prediction": [...]}

If you see that, the pipeline is closed.

## Files modified / created for this setup
- /home/sikmindz/projects/tiered-edge-ai/uno-q/server/requirements.txt
- /home/sikmindz/projects/tiered-edge-ai/uno-q/server/mock_client.py
- /home/sikmindz/projects/tiered-edge-ai/uno-q/server/start.sh
- /home/sikmindz/projects/tiered-edge-ai/uno-q/server/server.py (patched to support ai-edge-litert + tflite_runtime)

## Blockers to watch for
- UNO Q may need initial Arduino IDE/Cloud registration before Debian shell is reachable.
- WiFi may be on a different subnet; update `TIERED_UNO_Q_IP` in `esp32-s3/.env.local` accordingly.
- Python 3.14 on host works with `ai-edge-litert` and `onnxruntime`. The UNO Q may have Python 3.11; packages should still resolve, but watch for ARM wheel availability.
- The heavy model does not exist yet; mock mode returns sine(features) as a placeholder.

## Next decisions after first loop works
1. What real heavy model should the UNO Q run? (TinyStories char-LM, keyword/vision model, etc.)
2. Should the UNO Q train/export models for the ESP32-S3, or should we keep them separate?
3. Do you want the speculative decoding path (spec-engine) integrated, or focus on tiered wake first?
