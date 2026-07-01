#!/usr/bin/env python3
import socket, struct, argparse, json, sys

MAGIC = b"TEAI"

def build_payload(prediction, confidence, features, timestamp_ms=12345):
    b = bytearray()
    b.extend(MAGIC)
    b.extend(struct.pack("<I", len(prediction)))
    for v in prediction:
        b.extend(struct.pack("<f", v))
    b.extend(struct.pack("<f", confidence))
    b.extend(struct.pack("<I", len(features)))
    for v in features:
        b.extend(struct.pack("<f", v))
    b.extend(struct.pack("<Q", timestamp_ms))
    return bytes(b)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--confidence", type=float, default=0.6)
    args = p.parse_args()

    payload = build_payload(prediction=[0.5, -0.3], confidence=args.confidence,
                            features=[1.0, 2.0, 3.0])
    # Frame: MAGIC + length + payload
    pkt = MAGIC + struct.pack("<I", len(payload)) + payload

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5.0)
        s.connect((args.host, args.port))
        s.sendall(pkt)
        # Read length-prefixed JSON response
        raw_len = s.recv(4)
        if len(raw_len) < 4:
            print("no response length")
            sys.exit(1)
        resp_len = struct.unpack("<I", raw_len)[0]
        resp = b""
        while len(resp) < resp_len:
            chunk = s.recv(resp_len - len(resp))
            if not chunk:
                break
            resp += chunk
    print("sent", len(pkt), "bytes")
    print("received", resp.decode("utf-8", errors="replace"))

if __name__ == "__main__":
    main()
