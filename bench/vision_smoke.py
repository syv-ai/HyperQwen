#!/usr/bin/env python3
"""Check that a Vision-enabled server reads pixels, rather than answering from text alone.

Send two PNGs with different random arrangements of three large solid-color panels
and the same question. Expected color names/order live only in the pixels, never in
the text. This is a coarse Vision round-trip check, not an OCR accuracy test.
No image library, external image host or model files are needed on the client.

  VISION=1 bash single-user/start_qwen.sh   # or single-user/alternative.sh / batch/start_qwen.sh
  python bench/vision_smoke.py             # client, using bench/harness.py's URL/key

Exit 0: both color orders read correctly; 1: an incorrect/error response; 2: no server answered.
"""
import base64
import itertools
import random
import re
import struct
import urllib.error
import zlib

import harness

# Large separated RGB panels avoid OCR/glyph ambiguities on a healthy Vision tower.
COLORS = {"red": b"\xff\0\0", "green": b"\0\xff\0",
          "blue": b"\0\0\xff", "yellow": b"\xff\xff\0"}


def marker_png(marker):
    """Three different solid-color panels; no text or expected labels in PNG metadata."""
    names = marker.split()
    if len(names) != 3 or len(set(names)) != 3 or any(name not in COLORS for name in names):
        raise ValueError("marker must name three different supported colors")
    width, height = 768, 256
    pixels = bytearray(b"\xff" * width * height * 3)
    for i, name in enumerate(names):
        for y in range(32, 224):
            offset = (y * width + 32 + i * 256) * 3
            pixels[offset:offset + 192 * 3] = COLORS[name] * 192

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    raw = b"".join(b"\0" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def new_markers():
    choices = [" ".join(names) for names in itertools.permutations(COLORS, 3)]
    return random.SystemRandom().sample(choices, 2)


def main():
    reached, failures = 0, 0
    for i, marker in enumerate(new_markers(), 1):
        image = "data:image/png;base64," + base64.b64encode(marker_png(marker)).decode("ascii")
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Name the colors of the three large panels from left to right. Reply only with three English color names separated by commas."},
                {"type": "image_url", "image_url": {"url": image}},
            ]}],
            "temperature": 0, "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            reply = harness.post("/v1/chat/completions", payload, timeout=120)
        except urllib.error.HTTPError as exc:
            reached += 1
            failures += 1
            detail = exc.read(200).decode("utf-8", errors="replace")
            exc.close()
            print(f"FAIL image {i}: HTTP {exc.code} {detail}")
            continue
        except OSError as exc:
            failures += 1
            print(f"FAIL image {i}: {type(exc).__name__}")
            continue
        except ValueError:
            reached += 1
            failures += 1
            print(f"FAIL image {i}: invalid JSON response")
            continue
        reached += 1
        try:
            answer = reply["choices"][0]["message"]["content"] or ""
            colors = re.findall(r"\b(?:red|green|blue|yellow)\b", answer.lower()) if isinstance(answer, str) else []
        except (KeyError, IndexError, TypeError):
            colors = []
        if colors == marker.split():
            print(f"PASS image {i}: color order read from pixels")
        else:
            failures += 1
            print(f"FAIL image {i}: image color order not read correctly (check VISION=1 and the processor/tower)")
    if not reached:
        print("INVALID: no server answered, so there is no Vision verdict")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
