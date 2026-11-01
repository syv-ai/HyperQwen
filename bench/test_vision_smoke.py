#!/usr/bin/env python3
"""CPU-only HTTP checks that Vision verification depends on pixels, not just text."""
import base64
import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import threading
import unittest
from unittest import mock
import urllib.error
import zlib

import harness
import vision_smoke

REPO = Path(__file__).resolve().parents[1]
MARKERS = ["red blue yellow", "green yellow blue"]


@contextlib.contextmanager
def stub(image_answers=(), status=200, error_body=None):
    seen = []
    answers = iter(image_answers)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = payload["messages"][0]["content"]
            image = isinstance(content, list)
            seen.append((self.path, self.headers.get("Authorization"), payload))
            answer = next(answers, "Copenhagen") if image else "Copenhagen"
            self.send_response(status)
            self.end_headers()
            body = {"choices": [{"message": {"content": answer}}]} if error_body is None else error_body
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class VerifyTests(unittest.TestCase):
    def live_chat_checks(self, url, vision):
        source = (REPO / "verify.sh").read_text(encoding="utf-8")
        start = source.index('    R=$(curl -s $SRV/v1/chat/completions')
        end = source.index('    # /tokenize contract', start)
        # Run the actual live-chat section; no installed vLLM or model is required.
        # Equivalent JSON escapes keep the Danish prompt intact across Windows' Bash -c argv encoding.
        block = "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in source[start:end])
        script = ('RC=0\nok() { printf "PASS %s\\n" "$1"; }\n'
                  'fail() { printf "FAIL %s\\n" "$1"; RC=1; }\n' + block + '\nexit "$RC"\n')
        env = dict(os.environ, SRV=url, KEY="vision-key", VISION=vision, PY=Path(sys.executable).as_posix())
        return subprocess.run(["bash", "-c", script], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=30)

    def test_text_only_success_cannot_pass_vision_verification(self):
        with stub() as (url, seen):
            result = self.live_chat_checks(url, "1")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FAIL Vision", result.stdout)
        self.assertEqual(len(seen), 3)  # one text request and two different images
        self.assertTrue(all(auth == "Bearer vision-key" for _, auth, _ in seen))
        self.assertNotEqual(seen[1][2]["messages"][0]["content"][1],
                            seen[2][2]["messages"][0]["content"][1])

    def test_vision_disabled_keeps_the_text_check(self):
        with stub() as (url, seen):
            result = self.live_chat_checks(url, "0")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(seen), 1)


class SmokeTests(unittest.TestCase):
    def run_check(self, url="http://127.0.0.1:9"):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"VLLM_API": url + "/v1", "OPENAI_API_KEY": "vision-key",
                                          "VLLM_MODEL": "vision-model"}), \
                mock.patch.object(vision_smoke, "new_markers", return_value=MARKERS), \
                contextlib.redirect_stdout(out):
            code = vision_smoke.main()
        return code, out.getvalue()

    def test_two_correct_image_answers_pass(self):
        with stub(MARKERS) as (url, seen):
            code, out = self.run_check(url)
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("PASS"), 2)
        texts, images = [], []
        for path, auth, payload in seen:
            self.assertEqual((path, auth, payload["model"]),
                             ("/v1/chat/completions", "Bearer vision-key", "vision-model"))
            text, image = payload["messages"][0]["content"]
            texts.append(text)
            for marker in MARKERS:
                self.assertNotIn(marker, text["text"])
                for color in marker.split():
                    self.assertNotIn(color, text["text"].lower())
            url = image["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            images.append(base64.b64decode(url.split(",", 1)[1], validate=True))
        self.assertEqual(texts[0], texts[1])
        self.assertNotEqual(images[0], images[1])

    def test_same_answer_for_different_images_fails(self):
        with stub([MARKERS[0], MARKERS[0]]) as (url, _):
            code, out = self.run_check(url)
        self.assertEqual(code, 1, out)
        self.assertIn("PASS image 1", out)
        self.assertIn("FAIL image 2", out)

    def test_empty_image_answers_fail(self):
        with stub([None, ""]) as (url, _):
            code, out = self.run_check(url)
        self.assertEqual(code, 1, out)

    def test_http_error_is_failure_not_no_server(self):
        reason = "At most 0 image(s) may be provided in one prompt."
        with stub(status=400, error_body={"error": reason}) as (url, _):
            code, out = self.run_check(url)
        self.assertEqual(code, 1, out)
        self.assertEqual(out.count("HTTP 400"), 2)
        self.assertIn(reason, out)

    def test_color_reply_accepts_case_spaces_and_commas(self):
        with stub(["RED, BLUE, YELLOW.", "The panels are green, yellow and blue."]) as (url, _):
            code, out = self.run_check(url)
        self.assertEqual(code, 0, out)

    def test_markers_are_different_and_have_three_distinct_colors(self):
        markers = vision_smoke.new_markers()
        self.assertEqual(len(markers), 2)
        self.assertNotEqual(*markers)
        for marker in markers:
            self.assertEqual(len(set(marker.split())), 3)
            self.assertTrue(set(marker.split()) <= vision_smoke.COLORS.keys())

    def test_no_server_is_an_invalid_run(self):
        with mock.patch.object(harness, "post", side_effect=urllib.error.URLError("refused")):
            code, out = self.run_check()
        self.assertEqual(code, 2, out)
        self.assertIn("no server answered", out)

    def test_partial_connection_failure_does_not_pass(self):
        reply = {"choices": [{"message": {"content": MARKERS[0]}}]}
        with mock.patch.object(harness, "post", side_effect=[reply, urllib.error.URLError("refused")]):
            code, out = self.run_check()
        self.assertEqual(code, 1, out)

    def test_invalid_json_is_failure_not_no_server(self):
        with mock.patch.object(harness, "post", side_effect=ValueError("invalid JSON")):
            code, out = self.run_check()
        self.assertEqual(code, 1, out)

    def test_png_has_valid_chunks_and_nonempty_pixels(self):
        png = vision_smoke.marker_png(MARKERS[0])
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        chunks, offset = {}, 8
        while offset < len(png):
            size = struct.unpack(">I", png[offset:offset + 4])[0]
            kind, data = png[offset + 4:offset + 8], png[offset + 8:offset + 8 + size]
            crc = struct.unpack(">I", png[offset + 8 + size:offset + 12 + size])[0]
            self.assertEqual(crc, zlib.crc32(kind + data))
            chunks[kind] = data
            offset += size + 12
        self.assertEqual(set(chunks), {b"IHDR", b"IDAT", b"IEND"})  # no marker text metadata
        width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
        self.assertEqual((width, height, depth, color, interlace), (768, 256, 8, 2, 0))
        pixels = zlib.decompress(chunks[b"IDAT"])
        stride = width * 3 + 1
        self.assertEqual(len(pixels), stride * height)
        self.assertTrue(all(pixels[y * stride] == 0 for y in range(height)))
        self.assertEqual(set(pixels), {0, 255})
        for i, name in enumerate(MARKERS[0].split()):
            offset = 128 * stride + 1 + (128 + 256 * i) * 3
            self.assertEqual(pixels[offset:offset + 3], vision_smoke.COLORS[name])
        self.assertEqual(pixels[1:4], b"\xff" * 3)  # white margin, no labels


if __name__ == "__main__":
    unittest.main()
