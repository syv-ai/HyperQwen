#!/usr/bin/env python3
"""CPU-only interruption/resume checks for self-distillation JSONL output."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drafter"))
from generation_state import load_completed, exclusive_output, OutputLockedError

DRAFTER = Path(__file__).resolve().parents[1] / "drafter"


class GenerationTests(unittest.TestCase):
    def test_torn_tail_and_unterminated_complete_record(self):
        for tail, want in ((b'{"id":', {1}), (b'{"id":2,"text":"\xc3', {1}),
                           (b'{"id":2}', {1, 2}), (b'{"id":2}\n', {1, 2})):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "gen.jsonl"
                prefix = b'{"id":1}\n'
                path.write_bytes(prefix + tail)
                self.assertEqual(load_completed(path), want)
                self.assertTrue(path.read_bytes().startswith(prefix))
                with path.open("ab") as f:
                    f.write(b'{"id":3}\n')
                records = [json.loads(line) for line in path.read_bytes().splitlines()]
                self.assertEqual({r['id'] for r in records}, want | {3})
                before = path.read_bytes()
                self.assertEqual(load_completed(path), want | {3})
                self.assertEqual(path.read_bytes(), before)

    def test_internal_corruption_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            original = b'{"id":1}\nbroken\n{"id":3}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                load_completed(path)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_output_is_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            self.assertEqual(load_completed(path), set())
            self.assertFalse(path.exists())

    def test_empty_or_only_torn_output_and_invalid_record_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            for data in (b"", b'{"id":1,"out'):
                path.write_bytes(data)
                self.assertEqual(load_completed(path), set())
                self.assertEqual(path.read_bytes(), b"")
            for bad in (b'{"x":2}\n', b'2\n'):
                original = b'{"id":1}\n' + bad
                path.write_bytes(original)
                with self.assertRaisesRegex(ValueError, "no id"):
                    load_completed(path)
                self.assertEqual(path.read_bytes(), original)

    def test_second_process_cannot_repair_an_active_writers_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            partial = b'{"id":1}\n{"id":'
            with exclusive_output(path):
                path.write_bytes(partial)
                code = (f'import sys; sys.path.insert(0, {str(DRAFTER)!r})\n'
                        'from generation_state import exclusive_output, load_completed\n'
                        f'with exclusive_output({str(path)!r}): load_completed({str(path)!r})\n')
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("already being generated", result.stderr)
                self.assertEqual(path.read_bytes(), partial)
                with path.open("ab") as output:
                    output.write(b'2}\n')
            self.assertEqual(load_completed(path), {1, 2})

    def test_lock_released_after_exception_and_abrupt_process_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            with self.assertRaisesRegex(RuntimeError, "engine failed"):
                with exclusive_output(path):
                    raise RuntimeError("engine failed")
            code = (f'import os, sys; sys.path.insert(0, {str(DRAFTER)!r})\n'
                    'from generation_state import exclusive_output\n'
                    f'with exclusive_output({str(path)!r}): os._exit(0)\n')
            subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=10)
            with exclusive_output(path):
                pass

    def test_generator_refuses_before_tokenizer_or_engine_and_preserves_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen.jsonl"
            partial = b'{"id":1}\n{"id":'
            path.write_bytes(partial)
            (Path(tmp) / "prompts.jsonl").write_text('{"id":2}\n', encoding="utf-8")
            transformers, vllm = mock.Mock(), mock.Mock()
            vllm.LLM.side_effect = RuntimeError("simulated GPU OOM")
            spec = importlib.util.spec_from_file_location("tested_gen_data", DRAFTER / "gen_data.py")
            module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules, transformers=transformers, vllm=vllm):
                spec.loader.exec_module(module)
            module.D = tmp
            with exclusive_output(path), self.assertRaisesRegex(SystemExit, "already being generated"):
                module.main()
            transformers.AutoTokenizer.from_pretrained.assert_not_called()
            vllm.LLM.assert_not_called()
            self.assertEqual(path.read_bytes(), partial)


if __name__ == "__main__":
    unittest.main()
