#!/usr/bin/env python3
"""CPU-only launcher checks for Vision limits, tower offload and boolean validation.

Runs all three launchers against a stub vllm: no GPU, model, dependencies or network.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
LAUNCHERS = ("single-user/start_qwen.sh", "batch/start_qwen.sh", "single-user/alternative.sh")


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="hyperqwen-vision-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.repo = Path(cls.tmp.name)
        for name in ("launcher_common.sh", "resolve_config.sh", "resolve_api_key.sh"):
            shutil.copyfile(REPO / name, cls.repo / name)
        for name in ("single-user", "batch"):
            shutil.copytree(REPO / name, cls.repo / name)
        binary = cls.repo / "venv/bin/vllm"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'#!/bin/sh\nprintf "OFFLOAD=%s\\n" "${VLLM_VISION_CPU_OFFLOAD_GB-unset}"\nprintf "%s\\n" "$@"\n')
        binary.chmod(0o755)
        draft = cls.repo / "models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"
        draft.parent.mkdir(parents=True)
        draft.touch()  # only the stub runs; the launcher just checks the drafter exists

    def run_launcher(self, launcher, **knobs):
        env = dict(os.environ)
        for name in ("VISION", "VISION_OFFLOAD", "VLLM_VISION_CPU_OFFLOAD_GB", "PRINT_ARGV", "EXTRA_ARGS",
                     "KV_OFFLOAD_GB", "MODEL", "DRAFT", "CUDA_HOME", "KV_MEM"):
            env.pop(name, None)
        env.update(SPEC="off", CTX="fast", KV="fp8", VLLM_API_KEY="test-key", FA2_FP8KV="0", PTH_FA="0",
                   VLLM_OFFLOAD_KEEP_SHM="1")
        env.update(knobs)
        return subprocess.run(["bash", launcher], cwd=self.repo, env=env,
                              capture_output=True, text=True, timeout=30)

    def argv(self, launcher, **knobs):
        result = self.run_launcher(launcher, **knobs)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.splitlines()

    def test_disabled_vision_is_language_model_only(self):
        for launcher in LAUNCHERS:
            for knobs in ({}, {"VISION": ""}, {"VISION": "0"}):
                with self.subTest(launcher=launcher, knobs=knobs):
                    argv = self.argv(launcher, **knobs)
                    self.assertIn("--language-model-only", argv)
                    self.assertNotIn("--mm-processor-kwargs", argv)

    def test_enabled_vision_limits_images_and_offloads_the_tower(self):
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher):
                argv = self.argv(launcher, VISION="1")
                self.assertNotIn("--language-model-only", argv)
                self.assertIn("OFFLOAD=1", argv)
                self.assertIn("--limit-mm-per-prompt", argv)
                self.assertEqual(json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]), {"image": {"count": 1}})
                self.assertEqual(json.loads(argv[argv.index("--mm-processor-kwargs") + 1]),
                                 {"size": {"shortest_edge": 65536, "longest_edge": 2097152}})

    def test_offload_zero_overrides_an_inherited_budget(self):
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher):
                argv = self.argv(launcher, VISION="1", VISION_OFFLOAD="0", VLLM_VISION_CPU_OFFLOAD_GB="2")
                self.assertIn("OFFLOAD=0", argv)

    def test_offload_budget_and_empty_default(self):
        for launcher in LAUNCHERS:
            for offload in ("", "1"):
                with self.subTest(launcher=launcher, offload=offload):
                    argv = self.argv(launcher, VISION="1", VISION_OFFLOAD=offload, VLLM_VISION_CPU_OFFLOAD_GB="0.5")
                    self.assertIn("OFFLOAD=0.5", argv)

    def test_dflash2_vision_uses_the_same_limits_and_offload(self):
        for launcher in (LAUNCHERS[0], LAUNCHERS[2]):
            with self.subTest(launcher=launcher):
                argv = self.argv(launcher, SPEC="dflash2", VISION="1")
                self.assertIn("OFFLOAD=1", argv)
                self.assertIn("--limit-mm-per-prompt", argv)
                config = json.loads(argv[argv.index("--speculative-config") + 1])
                self.assertEqual(config["method"], "dflash")

    def test_boolean_typos_are_refused(self):
        for launcher in LAUNCHERS:
            for knob in ("VISION", "VISION_OFFLOAD"):
                for value in ("true", "false", "2"):
                    with self.subTest(launcher=launcher, knob=knob, value=value):
                        result = self.run_launcher(launcher, **{knob: value})
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(f"{knob}={value}", result.stderr)

    def test_extra_args_keep_the_last_limits(self):
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher):
                limits = '{"image":{"count":2}}'
                argv = self.argv(launcher, VISION="1", EXTRA_ARGS="--limit-mm-per-prompt " + limits)
                values = [argv[i + 1] for i, value in enumerate(argv) if value == "--limit-mm-per-prompt"]
                self.assertEqual(values[-1], limits)


    def test_launcher_alias_boolean_typos_are_rejected(self):
        for launcher in LAUNCHERS:
            for knob in ("LOOKUP", "INT4_MQ_3D", "TOOLS", "EMBED_UVA"):
                with self.subTest(launcher=launcher, knob=knob):
                    result = self.run_launcher(launcher, **{knob: "true"})
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(f"{knob}=true", result.stderr)
                    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
