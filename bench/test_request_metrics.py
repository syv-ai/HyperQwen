#!/usr/bin/env python3
"""CPU launcher regressions for metrics with and without speculative decoding."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SINGLE = "single-user/start_qwen.sh"
ALTERNATIVE = "single-user/alternative.sh"
BATCH = "batch/start_qwen.sh"


class RequestMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="hyperqwen-metrics-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        for name in ["launcher_common.sh", "resolve_config.sh", "resolve_api_key.sh"]:
            shutil.copyfile(REPO/name, cls.root/name)
        for directory in ["single-user", "batch"]:
            shutil.copytree(REPO/directory, cls.root/directory)
        binary = cls.root/"venv/bin/vllm"
        binary.parent.mkdir(parents=True)
        binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        binary.chmod(0o755)
        draft = cls.root/"models/Qwen3.8-27B-DFlash2-W4A16"
        draft.mkdir(parents=True)
        (draft/"model.safetensors").touch()

    def launch(self, launcher, spec, metrics="1", detailed="0"):
        env = dict(os.environ, MODEL="", SPEC=spec, CTX="fast", KV="fp8", PTH_FA="0", FA2_FP8KV="0",
                   VISION="0", PRINT_ARGV="1", EXTRA_ARGS="", KV_OFFLOAD_GB="0", KV_MEM="",
                   VLLM_API_KEY="fixture", VLLM_OFFLOAD_KEEP_SHM="1", REQ_METRICS=metrics,
                   REQ_METRICS_DETAILED=detailed)
        result = subprocess.run(["bash", launcher], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result.stdout.splitlines()

    def test_non_speculative_profiles_keep_timing_without_spec_stats(self):
        for launcher in [SINGLE, ALTERNATIVE, BATCH]:
            for detailed in ["0", "1"]:
                with self.subTest(launcher=launcher, detailed=detailed):
                    # SPEC is ignored by batch; it must not enable spec metrics there.
                    args = self.launch(launcher, "dflash2" if launcher==BATCH else "off", detailed=detailed)
                    self.assertIn("--enable-per-request-metrics", args)
                    self.assertIn("--enable-force-include-usage", args)
                    self.assertNotIn("--speculative-config", args)
                    self.assertNotIn("--per-request-spec-decode-metrics", args)

    def test_speculative_profiles_keep_summary_and_detailed(self):
        for launcher, spec in [(SINGLE,"mtp"),(SINGLE,"dflash2"),(ALTERNATIVE,"dflash2")]:
            for detailed, level in [("0","summary"),("1","detailed")]:
                with self.subTest(launcher=launcher, spec=spec, detailed=detailed):
                    args = self.launch(launcher,spec,detailed=detailed)
                    self.assertIn("--speculative-config",args)
                    index = args.index("--per-request-spec-decode-metrics")
                    self.assertEqual(args[index+1],level)
                    self.assertIn("--enable-per-request-metrics",args)

    def test_none_alias_matches_off_with_request_metrics(self):
        for launcher in [SINGLE, ALTERNATIVE]:
            for detailed in ["0", "1"]:
                with self.subTest(launcher=launcher, detailed=detailed):
                    args = self.launch(launcher, "none", detailed=detailed)
                    self.assertNotIn("--speculative-config", args)
                    self.assertNotIn("--per-request-spec-decode-metrics", args)
                    self.assertEqual(args, self.launch(launcher, "off", detailed=detailed))

    def test_disabled_metrics_stay_disabled_even_with_detailed_selected(self):
        for launcher in [SINGLE,ALTERNATIVE,BATCH]:
            with self.subTest(launcher=launcher):
                args = self.launch(launcher,"dflash2",metrics="0",detailed="1")
                self.assertNotIn("--enable-per-request-metrics",args)
                self.assertNotIn("--enable-force-include-usage",args)
                self.assertNotIn("--per-request-spec-decode-metrics",args)


if __name__=="__main__":
    unittest.main()
