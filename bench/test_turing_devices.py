#!/usr/bin/env python3
"""CPU fixtures for the Turing wrapper's visible-device selection."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TuringDeviceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copyfile(Path(__file__).resolve().parents[1]/"start-turing.sh", self.root/"start-turing.sh")
        scripts = {
            "venv/bin/python": '''#!/bin/sh
test "$1" = -c || exit 77
case "$2" in *torch.cuda.device_count*) ;; *) exit 78 ;; esac
case "${CUDA_VISIBLE_DEVICES-unset}" in
  unset) echo 2 ;; 1|GPU-one|MIG-one) echo 1 ;; 1,0) echo 2 ;; ''|-1) echo 0 ;; *) exit 79 ;;
esac
''',
            "bin/nvidia-smi": '#!/bin/sh\nprintf "GPU 0: fixture\\nGPU 1: fixture\\n"\n',
            "single-user/start_qwen.sh": '#!/bin/sh\nprintf "CTX_LEN=%s\\nARGV=%s\\n" "$MAX_LEN" "$EXTRA_ARGS"\n',
        }
        for name, content in scripts.items():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            path.chmod(0o755)

    def launch(self, visible=None, tp=""):
        env = dict(os.environ, TP=tp, MAX_LEN="", HYPERQWEN_LOG_FILE=str(self.root/"test.log"), EXTRA_ARGS="")
        env["PATH"] = str(self.root/"bin") + os.pathsep + env["PATH"]
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible
        return subprocess.run(["bash", "start-turing.sh"], cwd=self.root, env=env, capture_output=True, text=True)

    def test_single_visible_device_uses_tp_one_and_small_context(self):
        for visible in ("1", "GPU-one", "MIG-one"):
            with self.subTest(visible=visible):
                result = self.launch(visible)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("CTX_LEN=32768", result.stdout)
                self.assertNotIn("--tensor-parallel-size", result.stdout)

    def test_two_visible_devices_keep_tp_two_defaults(self):
        for visible in (None, "1,0"):
            with self.subTest(visible=visible):
                result = self.launch(visible)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("CTX_LEN=262136", result.stdout)
                self.assertIn("--tensor-parallel-size 2", result.stdout)

    def test_empty_mask_refused_before_launch(self):
        for visible in ("", "-1"):
            with self.subTest(visible=visible):
                result = self.launch(visible)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("No visible CUDA devices", result.stderr)
                self.assertNotIn("ARGV=", result.stdout)

    def test_explicit_tp_overrides_detection(self):
        # A failing detector makes an accidental probe visible.
        for tp in ("1", "2", "4"):
            with self.subTest(tp=tp):
                result = self.launch("bad-fixture", tp=tp)
                self.assertEqual(result.returncode, 0, result.stderr)
                if tp != "1":
                    self.assertIn("--tensor-parallel-size " + tp, result.stdout)
        self.assertNotEqual(self.launch(tp="3").returncode, 0)


if __name__ == "__main__":
    unittest.main()
