#!/usr/bin/env python3
"""CPU-only checks for doctor health failures and redacted diagnostics."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class DoctorTests(unittest.TestCase):
    def run_doctor(self, code, status, config_status=0, large=False, env_key=False):
        with tempfile.TemporaryDirectory(prefix="hq-doctor-") as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copyfile(REPO / "scripts/hq-doctor.sh", root / "scripts/hq-doctor.sh")
            (root / "resolve_config.sh").write_text(
                f'resolve_effective_config() {{ return {config_status}; }}\n', encoding="utf-8")
            binary = root / "bin"
            binary.mkdir()
            if env_key:
                (root / ".env").write_text("VLLM_API_KEY=secret-from-env-file\n", encoding="utf-8")
            curl = ('''connect_timeout= max_time=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --connect-timeout|--max-time)
            [ "$#" -ge 2 ] || exit 99
            case "$1" in
                --connect-timeout) connect_timeout=$2;;
                --max-time) max_time=$2;;
            esac
            shift 2;;
        *) shift;;
    esac
done
[ "$connect_timeout" = 5 ] && [ "$max_time" = 10 ] || exit 99
'''
                    + f'printf %s {code}; exit {status}')
            ss = "printf 'LISTEN 0 128 127.0.0.1:18020 0.0.0.0:*\\n'; printf '%0400000d\\n' 0"
            docker = ('case "$1" in ps) printf "hyperqwen-single-1\\n"; printf "%0400000d\\n" 0;; '
                      'logs) printf "ERROR fixture container failure\\n";; esac')
            for name, body in {"curl": curl, "docker": docker if large else "exit 1",
                               "ss": ss if large else "exit 1", "nvidia-smi": "exit 1"}.items():
                path = binary / name
                path.write_bytes(("#!/bin/sh\n" + body + "\n").encode())
                path.chmod(0o755)
            # Git Bash needs its own POSIX PATH for a Windows temporary directory.
            script = 'export PATH="$PWD/bin:$PATH"; bash scripts/hq-doctor.sh'
            return subprocess.run(["bash", "-c", script], cwd=root, capture_output=True, text=True,
                                  env=dict(os.environ, VLLM_API_KEY="test-value", PORT="18020"), timeout=15)

    def test_failed_curl_is_no_server(self):
        for code in ("000", ""):
            with self.subTest(code=code):
                result = self.run_doctor(code, 7)
                self.assertEqual(result.returncode, 1)
                self.assertIn("doctor: no server on", result.stdout)
                self.assertNotIn("HTTP 000000", result.stdout)

    def test_http_failure_is_not_ready(self):
        result = self.run_doctor("503", 0)
        self.assertEqual(result.returncode, 1)
        self.assertIn("answers HTTP 503", result.stdout)

    def test_health_200_passes(self):
        result = self.run_doctor("200", 0)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("test-value", result.stdout + result.stderr)

    def test_bad_config_is_reported_through_the_pipeline(self):
        result = self.run_doctor("200", 0, config_status=1)
        self.assertIn("unvalidated config", result.stdout)

    def test_large_listings_do_not_hide_port_or_container(self):
        result = self.run_doctor("503", 0, large=True)
        self.assertIn("occupant:", result.stdout)
        self.assertIn("fixture container failure", result.stdout)
        self.assertNotIn("free (nothing listening)", result.stdout)

    def test_timeout_is_not_reported_as_no_server(self):
        result = self.run_doctor("", 28)
        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stdout)
        self.assertIn("do NOT restart mid-compile", result.stdout)
        self.assertNotIn("doctor: no server on", result.stdout)

    def test_env_file_key_is_redacted(self):
        result = self.run_doctor("200", 0, env_key=True)
        self.assertNotIn("secret-from-env-file", result.stdout + result.stderr)
        self.assertIn("chars, value not shown", result.stdout)


if __name__ == "__main__":
    unittest.main()
