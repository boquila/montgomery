from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bench_training_matrix as matrix  # noqa: E402


class BenchmarkTrainingTests(unittest.TestCase):
    def test_observer_shutdown_is_outside_command_wall_time(self) -> None:
        class SlowStopMonitor:
            def __init__(self, _pid: int, _interval: float) -> None:
                pass

            def start(self) -> None:
                pass

            def stop(self) -> dict:
                time.sleep(0.15)
                return {}

        with patch.object(matrix, "ResourceMonitor", SlowStopMonitor):
            started = time.perf_counter()
            _, wall_seconds, _ = matrix.run_observed(
                [sys.executable, "-c", "import time; time.sleep(0.05)"], 0.1
            )
            total_seconds = time.perf_counter() - started
        self.assertGreater(total_seconds - wall_seconds, 0.12)

    def test_failed_command_resources_are_persisted(self) -> None:
        scenario = matrix.SCENARIOS[0]
        command = [
            sys.executable,
            "-c",
            "import sys,time; x=bytearray(1024*1024); time.sleep(.2); sys.exit(7)",
        ]
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            output_root = Path(directory)
            with patch.object(matrix, "command_for", return_value=command):
                with self.assertRaises(matrix.ObservedCommandError) as raised:
                    matrix.run_once(
                        "native",
                        scenario,
                        output_root,
                        "failure-test",
                        resource_interval_seconds=0.05,
                    )
            result = {"schema": "test", "scenarios": [], "failures": []}
            output = output_root / "results.json"
            matrix.persist_failure(result, output, raised.exception)
            failure = json.loads(output.read_text(encoding="utf-8"))["failures"][0]
            self.assertEqual(failure["returncode"], 7)
            self.assertGreaterEqual(failure["resources"]["sampling"]["sample_count"], 2)
            self.assertTrue(list((output_root / "logs").glob("*.resources.json")))

    def test_successful_command_output_failure_retains_resources(self) -> None:
        scenario = matrix.SCENARIOS[0]
        command = [sys.executable, "-c", "import time; time.sleep(.2)"]
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            with patch.object(matrix, "command_for", return_value=command):
                with self.assertRaises(matrix.ObservedCommandError) as raised:
                    matrix.run_once(
                        "native",
                        scenario,
                        Path(directory),
                        "output-failure-test",
                        resource_interval_seconds=0.05,
                    )
            self.assertEqual(raised.exception.record["phase"], "training-output")
            self.assertIsNotNone(raised.exception.record["resources"])

    def test_resume_rejects_different_telemetry_settings(self) -> None:
        existing = {
            "schema": "v3",
            "methodology": {"resource_monitoring": {"target_sample_interval_ms": 100.0}},
        }
        requested = {
            "schema": "v3",
            "methodology": {"resource_monitoring": {"target_sample_interval_ms": 200.0}},
        }
        with self.assertRaisesRegex(ValueError, "different benchmark settings"):
            matrix.validate_resume_compatibility(existing, requested)


if __name__ == "__main__":
    unittest.main()
