from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.bench_training_matrix import run_observed, selected_scenarios, summarize
from tools.bench_ultralytics_train import no_validation_trainer
from tools.benchmark_resources import (
    LinuxDrmGpuMemory,
    NvmlGpuMemory,
    ResourceMonitor,
    WindowsPdhGpuMemory,
)


class ResourceMonitorTests(unittest.TestCase):
    def test_docs_matrix_selects_published_report_scope(self) -> None:
        scenarios = selected_scenarios([], docs_matrix=True)

        self.assertEqual(len(scenarios), 21)
        self.assertEqual({scenario.group for scenario in scenarios}, {"family-task", "resolution"})
        self.assertEqual(sum(scenario.imgsz == 224 for scenario in scenarios), 3)
        self.assertEqual(sum(scenario.imgsz == 640 for scenario in scenarios), 9)
        self.assertEqual(sum(scenario.imgsz == 1280 for scenario in scenarios), 9)

    def test_benchmark_child_clears_native_only_profilers(self) -> None:
        variable = "MONTGOMERY_PROFILE_TRAINING"
        previous = os.environ.get(variable)
        os.environ[variable] = "1"
        try:
            completed, _, _ = run_observed(
                [
                    sys.executable,
                    "-c",
                    f"import os; print(os.environ.get('{variable}', 'absent'))",
                ],
                None,
            )
        finally:
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.strip(), "absent")

    def test_no_validation_trainer_skips_both_forced_validation_paths(self) -> None:
        class BaseTrainer:
            def validate(self):
                raise AssertionError("final-epoch validation must be disabled")

            def final_eval(self):
                raise AssertionError("post-training validation must be disabled")

        trainer = no_validation_trainer(BaseTrainer)()

        self.assertEqual(trainer.validate(), ({}, 0.0))
        self.assertIsNone(trainer.final_eval())

    def test_summary_keeps_timing_and_resource_passes_separate(self) -> None:
        timing_trials = [
            {"wall_seconds": 10.0, "resources": None},
            {"wall_seconds": 12.0, "resources": None},
            {"wall_seconds": 11.0, "resources": None},
        ]
        resource_trials = [
            {
                "wall_seconds": 99.0,
                "resources": {
                    "process_tree": {
                        "peak_process_count": 5,
                        "peak_rss_bytes": 2_000,
                        "peak_private_bytes": 1_500,
                    },
                    "gpu": {
                        "peak_dedicated_bytes": 3_000,
                        "peak_shared_bytes": 500,
                    },
                },
            }
        ]

        summary = summarize(timing_trials, resource_trials)

        self.assertEqual(summary["median_seconds"], 11.0)
        self.assertEqual(summary["max_seconds"], 12.0)
        self.assertEqual(
            summary["resources"]["peak_rss_bytes"]["median"], 2_000
        )
        self.assertEqual(
            summary["resources"]["peak_gpu_dedicated_bytes"]["median"], 3_000
        )

    def test_summary_can_explicitly_omit_resources(self) -> None:
        summary = summarize([{"wall_seconds": 4.0, "resources": None}], [])

        self.assertEqual(summary["median_seconds"], 4.0)
        self.assertTrue(
            all(value is None for value in summary["resources"].values())
        )

    def test_windows_counter_instances_are_summed_by_pid(self) -> None:
        values = {
            "pid_42_luid_0x00000000_0x00000001_phys_0": 10,
            "pid_42_luid_0x00000000_0x00000002_phys_1": 20,
            "pid_7_luid_0x00000000_0x00000001_phys_0": 100,
            "not-a-process": 1000,
        }
        self.assertEqual(WindowsPdhGpuMemory._by_pid(values, {42}), {42: 30})

    def test_drm_memory_units(self) -> None:
        self.assertEqual(LinuxDrmGpuMemory._memory_value("drm-memory-vram:\t2 KiB"), 2048)
        self.assertEqual(LinuxDrmGpuMemory._memory_value("drm-memory-gtt:\t3 MiB"), 3 * 1024**2)

    def test_drm_prefers_standard_resident_fields_and_classifies_regions(self) -> None:
        lines = [
            "drm-memory-vram:\t1 MiB",
            "drm-resident-vram:\t2 MiB",
            "drm-resident-local:\t3 MiB",
            "drm-resident-system:\t4 MiB",
        ]
        self.assertEqual(
            LinuxDrmGpuMemory._fd_memory(lines),
            (5 * 1024**2, 4 * 1024**2),
        )

    def test_drm_deduplicates_shared_clients_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid in (10, 20):
                fdinfo = root / str(pid) / "fdinfo"
                fdinfo.mkdir(parents=True)
                (fdinfo / "7").write_text(
                    "drm-pdev:\t0000:01:00.0\n"
                    "drm-client-id:\t42\n"
                    "drm-resident-vram:\t8 MiB\n",
                    encoding="utf-8",
                )
            sampler = LinuxDrmGpuMemory(root)
            values = sampler.sample({10, 20})
            self.assertEqual(sum(value[0] for value in values.values()), 8 * 1024**2)
            self.assertTrue(sampler.available)

    def test_drm_warns_when_client_identity_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fdinfo = root / "10" / "fdinfo"
            fdinfo.mkdir(parents=True)
            (fdinfo / "7").write_text(
                "drm-resident-vram:\t8 MiB\n", encoding="utf-8"
            )
            sampler = LinuxDrmGpuMemory(root)
            sampler.sample({10})
            self.assertTrue(sampler.warnings)

    def test_nvml_unsupported_queries_do_not_report_available_zero(self) -> None:
        class Unsupported(Exception):
            pass

        class FakeNvml:
            NVMLError_NotSupported = Unsupported
            NVMLError_FunctionNotFound = Unsupported

            @staticmethod
            def nvmlDeviceGetComputeRunningProcesses_v3(_handle):
                raise Unsupported()

            @staticmethod
            def nvmlDeviceGetGraphicsRunningProcesses_v3(_handle):
                raise Unsupported()

        sampler = NvmlGpuMemory.__new__(NvmlGpuMemory)
        sampler._nvml = FakeNvml()
        sampler._handles = [object()]
        sampler._queries_complete = False
        sampler._memory_unavailable = False
        sampler.warnings = []
        self.assertEqual(sampler.sample({42}), {})
        self.assertFalse(sampler.available)

    def test_nvml_unavailable_memory_invalidates_the_measurement(self) -> None:
        class Unsupported(Exception):
            pass

        class FakeNvml:
            NVMLError_NotSupported = Unsupported
            NVMLError_FunctionNotFound = Unsupported

            @staticmethod
            def nvmlDeviceGetComputeRunningProcesses_v3(_handle):
                return [SimpleNamespace(pid=42, usedGpuMemory=(1 << 64) - 1)]

            @staticmethod
            def nvmlDeviceGetGraphicsRunningProcesses_v3(_handle):
                return []

        sampler = NvmlGpuMemory.__new__(NvmlGpuMemory)
        sampler._nvml = FakeNvml()
        sampler._handles = [object()]
        sampler._queries_complete = False
        sampler._memory_unavailable = False
        sampler.warnings = []
        self.assertEqual(sampler.sample({42}), {})
        self.assertFalse(sampler.available)
        self.assertTrue(sampler.warnings)

    def test_nvml_one_unsupported_query_category_is_not_available(self) -> None:
        class Unsupported(Exception):
            pass

        class FakeNvml:
            NVMLError_NotSupported = Unsupported
            NVMLError_FunctionNotFound = Unsupported

            @staticmethod
            def nvmlDeviceGetComputeRunningProcesses_v3(_handle):
                return []

            @staticmethod
            def nvmlDeviceGetGraphicsRunningProcesses_v3(_handle):
                raise Unsupported()

        sampler = NvmlGpuMemory.__new__(NvmlGpuMemory)
        sampler._nvml = FakeNvml()
        sampler._handles = [object()]
        sampler._queries_complete = False
        sampler._memory_unavailable = False
        sampler.warnings = []
        self.assertEqual(sampler.sample({42}), {})
        self.assertFalse(sampler.available)

    def test_process_tree_ram_is_observed(self) -> None:
        child_code = (
            "import subprocess,sys,time; "
            "data=bytearray(8*1024*1024); "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import time; data=bytearray(8*1024*1024); time.sleep(0.7)']); "
            "time.sleep(0.7); child.wait()"
        )
        process = subprocess.Popen([sys.executable, "-c", child_code])
        monitor = ResourceMonitor(process.pid, interval_seconds=0.02)
        monitor.start()
        process.wait(timeout=5)
        result = monitor.stop()
        self.assertGreaterEqual(result["process_tree"]["peak_process_count"], 2)
        self.assertGreater(result["process_tree"]["peak_rss_bytes"], 8 * 1024**2)
        self.assertGreaterEqual(result["sampling"]["sample_count"], 2)
        self.assertTrue(result["processes"])


if __name__ == "__main__":
    unittest.main()
