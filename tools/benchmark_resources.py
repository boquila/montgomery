"""Process-tree RAM and GPU-memory telemetry for benchmarks.

The monitor reports simultaneous process-tree peaks as well as per-process peaks.  GPU memory is
read from Windows Performance Data Helper counters, NVIDIA's management library on Linux, and
Linux DRM fdinfo when available.  Unsupported counters are reported explicitly rather than being
silently replaced with whole-device memory usage.
"""

from __future__ import annotations

import os
import platform
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import psutil


GPU_COUNTER_PID = re.compile(r"(?:^|_)pid_(\d+)(?:_|$)", re.IGNORECASE)
DRM_RESIDENT_MEMORY = re.compile(r"^drm-(resident|memory)-([^:]+):")


class GpuMemorySampler(Protocol):
    provider: str
    description: str
    available: bool
    per_process_attribution: str

    def sample(self, pids: set[int]) -> dict[int, tuple[int, int | None]]: ...

    def close(self) -> None: ...


class WindowsPdhGpuMemory:
    """Per-process GPU memory from Windows' vendor-neutral GPU counters."""

    provider = "windows-pdh"
    available = True
    per_process_attribution = "reported by Windows counter instance PID"
    description = (
        "Windows GPU Process Memory counters; dedicated bytes are VRAM and shared bytes are "
        "system memory mapped for GPU use"
    )

    def __init__(self) -> None:
        import win32pdh

        self._pdh = win32pdh
        self._query = win32pdh.OpenQuery()
        try:
            self._dedicated = win32pdh.AddEnglishCounter(
                self._query, r"\GPU Process Memory(*)\Dedicated Usage"
            )
            self._shared = win32pdh.AddEnglishCounter(
                self._query, r"\GPU Process Memory(*)\Shared Usage"
            )
            win32pdh.CollectQueryData(self._query)
        except Exception:
            win32pdh.CloseQuery(self._query)
            raise

    @staticmethod
    def _by_pid(values: Mapping[str, int], pids: set[int]) -> dict[int, int]:
        totals: dict[int, int] = {}
        for instance, value in values.items():
            match = GPU_COUNTER_PID.search(instance)
            if match is None:
                continue
            pid = int(match.group(1))
            if pid in pids:
                totals[pid] = totals.get(pid, 0) + max(0, int(value))
        return totals

    def sample(self, pids: set[int]) -> dict[int, tuple[int, int | None]]:
        if not pids:
            return {}
        self._pdh.CollectQueryData(self._query)
        dedicated = self._by_pid(
            self._pdh.GetFormattedCounterArray(self._dedicated, self._pdh.PDH_FMT_LARGE),
            pids,
        )
        shared = self._by_pid(
            self._pdh.GetFormattedCounterArray(self._shared, self._pdh.PDH_FMT_LARGE),
            pids,
        )
        return {
            pid: (dedicated.get(pid, 0), shared.get(pid, 0))
            for pid in dedicated.keys() | shared.keys()
        }

    def close(self) -> None:
        for counter in (self._dedicated, self._shared):
            try:
                self._pdh.RemoveCounter(counter)
            except Exception:
                pass
        try:
            self._pdh.CloseQuery(self._query)
        except Exception:
            pass


class NvmlGpuMemory:
    """Per-process dedicated NVIDIA memory on Linux."""

    provider = "linux-nvml"
    per_process_attribution = "reported by NVML process PID"
    description = "NVML compute and graphics process usedGpuMemory, summed across NVIDIA devices"

    def __init__(self) -> None:
        import pynvml

        self._nvml = pynvml
        self._queries_complete = False
        self._memory_unavailable = False
        self.warnings: list[str] = []
        pynvml.nvmlInit()
        self._handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]

    @property
    def available(self) -> bool:
        return self._queries_complete and not self._memory_unavailable

    def _running(self, handle: Any, kind: str) -> tuple[list[Any], bool]:
        unsupported = tuple(
            error
            for error in (
                getattr(self._nvml, "NVMLError_NotSupported", None),
                getattr(self._nvml, "NVMLError_FunctionNotFound", None),
            )
            if error is not None
        )
        for suffix in ("_v3", "_v2", ""):
            function = getattr(
                self._nvml, f"nvmlDeviceGet{kind}RunningProcesses{suffix}", None
            )
            if function is not None:
                try:
                    processes = list(function(handle))
                    return processes, True
                except unsupported:
                    continue
        return [], False

    def sample(self, pids: set[int]) -> dict[int, tuple[int, int | None]]:
        totals: dict[int, int] = {}
        unavailable = (1 << 64) - 1
        queries_complete = bool(self._handles)
        for handle in self._handles:
            # A PID can appear in both lists. Count its largest report once per device.
            on_device: dict[int, int] = {}
            for kind in ("Compute", "Graphics"):
                processes, supported = self._running(handle, kind)
                queries_complete = queries_complete and supported
                for process in processes:
                    pid = int(process.pid)
                    used = getattr(process, "usedGpuMemory", None)
                    if pid not in pids:
                        continue
                    if used is None or int(used) == unavailable:
                        self._memory_unavailable = True
                        warning = "NVML returned unavailable per-process GPU memory"
                        if warning not in self.warnings:
                            self.warnings.append(warning)
                        continue
                    on_device[pid] = max(on_device.get(pid, 0), max(0, int(used)))
            for pid, used in on_device.items():
                totals[pid] = totals.get(pid, 0) + used
        self._queries_complete = queries_complete
        return {pid: (used, None) for pid, used in totals.items()}

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


class LinuxDrmGpuMemory:
    """Per-process VRAM/GTT accounting exposed by Linux DRM drivers."""

    provider = "linux-drm-fdinfo"
    per_process_attribution = (
        "each deduplicated DRM client is attributed once to the lowest observed PID exposing it; "
        "the aggregate process-tree value is authoritative"
    )
    description = (
        "Linux /proc/PID/fdinfo standardized drm-resident-* counters and deprecated "
        "AMDGPU drm-memory-* aliases"
    )

    def __init__(self, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root
        self.accounting_observed = False
        self.warnings: list[str] = []

    @property
    def available(self) -> bool:
        return self.accounting_observed

    @staticmethod
    def _memory_value(line: str) -> int:
        fields = line.split()
        if len(fields) < 2:
            return 0
        value = int(fields[1])
        unit = fields[2].lower() if len(fields) > 2 else "b"
        return value * {"b": 1, "kib": 1024, "mib": 1024**2}.get(unit, 1)

    @classmethod
    def _fd_memory(cls, lines: list[str]) -> tuple[int, int]:
        # A driver may expose the old drm-memory-* alias or the standardized drm-resident-* key.
        # Prefer the standardized value when both exist so the same region is never double-counted.
        regions: dict[str, tuple[int, int]] = {}
        for line in lines:
            match = DRM_RESIDENT_MEMORY.match(line)
            if match is None:
                continue
            kind, region = match.groups()
            priority = 2 if kind == "resident" else 1
            if region not in regions or priority > regions[region][0]:
                regions[region] = (priority, cls._memory_value(line))
        dedicated = 0
        shared = 0
        for region, (_, value) in regions.items():
            normalized = region.lower()
            if normalized == "memory" or normalized.startswith(
                ("system", "gtt", "stolen")
            ):
                shared += value
            else:
                dedicated += value
        return dedicated, shared

    @staticmethod
    def _client_identity(
        lines: list[str], pid: int, fdinfo_path: Path
    ) -> tuple[str, str] | tuple[str, int, str]:
        fields = {}
        for line in lines:
            key, separator, value = line.partition(":")
            if separator:
                fields[key] = value.strip()
        client_id = fields.get("drm-client-id")
        if client_id is None:
            return ("fd", pid, fdinfo_path.name)
        device = fields.get("drm-pdev", "global")
        return (device, client_id)

    def sample(self, pids: set[int]) -> dict[int, tuple[int, int | None]]:
        result: dict[int, tuple[int, int | None]] = {}
        seen_clients: set[tuple[str, str] | tuple[str, int, str]] = set()
        for pid in sorted(pids):
            dedicated = 0
            shared = 0
            try:
                fdinfo_paths = list((self._proc_root / str(pid) / "fdinfo").iterdir())
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            for path in fdinfo_paths:
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                    continue
                if not any(DRM_RESIDENT_MEMORY.match(line) for line in lines):
                    continue
                self.accounting_observed = True
                if not any(line.startswith("drm-client-id:") for line in lines):
                    warning = (
                        "DRM memory counters lack drm-client-id; duplicate file descriptors "
                        "cannot be deduplicated"
                    )
                    if warning not in self.warnings:
                        self.warnings.append(warning)
                identity = self._client_identity(lines, pid, path)
                if identity in seen_clients:
                    continue
                seen_clients.add(identity)
                fd_dedicated, fd_shared = self._fd_memory(lines)
                dedicated += fd_dedicated
                shared += fd_shared
            if dedicated or shared:
                result[pid] = (dedicated, shared)
        return result

    def close(self) -> None:
        pass


class LinuxGpuMemory:
    """Combine NVML and DRM without double-counting the same PID."""

    provider = "linux-nvml+drm-fdinfo"
    per_process_attribution = (
        "NVML values use reported PIDs; each deduplicated DRM client is attributed once to the "
        "lowest observed PID exposing it; the aggregate process-tree value is authoritative"
    )
    description = (
        "NVML is preferred per PID when available; Linux DRM fdinfo covers other DRM drivers"
    )

    def __init__(self) -> None:
        self._drm = LinuxDrmGpuMemory()
        self._nvml: NvmlGpuMemory | None = None
        self._drm_available = Path("/sys/class/drm").is_dir()
        self.initialization_notes: list[str] = []
        try:
            self._nvml = NvmlGpuMemory()
        except Exception as error:
            self.initialization_notes.append(f"NVML unavailable: {type(error).__name__}: {error}")
        if self._nvml is None and not self._drm_available:
            raise RuntimeError("neither NVML nor Linux DRM accounting is available")

    @property
    def available(self) -> bool:
        return (self._nvml is not None and self._nvml.available) or self._drm.available

    @property
    def warnings(self) -> list[str]:
        nvml_warnings = self._nvml.warnings if self._nvml is not None else []
        return [*nvml_warnings, *self._drm.warnings]

    def sample(self, pids: set[int]) -> dict[int, tuple[int, int | None]]:
        result = self._drm.sample(pids)
        if self._nvml is not None:
            nvml_values = self._nvml.sample(pids)
            if self._nvml.available:
                for pid, nvml_value in nvml_values.items():
                    # NVML is authoritative for NVIDIA dedicated memory. Retain DRM GTT if present.
                    result[pid] = (nvml_value[0], result.get(pid, (0, None))[1])
        return result

    def close(self) -> None:
        if self._nvml is not None:
            self._nvml.close()


def create_gpu_sampler() -> tuple[GpuMemorySampler | None, str | None]:
    try:
        if os.name == "nt":
            return WindowsPdhGpuMemory(), None
        if sys_platform() == "linux":
            sampler = LinuxGpuMemory()
            note = "; ".join(sampler.initialization_notes) or None
            return sampler, note
        return None, f"per-process GPU memory is unsupported on {platform.system()}"
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def sys_platform() -> str:
    # Isolated for tests without mutating sys.platform.
    import sys

    return sys.platform


def _private_bytes(process: psutil.Process, memory: Any) -> tuple[int | None, str]:
    private = getattr(memory, "private", None)
    if private is not None:
        return int(private), "private_bytes"
    try:
        unique = getattr(process.memory_full_info(), "uss", None)
    except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError, OSError):
        unique = None
    return (int(unique), "uss") if unique is not None else (None, "unavailable")


class ResourceMonitor:
    """Sample a root process and all descendants until stopped."""

    def __init__(self, root_pid: int, interval_seconds: float = 0.1) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource sampling interval must be positive")
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._started_at = time.perf_counter()
        self._sample_count = 0
        self._regular_sample_count = 0
        self._first_regular_sample_at: float | None = None
        self._last_regular_sample_at: float | None = None
        self._ram_samples = 0
        self._gpu_samples = 0
        self._peak_process_count = 0
        self._peak_rss = 0
        self._peak_private = 0
        self._peak_gpu_dedicated = 0
        self._peak_gpu_shared = 0
        self._gpu_shared_observed = False
        self._private_metric = "unavailable"
        self._known: dict[int, psutil.Process] = {}
        self._process_peaks: dict[int, dict[str, Any]] = {}
        self._errors: list[str] = []
        self._monitoring_errors: list[str] = []
        self._gpu_sampler, self._gpu_initialization_note = create_gpu_sampler()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        if self._thread.is_alive():
            raise RuntimeError("resource monitor did not stop")
        return self.result()

    def _discover(self) -> list[psutil.Process]:
        try:
            root = psutil.Process(self.root_pid)
            candidates = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            candidates = []
        for process in candidates:
            self._known.setdefault(process.pid, process)
        alive = []
        for process in self._known.values():
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    alive.append(process)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                pass
        return alive

    def _sample(self, regular: bool = True) -> None:
        sample_started = time.perf_counter()
        processes = self._discover()
        rss_total = 0
        private_total = 0
        private_available = False
        live_pids: set[int] = set()
        for process in processes:
            try:
                memory = process.memory_info()
                rss = int(memory.rss)
                private, metric = _private_bytes(process, memory)
                name = process.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, OSError):
                continue
            live_pids.add(process.pid)
            rss_total += rss
            peak = self._process_peaks.setdefault(
                process.pid,
                {
                    "pid": process.pid,
                    "name": name,
                    "role": "root" if process.pid == self.root_pid else "descendant",
                    "peak_rss_bytes": 0,
                    "peak_private_bytes": None,
                    "peak_gpu_dedicated_bytes": None,
                    "peak_gpu_shared_bytes": None,
                },
            )
            peak["peak_rss_bytes"] = max(peak["peak_rss_bytes"], rss)
            if private is not None:
                private_available = True
                private_total += private
                peak["peak_private_bytes"] = max(peak["peak_private_bytes"] or 0, private)
                if self._private_metric == "unavailable":
                    self._private_metric = metric
        if live_pids:
            self._ram_samples += 1
            self._peak_process_count = max(self._peak_process_count, len(live_pids))
            self._peak_rss = max(self._peak_rss, rss_total)
            if private_available:
                self._peak_private = max(self._peak_private, private_total)

        if self._gpu_sampler is not None and live_pids:
            try:
                gpu = self._gpu_sampler.sample(live_pids)
                self._gpu_samples += 1
                dedicated_total = sum(value[0] for value in gpu.values())
                shared_values = [value[1] for value in gpu.values() if value[1] is not None]
                self._peak_gpu_dedicated = max(self._peak_gpu_dedicated, dedicated_total)
                if shared_values:
                    self._gpu_shared_observed = True
                    self._peak_gpu_shared = max(self._peak_gpu_shared, sum(shared_values))
                for pid, (dedicated, shared) in gpu.items():
                    peak = self._process_peaks.get(pid)
                    if peak is None:
                        continue
                    peak["peak_gpu_dedicated_bytes"] = max(
                        peak["peak_gpu_dedicated_bytes"] or 0, dedicated
                    )
                    if shared is not None:
                        peak["peak_gpu_shared_bytes"] = max(
                            peak["peak_gpu_shared_bytes"] or 0, shared
                        )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if not self._errors or self._errors[-1] != message:
                    self._errors.append(message)
        self._sample_count += 1
        if regular:
            self._regular_sample_count += 1
            if self._first_regular_sample_at is None:
                self._first_regular_sample_at = sample_started
            self._last_regular_sample_at = sample_started

    def _run(self) -> None:
        next_sample = time.perf_counter()
        try:
            while True:
                try:
                    self._sample()
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    if not self._monitoring_errors or self._monitoring_errors[-1] != message:
                        self._monitoring_errors.append(message)
                next_sample += self.interval_seconds
                now = time.perf_counter()
                if next_sample <= now:
                    missed = int((now - next_sample) / self.interval_seconds) + 1
                    next_sample += missed * self.interval_seconds
                if self._stop.wait(max(0.0, next_sample - now)):
                    break
            try:
                self._sample(regular=False)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if not self._monitoring_errors or self._monitoring_errors[-1] != message:
                    self._monitoring_errors.append(message)
        finally:
            if self._gpu_sampler is not None:
                self._gpu_sampler.close()

    def result(self) -> dict[str, Any]:
        gpu_available = (
            self._gpu_sampler is not None
            and self._gpu_sampler.available
            and self._gpu_samples > 0
        )
        if (
            self._first_regular_sample_at is not None
            and self._last_regular_sample_at is not None
            and self._regular_sample_count > 1
        ):
            effective_interval_ms = (
                (self._last_regular_sample_at - self._first_regular_sample_at)
                / (self._regular_sample_count - 1)
                * 1000
            )
        else:
            effective_interval_ms = None
        return {
            "schema": "montgomery-process-resources-v1",
            "sampling": {
                "interval_ms": round(self.interval_seconds * 1000, 3),
                "effective_interval_ms": effective_interval_ms,
                "interval_semantics": "target sample start-to-start interval",
                "duration_seconds": time.perf_counter() - self._started_at,
                "sample_count": self._sample_count,
                "scheduled_sample_count": self._regular_sample_count,
                "monitoring_errors": self._monitoring_errors,
            },
            "process_tree": {
                "memory_scope": "root process plus recursive descendants, sampled simultaneously",
                "private_metric": self._private_metric,
                "samples_with_processes": self._ram_samples,
                "peak_process_count": self._peak_process_count,
                "peak_rss_bytes": self._peak_rss if self._ram_samples else None,
                "peak_private_bytes": self._peak_private
                if self._ram_samples and self._private_metric != "unavailable"
                else None,
            },
            "gpu": {
                "available": gpu_available,
                "provider": self._gpu_sampler.provider if self._gpu_sampler else None,
                "description": self._gpu_sampler.description if self._gpu_sampler else None,
                "per_process_attribution": self._gpu_sampler.per_process_attribution
                if self._gpu_sampler
                else None,
                "samples_succeeded": self._gpu_samples,
                "peak_dedicated_bytes": self._peak_gpu_dedicated if gpu_available else None,
                "peak_shared_bytes": self._peak_gpu_shared
                if gpu_available and self._gpu_shared_observed
                else None,
                "initialization_note": self._gpu_initialization_note,
                "provider_warnings": list(getattr(self._gpu_sampler, "warnings", [])),
                "sampling_errors": self._errors,
            },
            "processes": sorted(self._process_peaks.values(), key=lambda entry: entry["pid"]),
        }
