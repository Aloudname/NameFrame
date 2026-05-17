"""System resource monitor, GPU memory/utilization and CPU memory tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MonitorReport:
    """Aggregated resource usage statistics collected by :class:`Monitor`.

    Attributes:
        peak_gpu_memory: Peak GPU memory in MiB over the monitoring window.
        avg_gpu_util: Average GPU utilization (0-100).
        peak_cpu_memory: Peak resident CPU memory in MiB.
        timeline: List of per-sample snapshots.
    """

    peak_gpu_memory: float = 0.0
    avg_gpu_util: float = 0.0
    peak_cpu_memory: float = 0.0
    timeline: List[Dict[str, float]] = field(default_factory=list)


class Monitor:
    """Background system resource monitor.

    Samples GPU memory/utilization and CPU memory at a configurable
    interval. Runs in a daemon thread so it does not block training.

    Attributes:
        log_interval: Sampling interval in seconds.
    """

    def __init__(self, log_interval: float = 5.0) -> None:
        """Initialize the monitor.

        Args:
            log_interval: Seconds between resource snapshots.
        """
        self.log_interval: float = log_interval
        self._snapshots: List[Dict[str, float]] = []
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._snapshots.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> MonitorReport:
        """Stop monitoring and return aggregated statistics.

        Returns:
            A :class:`MonitorReport` with peak and average values.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        return self._aggregate()

    def get_current(self) -> Dict[str, float]:
        """Take a single snapshot of current resource usage.

        Returns:
            Dict with keys like ``"gpu_memory_mib"``, ``"gpu_util_pct"``,
            ``"cpu_memory_mib"``.
        """
        return self._sample()

    def _run(self) -> None:
        while self._running:
            self._snapshots.append(self._sample())
            time.sleep(self.log_interval)

    @staticmethod
    def _sample() -> Dict[str, float]:
        snapshot: Dict[str, float] = {
            "gpu_memory_mib": 0.0,
            "gpu_util_pct": 0.0,
            "cpu_memory_mib": 0.0,
        }

        # GPU metrics via pynvml (if available)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snapshot["gpu_memory_mib"] = mem_info.used / (1024 * 1024)
            snapshot["gpu_util_pct"] = float(util_info.gpu)
        except (ImportError, Exception):
            # fallback: torch.cuda
            try:
                import torch
                if torch.cuda.is_available():
                    snapshot["gpu_memory_mib"] = (
                        torch.cuda.max_memory_allocated() / (1024 * 1024)
                    )
            except ImportError:
                pass

        # CPU memory
        try:
            import psutil
            proc = psutil.Process()
            mem_bytes: int = proc.memory_info().rss
            snapshot["cpu_memory_mib"] = mem_bytes / (1024 * 1024)
        except (ImportError, Exception):
            pass

        return snapshot

    def _aggregate(self) -> MonitorReport:
        if not self._snapshots:
            return MonitorReport()

        return MonitorReport(
            peak_gpu_memory=float(
                max(s.get("gpu_memory_mib", 0.0) for s in self._snapshots)
            ),
            avg_gpu_util=float(
                sum(s.get("gpu_util_pct", 0.0) for s in self._snapshots)
                / len(self._snapshots)
            ),
            peak_cpu_memory=float(
                max(s.get("cpu_memory_mib", 0.0) for s in self._snapshots)
            ),
            timeline=list(self._snapshots),
        )


def monitor(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that wraps a function with Monitor start/stop.

    The :class:`MonitorReport` is attached to the result as
    ``result._monitor_report``.

    Args:
        func: The function to monitor.

    Returns:
        Wrapped function.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        mon: Monitor = Monitor()
        mon.start()
        try:
            result: Any = func(*args, **kwargs)
        finally:
            report: MonitorReport = mon.stop()
            if hasattr(result, "_monitor_report"):
                pass  # avoid overwriting
            else:
                try:
                    result._monitor_report = report  # type: ignore[attr-defined]
                except (AttributeError, TypeError):
                    pass
        return result

    return wrapper
