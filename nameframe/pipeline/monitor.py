"""Resource monitor, GPU and CPU memory/util tracking."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MonitorReport:
    """Resource metrics collected by `Monitor` instance.

    Attributes:
        peak_gpu_memory: Peak GPU memory GB over the monitoring window.
        avg_gpu_util: Average GPU util rate in %.
        peak_cpu_memory: Peak CPU memory GB.
        timeline: List of per-sample snapshots.
    """

    peak_gpu_memory: float = 0.0
    avg_gpu_util: float = 0.0
    peak_cpu_memory: float = 0.0
    timeline: List[Dict[str, float]] = field(default_factory=list)


# this is a script to be written in a .temp,
# and launched in a separate terminal,
# for live rendering of monitor.
_RENDER_SERVER_SCRIPT = r"""
import json, os, sys, time, signal, shutil

def bar(val, max_val, width=20):
    pct = min(val / max(max_val, 1e-8) * 100, 100)
    filled = int(pct / 100 * width)
    blocks = '█' * filled + '░' * (width - filled)
    return blocks, pct

def format_elapsed(sec):
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def color_for(pct):
    if pct < 50:
        return '\033[32m'
    elif pct < 80:
        return '\033[33m'
    return '\033[31m'

def render(frame, history):
    gpu_mem = frame.get('gpu_memory', 0)
    gpu_util = frame.get('gpu_util', 0)
    cpu_mem = frame.get('cpu_memory', 0)
    elapsed = frame.get('elapsed', 0)
    samples = len(history) + 1

    peak_gpu = max((f.get('gpu_memory', 0) for f in history), default=gpu_mem)
    peak_cpu = max((f.get('cpu_memory', 0) for f in history), default=cpu_mem)
    avg_util = sum(f.get('gpu_util', 0) for f in history + [frame]) / max(samples, 1)

    bw = shutil.get_terminal_size((80, 24)).columns
    w = max(38, min(bw - 2, 60))

    def line(left, right=''):
        content = f"  {left}"
        pad = w - len(content) - 2
        if right:
            right_str = f" {right}"
            pad -= len(right_str)
            content += ' ' * max(pad, 1) + right_str
        return f"│{content}│"

    top = '┌' + '─' * w + '┐'
    sep = '│' + ' ' * w + '│'
    bot = '└' + '─' * w + '┘'

    # Detect total GPU memory for bar scaling
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_total = info.total / (1024 * 1024 * 1024)
    except Exception:
        gpu_total = max(gpu_mem * 1.2, 24)

    # Detect total CPU memory for bar scaling
    try:
        import psutil
        cpu_total = psutil.virtual_memory().total / (1024 * 1024 * 1024)
    except Exception:
        cpu_total = max(cpu_mem * 1.2, 64)

    gpu_bar, gpu_pct = bar(gpu_mem, gpu_total)
    util_bar, util_pct = bar(gpu_util, 100)
    cpu_bar, cpu_pct = bar(cpu_mem, cpu_total)

    out = ['\033[H\033[J']
    out.append(top)
    out.append(line('\033[1mNameFrame Resource Monitor\033[0m'))
    out.append(sep)
    out.append(line(f'Elapsed: {format_elapsed(elapsed)}', f'Samples: {samples}'))
    out.append(sep)
    out.append(line(f'GPU Memory: {color_for(gpu_pct)}{gpu_bar}\033[0m',
                    f'{gpu_mem:5.1f} / {gpu_total:.0f} GiB'))
    out.append(line(f'GPU Util:   {color_for(util_pct)}{util_bar}\033[0m',
                    f'{gpu_util:5.1f} %'))
    out.append(line(f'CPU Memory: {color_for(cpu_pct)}{cpu_bar}\033[0m',
                    f'{cpu_mem:5.1f} / {cpu_total:.0f} GiB'))
    out.append(sep)
    out.append(line(f'Peak GPU Mem:', f'{peak_gpu:5.1f} GiB'))
    out.append(line(f'Peak CPU Mem:', f'{peak_cpu:5.1f} GiB'))
    out.append(line(f'Avg GPU Util:', f'{avg_util:5.1f} %'))
    out.append(bot)
    sys.stdout.write('\n'.join(out) + '\n')
    sys.stdout.flush()

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    fifo_path = sys.argv[1]

    sys.stdout.write('\033[?25l\033[2J\033[H')
    sys.stdout.flush()

    def cleanup(sig=None, frame=None):
        sys.stdout.write('\033[?25h\033[2J\033[H')
        sys.stdout.flush()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    history = []
    try:
        with open(fifo_path, 'r') as fifo:
            while True:
                line = fifo.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line.strip())
                    render(frame, history)
                    history.append(frame)
                    if len(history) > 10000:
                        history = history[-5000:]
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
    except (KeyboardInterrupt, IOError):
        pass
    finally:
        cleanup()

if __name__ == '__main__':
    main()
"""


class Monitor:
    """
    Resource monitor.
    Samples GPU memory/util rate and CPU memory at an interval.

    Attributes:
        log_interval: Sampling interval in seconds.
    """

    _render_fifo_path: Optional[str] = None
    _render_fifo_fd: Optional[int] = None
    _render_proc: "Optional[subprocess.Popen[bytes]]" = None
    _render_script_path: Optional[str] = None
    _render_data: Optional[Dict[str, float]] = None
    _render_start_time: float = 0.0

    def __init__(self, log_interval: float = 1.0) -> None:
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
        Monitor._render_start_time = time.time()
        try:
            self._setup_render()
        except Exception:
            pass
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
        self._teardown_render()
        return self._aggregate()

    def get_current(self) -> Dict[str, float]:
        """Take a single snapshot of current resource usage.

        Returns:
            Dict with keys like ``"gpu_memory"``, ``"gpu_util"``,
            ``"cpu_memory"``.
        """
        return self._sample()

    def _run(self) -> None:
        while self._running:
            snapshot = self._sample()
            self._snapshots.append(snapshot)
            Monitor._render_data = snapshot
            self._render()
            time.sleep(self.log_interval)

    @staticmethod
    def _sample() -> Dict[str, float]:
        snapshot: Dict[str, float] = {
            "gpu_memory": 0.0,
            "gpu_util": 0.0,
            "cpu_memory": 0.0,
        }

        # GPU metrics via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            snapshot["gpu_memory"] = mem_info.used / (1024 * 1024 * 1024)
            snapshot["gpu_util"] = float(util_info.gpu)
        except (ImportError, Exception):
            # fallback: torch.cuda
            try:
                import torch
                if torch.cuda.is_available():
                    snapshot["gpu_memory"] = (
                        torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
                    )
            except ImportError:
                pass

        # CPU memory
        try:
            import psutil
            proc = psutil.Process()
            mem_bytes: int = proc.memory_info().rss
            snapshot["cpu_memory"] = mem_bytes / (1024 * 1024 * 1024)
        except (ImportError, Exception):
            pass

        return snapshot

    @staticmethod
    def _render() -> bool:
        fd = Monitor._render_fifo_fd
        if fd is None:
            return False
        data = Monitor._render_data
        if data is None:
            return False
        try:
            elapsed = time.time() - Monitor._render_start_time
            frame: Dict[str, float] = {"elapsed": elapsed, **data}
            line = (json.dumps(frame, separators=(",", ":")) + "\n").encode()
            written = os.write(fd, line)
            return written == len(line)
        except (OSError, BrokenPipeError):
            Monitor._render_fifo_fd = None
            return False

    def _setup_render(self) -> bool:
        fifo_path = f"/tmp/monitor_{os.getpid()}.fifo"
        if os.path.exists(fifo_path):
            try:
                os.remove(fifo_path)
            except OSError:
                pass
        try:
            os.mkfifo(fifo_path, 0o600)
        except OSError:
            return False
        Monitor._render_fifo_path = fifo_path

        try:
            fd, script_path = tempfile.mkstemp(
                suffix=".py", prefix="render_"
            )
            with os.fdopen(fd, "w") as f:
                f.write(_RENDER_SERVER_SCRIPT)
            os.chmod(script_path, 0o700)
            Monitor._render_script_path = script_path
        except OSError:
            self._cleanup_fifo()
            return False

        proc = Monitor._launch_render_terminal(script_path, fifo_path)
        if proc is None:
            self._cleanup_fifo()
            try:
                os.remove(script_path)
            except OSError:
                pass
            Monitor._render_script_path = None
            return False
        Monitor._render_proc = proc

        fifo_fd = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                fifo_fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError:
                time.sleep(0.1)
        if fifo_fd is None:
            self._cleanup_fifo()
            return False
        Monitor._render_fifo_fd = fifo_fd
        return True

    @staticmethod
    def _launch_render_terminal(
        script_path: str, fifo_path: str
    ) -> "Optional[subprocess.Popen[bytes]]":
        python = sys.executable
        cmd_parts = [python, script_path, fifo_path]

        candidates = [
            ("gnome-terminal", ["--"]),
            ("xterm", ["-e"]),
            ("konsole", ["-e"]),
            ("xfce4-terminal", ["-e"]),
            ("lxterminal", ["-e"]),
            ("terminator", ["-e"]),
            ("terminology", ["-e"]),
            ("urxvt", ["-e"]),
            ("rxvt", ["-e"]),
            ("st", ["-e"]),
            ("alacritty", ["-e"]),
        ]
        for binary, prefix in candidates:
            bin_path = shutil.which(binary)
            if bin_path is None:
                continue
            try:
                return subprocess.Popen(
                    [bin_path] + prefix + cmd_parts,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except (OSError, subprocess.SubprocessError):
                continue
        return None

    def _teardown_render(self) -> None:
        fd = Monitor._render_fifo_fd
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            Monitor._render_fifo_fd = None

        proc = Monitor._render_proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            Monitor._render_proc = None

        path = Monitor._render_fifo_path
        if path is not None and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
            Monitor._render_fifo_path = None

        script = Monitor._render_script_path
        if script is not None and os.path.exists(script):
            try:
                os.remove(script)
            except OSError:
                pass
            Monitor._render_script_path = None

        Monitor._render_data = None

    def _cleanup_fifo(self) -> None:
        path = Monitor._render_fifo_path
        if path is not None and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        Monitor._render_fifo_path = None

    def _aggregate(self) -> MonitorReport:
        if not self._snapshots:
            return MonitorReport()

        return MonitorReport(
            peak_gpu_memory=float(
                max(s.get("gpu_memory", 0.0) for s in self._snapshots)
            ),
            avg_gpu_util=float(
                sum(s.get("gpu_util", 0.0) for s in self._snapshots)
                / len(self._snapshots)
            ),
            peak_cpu_memory=float(
                max(s.get("cpu_memory", 0.0) for s in self._snapshots)
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
