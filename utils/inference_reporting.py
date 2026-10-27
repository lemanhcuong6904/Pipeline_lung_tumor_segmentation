"""Reporting and optional model-forward resource measurements."""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import numpy as np
import torch


def finite_std(values) -> float:
    """Population standard deviation over finite observations (ddof=0)."""
    valid = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    return float(valid.std(ddof=0)) if valid.size else float('nan')


def rounded_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: rounded_output(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded_output(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class InferenceProfiler:
    """Time model forward only; sample this process's RSS during forward calls."""

    def __init__(self, enabled: bool, device: torch.device, warmup_batches: int = 2):
        if not isinstance(enabled, bool) or not isinstance(warmup_batches, int) or isinstance(warmup_batches, bool) or warmup_batches < 0:
            raise ValueError('profiling flag must be bool and warmup_batches a non-negative integer')
        self.enabled, self.device, self.warmup_batches = enabled, device, warmup_batches
        self.warmed: set[str] = set()
        self.times: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.samples: dict[str, int] = {}
        self.peak_cpu = 0
        self.peak_gpu = 0
        if enabled:
            try:
                import psutil
            except ImportError as error:
                raise RuntimeError('Profiling requires psutil: pip install psutil') from error
            self.process = psutil.Process()

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def run(self, stage: str, forward: Callable[[], Any], batch_size: int = 1) -> Any:
        if not self.enabled:
            return forward()
        if stage not in self.warmed:
            for _ in range(self.warmup_batches):
                forward()
            self._sync()
            self.warmed.add(stage)
        self._sync()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        stopped = threading.Event()
        peak = [self.process.memory_info().rss]
        def sample():
            while not stopped.wait(0.005):
                peak[0] = max(peak[0], self.process.memory_info().rss)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        start = time.perf_counter()
        try:
            result = forward()
            self._sync()
            elapsed = time.perf_counter() - start
            self.times[stage] = self.times.get(stage, 0.0) + elapsed
            self.calls[stage] = self.calls.get(stage, 0) + 1
            self.samples[stage] = self.samples.get(stage, 0) + batch_size
            if self.device.type == 'cuda':
                self.peak_gpu = max(self.peak_gpu, torch.cuda.max_memory_allocated(self.device))
            return result
        finally:
            peak[0] = max(peak[0], self.process.memory_info().rss)
            stopped.set()
            sampler.join()
            self.peak_cpu = max(self.peak_cpu, peak[0])

    def report(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        total_samples = sum(self.samples.values())
        return {
            'measurement_scope': 'model_forward_only',
            'warmup_batches_per_model': self.warmup_batches,
            'model_forward_seconds': sum(self.times.values()),
            'model_forward_calls': sum(self.calls.values()),
            'model_forward_samples': total_samples,
            'model_forward_seconds_per_sample': sum(self.times.values()) / total_samples if total_samples else None,
            'stages_seconds': dict(self.times),
            'stages_samples': dict(self.samples),
            'stages_seconds_per_sample': {stage: seconds / self.samples[stage] if self.samples[stage] else None for stage, seconds in self.times.items()},
            'peak_cpu_rss_mb': self.peak_cpu / (1024 ** 2) if self.calls else None,
            'peak_gpu_allocated_mb': self.peak_gpu / (1024 ** 2) if self.device.type == 'cuda' and self.calls else None,
            'cpu_memory_scope': 'sampled_process_rss_during_forward',
            'gpu_memory_scope': 'pytorch_allocated_during_forward',
        }
