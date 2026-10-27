import math

import pytest
import torch


def test_report_rounding_and_population_std_ignore_nonfinite():
    from utils.inference_reporting import finite_std, rounded_output

    assert finite_std([1.0, 3.0, float('nan')]) == 1.0
    assert math.isnan(finite_std([float('nan')]))
    assert rounded_output({'a': 0.123456, 'b': [float('nan'), 2]}) == {'a': 0.1235, 'b': [None, 2]}


def test_profiler_warmup_and_disabled_behavior():
    from utils.inference_reporting import InferenceProfiler

    calls = []
    def forward():
        calls.append(1)
        return 42
    profiler = InferenceProfiler(True, torch.device('cpu'), warmup_batches=2)
    assert profiler.run('model', forward) == 42
    assert profiler.run('model', forward) == 42
    report = profiler.report()
    assert len(calls) == 4
    assert report['model_forward_calls'] == 2
    assert report['model_forward_seconds'] >= 0
    assert report['peak_cpu_rss_mb'] > 0
    assert report['peak_gpu_allocated_mb'] is None
    disabled = InferenceProfiler(False, torch.device('cpu'))
    assert disabled.run('model', forward) == 42
    assert disabled.report() is None


def test_profiler_average_uses_actual_batch_sizes_per_stage(monkeypatch):
    from utils.inference_reporting import InferenceProfiler

    clock = iter([0.0, 2.0, 3.0, 4.0, 5.0, 9.0])
    monkeypatch.setattr('utils.inference_reporting.time.perf_counter', lambda: next(clock))
    profiler = InferenceProfiler(True, torch.device('cpu'), warmup_batches=1)
    assert profiler.report()['model_forward_seconds_per_sample'] is None
    profiler.run('lung', lambda: None, batch_size=4)
    profiler.run('lung', lambda: None, batch_size=1)
    profiler.run('tumor', lambda: None, batch_size=2)
    report = profiler.report()
    assert report['model_forward_samples'] == 7
    assert report['model_forward_seconds_per_sample'] == 1.0
    assert report['stages_samples'] == {'lung': 5, 'tumor': 2}
    assert report['stages_seconds_per_sample'] == {'lung': 0.6, 'tumor': 2.0}
