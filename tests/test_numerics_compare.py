from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from road_void.numerics.compare import compare_1d_wave_methods
from road_void.numerics.fdtd import run_fdtd1d_wave_demo


def test_fdtd1d_wave_demo_runs_and_is_finite():
    result = run_fdtd1d_wave_demo(duration=0.04, dt=0.0005, save=False, show=False)
    assert result.receiver_trace.ndim == 1
    assert np.all(np.isfinite(result.receiver_trace))
    assert 0.0 < result.cfl < 1.0


def test_compare_1d_wave_methods_runs_and_returns_metrics():
    outdir = Path(".tmp_test_outputs/numerics_compare")
    outdir.mkdir(parents=True, exist_ok=True)
    result = compare_1d_wave_methods(duration=0.08, dt=0.0005, outdir=outdir, save=True, show=False)
    assert np.isfinite(result.metrics["fdtd_arrival_time"])
    assert np.isfinite(result.metrics["fem_vs_fdtd_l2"])
    assert np.isfinite(result.metrics["sem_vs_fdtd_l2"])
    assert (outdir / "compare_1d_traces.png").exists()
    assert (outdir / "compare_1d_wavefields.png").exists()
    metrics = json.loads((outdir / "compare_1d_metrics.json").read_text(encoding="utf-8"))
    assert "fdtd_arrival_time" in metrics
    assert "sem_peak_amplitude" in metrics


def test_compare_1d_arrivals_are_same_order():
    result = compare_1d_wave_methods(duration=0.24, dt=0.0005, save=False, show=False)
    fdtd = result.metrics["fdtd_arrival_time"]
    fem = result.metrics["fem_arrival_time"]
    sem = result.metrics["sem_arrival_time"]
    assert abs(fdtd - fem) < 0.08
    assert abs(fdtd - sem) < 0.08
