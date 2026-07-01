from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from road_void.anomaly import Cavity
from road_void.elastic3d import Elastic3DConfig, build_elastic3d_model, check_cfl, run_elastic3d


def test_elastic3d_builds_small_model_and_cfl_is_stable():
    cfg = Elastic3DConfig(nx=24, ny=20, nz=16, nt=12)
    model = build_elastic3d_model(cfg)
    cfl = check_cfl(cfg, model)
    assert model.vp.shape == (24, 20, 16)
    assert 0.0 < cfl < 0.45


def test_elastic3d_cfl_rejects_unstable_dt():
    cfg = Elastic3DConfig(nx=24, ny=20, nz=16, dt=0.003, nt=4)
    model = build_elastic3d_model(cfg)
    with pytest.raises(ValueError, match="CFL"):
        check_cfl(cfg, model)


def test_elastic3d_runs_and_returns_gather_shape():
    cfg = Elastic3DConfig(nx=24, ny=20, nz=16, nt=40)
    result = run_elastic3d(cfg)
    assert result.gather.shape[0] == cfg.nt
    assert result.gather.shape[1] > 0
    assert result.cfl > 0
    assert result.snapshots


def test_elastic3d_anomaly_changes_model_and_output():
    base = Elastic3DConfig(nx=28, ny=22, nz=16, nt=80, with_anomaly=False)
    cav = Cavity(7.0, 5.0, 2.0, radius=1.2, shape="cylinder", size_x=1.2, size_z=2.5)
    no_anomaly = run_elastic3d(base)
    with_anomaly = run_elastic3d(Elastic3DConfig(nx=28, ny=22, nz=16, nt=80, with_anomaly=True), [cav])
    assert not np.allclose(no_anomaly.model.vp, with_anomaly.model.vp)
    assert not np.allclose(no_anomaly.gather, with_anomaly.gather)


def test_main_elastic3d_tiny_cli_runs():
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(
        [
            sys.executable,
            "main.py",
            "elastic3d",
            "--no-save",
            "--nx",
            "24",
            "--ny",
            "20",
            "--nz",
            "16",
            "--elastic-nt",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "CFL number" in completed.stdout
