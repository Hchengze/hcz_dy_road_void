from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
import os
from pathlib import Path

import numpy as np

from main import build_parser, config_from_args
from road_void.workflow import simulate_from_config


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        [sys.executable, "main.py", *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_main_geometry_no_save_runs():
    completed = _run_cli("geometry", "--no-save")
    assert "关键参数" in completed.stdout


def test_main_forward_no_save_runs():
    completed = _run_cli("forward", "--no-save")
    assert "合成数据形状" in completed.stdout


def test_main_scan_no_save_runs():
    completed = _run_cli("scan", "--no-save")
    assert "最佳疑似异常体" in completed.stdout


def test_main_scan_parameter_override_runs():
    completed = _run_cli("scan", "--road-width", "30", "--cavity-depth", "2.5", "--noise-level", "0.1", "--no-save")
    assert "W=30.0" in completed.stdout
    assert "2.5" in completed.stdout


def test_main_tutorial_no_save_is_compact():
    completed = _run_cli("tutorial", "--no-save")
    assert "教学流程完成" in completed.stdout
    assert "敏感性分析" not in completed.stdout


def test_road_width_changes_geometry_and_times():
    parser = build_parser()
    args1 = parser.parse_args(["scan", "--road-width", "15", "--no-save"])
    args2 = parser.parse_args(["scan", "--road-width", "30", "--no-save"])
    cfg1 = config_from_args(args1)
    cfg2 = config_from_args(args2)
    geom1 = cfg1.to_geometry()
    geom2 = cfg2.to_geometry()
    assert geom1.road_width == 15.0
    assert geom2.road_width == 30.0
    assert not np.allclose(geom1.direct_times(240.0), geom2.direct_times(240.0))


def test_rayleigh_velocity_changes_travel_times():
    parser = build_parser()
    args = parser.parse_args(["scan", "--rayleigh-velocity", "220", "--no-save"])
    cfg = config_from_args(args)
    geom = cfg.to_geometry()
    t220 = geom.direct_times(220.0)
    t260 = geom.direct_times(260.0)
    assert not np.allclose(t220, t260)


def test_show_and_save_flags_do_not_conflict():
    completed = _run_cli("geometry", "--show", "--save", "--outdir", ".tmp_test_outputs/show_save")
    assert completed.returncode == 0


def test_minimal_examples_run():
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(
        [sys.executable, "examples/example_minimal_forward.py", "--no-save"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "合成数据形状" in completed.stdout


def test_readme_no_longer_recommends_config_as_main_entry():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "日常使用优先运行 `main.py`" in text
    assert "历史兼容" in text
    assert "python main.py scan --config" not in text
