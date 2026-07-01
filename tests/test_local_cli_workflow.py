from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
import os
from pathlib import Path

import numpy as np

import main
from main import build_args_from_local_config, build_parser, config_from_args
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


def test_main_without_subcommand_runs_default_workflow():
    completed = _run_cli("--no-save")
    assert "workflow" in completed.stdout
    assert "Step 1" in completed.stdout
    assert "Step 7" in completed.stdout


def test_local_debug_config_builds_workflow_args():
    args = build_args_from_local_config("workflow")
    assert args.command == "workflow"
    assert hasattr(args, "road_width")


def test_local_debug_config_builds_scan_args():
    args = build_args_from_local_config("scan")
    assert args.command == "scan"
    assert args.scan_mode == main.LOCAL_SCAN_PARAMS["scan_mode"]


def test_local_debug_config_builds_elastic3d_args():
    args = build_args_from_local_config("elastic3d")
    assert args.command == "elastic3d"
    assert args.elastic_space_order == main.LOCAL_ELASTIC3D_PARAMS["elastic_space_order"]


def test_main_workflow_no_save_runs():
    completed = _run_cli("workflow", "--no-save")
    assert "workflow" in completed.stdout
    assert "Step 5" in completed.stdout
    assert "合成数据形状" in completed.stdout


def test_main_workflow_parameter_override_runs():
    completed = _run_cli("workflow", "--road-width", "30", "--cavity-depth", "2.5", "--noise-level", "0.1", "--no-save")
    assert "W=30.0" in completed.stdout
    assert "2.5" in completed.stdout
    assert "noise=0.100" in completed.stdout


def test_main_all_alias_runs_workflow():
    completed = _run_cli("all", "--no-save")
    assert "workflow" in completed.stdout
    assert "Step 7" in completed.stdout


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


def test_main_scan_modes_run_with_coarse_grid():
    common = [
        "--scan-x-step",
        "6",
        "--scan-y-step",
        "4",
        "--scan-h-step",
        "1.6",
        "--scan-vr-step",
        "40",
        "--no-save",
    ]
    for mode in ["joint", "single-shot", "compare"]:
        extra = ["--shot-index", "5"] if mode == "single-shot" else []
        completed = _run_cli("scan", "--scan-mode", mode, *extra, *common)
        assert "最佳疑似异常体" in completed.stdout


def test_wavefield_animate_save_generates_gif():
    outdir = Path(".tmp_test_outputs/wavefield_gif")
    completed = _run_cli("wavefield", "--animate", "--save", "--frames", "3", "--outdir", str(outdir))
    assert completed.returncode == 0
    assert (outdir / "kinematic_wavefield.gif").exists()


def test_wavefield_save_generates_physical_key_frames():
    outdir = Path(".tmp_test_outputs/wavefield_frames")
    completed = _run_cli("wavefield", "--save", "--outdir", str(outdir))
    assert "early: 直达波刚离开震源" in completed.stdout
    assert "hit_cavity" in completed.stdout
    assert "scattered" in completed.stdout
    assert (outdir / "wavefield_frame_early.png").exists()
    assert (outdir / "wavefield_frame_hit_cavity.png").exists()
    assert (outdir / "wavefield_frame_scattered.png").exists()


def test_workflow_cylinder_anomaly_runs_with_coarse_scan():
    completed = _run_cli(
        "workflow",
        "--anomalies",
        "cylinder:42,8.5,2.2,2.0,4.0,1.0",
        "--scan-x-step",
        "6",
        "--scan-y-step",
        "4",
        "--scan-h-step",
        "1.6",
        "--scan-vr-step",
        "40",
        "--no-save",
    )
    assert "异常体数量=1" in completed.stdout
    assert "shape 仅表示等效散射几何" in completed.stdout


def test_main_tutorial_no_save_is_compact():
    completed = _run_cli("tutorial", "--no-save")
    assert "教学流程完成" in completed.stdout
    assert "敏感性分析" not in completed.stdout


def test_workflow_no_save_does_not_run_sensitivity_or_animation():
    completed = _run_cli("workflow", "--no-save")
    assert "parameter_sensitivity_results.csv" not in completed.stdout
    assert "06_kinematic_wavefield.gif" in completed.stdout
    assert "未生成动画" in completed.stdout


def test_workflow_and_scan_share_parameter_mapping():
    parser = build_parser()
    workflow_args = parser.parse_args(["workflow", "--road-width", "30", "--cavity-depth", "2.5", "--noise-level", "0.1", "--no-save"])
    scan_args = parser.parse_args(["scan", "--road-width", "30", "--cavity-depth", "2.5", "--noise-level", "0.1", "--no-save"])
    workflow_cfg = config_from_args(workflow_args)
    scan_cfg = config_from_args(scan_args)
    assert workflow_cfg.geometry.road_width == scan_cfg.geometry.road_width == 30.0
    assert workflow_cfg.cavity.cavity_h == scan_cfg.cavity.cavity_h == 2.5
    assert workflow_cfg.noise.noise_level == scan_cfg.noise.noise_level == 0.1


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
    assert "VSCode" in text
    assert "USE_LOCAL_DEBUG_CONFIG" in text
    assert "python main.py\n" in text
    assert "python main.py workflow" in text
    assert "历史兼容" in text
    assert "python main.py scan --config" not in text
    assert "layered-effective" in text
    assert "不是完整弹性波场" in text
    assert "elastic3d" in text


def test_docs_explain_elastic3d_is_small_scale_prototype():
    text = Path("docs/elastic_3d_forward_research_zh.md").read_text(encoding="utf-8")
    assert "小尺度" in text
    assert "velocity-stress" in text
    assert "CFL" in text
    assert "不替代默认绕射扫描定位流程" in text
