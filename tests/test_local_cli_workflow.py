from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
import os
from pathlib import Path

import numpy as np

import main
from main import build_args_from_local_config, build_parser, build_road_void_config_from_args, config_from_args
from road_void.visualization import compute_direct_wavefield_snapshot
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
    assert args.outdir == "outputs/workflow"
    assert main.LOCAL_OUTPUT["outdir"] != "outputs/local_debug"


def test_wavefield_default_outdir_is_workflow_tree():
    parser = build_parser()
    args = parser.parse_args(["wavefield", "--save"])
    assert main.command_outdir(args, "wavefield") == Path("outputs/workflow")


def test_local_debug_config_builds_scan_args():
    args = build_args_from_local_config("scan")
    assert args.command == "scan"
    assert args.scan_mode == main.LOCAL_WORKFLOW["scan"]["scan_mode"]


def test_local_debug_config_builds_elastic3d_args():
    args = build_args_from_local_config("elastic3d")
    assert args.command == "elastic3d"
    assert args.elastic_space_order == main.LOCAL_ELASTIC3D["elastic_space_order"]


def test_local_debug_config_builds_numerics_args():
    args = build_args_from_local_config("numerics-demo")
    assert args.command == "numerics-demo"
    assert args.method == main.LOCAL_NUMERICS["method"]


def test_local_debug_config_builds_numerics_compare_args():
    args = build_args_from_local_config("numerics-compare")
    assert args.command == "numerics-compare"
    assert args.numerics_velocity == main.LOCAL_NUMERICS_COMPARE["numerics_velocity"]


def test_local_geometry_velocity_anomaly_flow_into_single_config():
    args = build_args_from_local_config("workflow")
    cfg = build_road_void_config_from_args(args)
    cavities = cfg.to_cavities()
    assert cfg.geometry.road_width == main.LOCAL_WORKFLOW["geometry"]["road_width"]
    assert cfg.geometry.road_length == main.LOCAL_WORKFLOW["geometry"]["road_length"]
    assert cfg.velocity.velocity_model_type == main.LOCAL_WORKFLOW["velocity"]["velocity_mode"]
    assert cavities
    if not main.LOCAL_WORKFLOW["anomaly"].get("anomalies"):
        assert cavities[0].shape == main.LOCAL_WORKFLOW["anomaly"]["cavity_shape"]
        assert cavities[0].x0 == main.LOCAL_WORKFLOW["anomaly"]["cavity_x"]


def test_single_cylinder_args_flow_to_config_and_geometry():
    parser = build_parser()
    args = parser.parse_args([
        "workflow",
        "--road-width",
        "30",
        "--road-length",
        "100",
        "--cavity-shape",
        "cylinder",
        "--cavity-x",
        "50",
        "--cavity-y",
        "10",
        "--cavity-depth",
        "3",
        "--cavity-size-z",
        "5",
        "--no-save",
    ])
    cfg = build_road_void_config_from_args(args)
    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()
    assert geom.road_width == 30.0
    assert geom.shot_y == 30.0
    assert cavities[0].shape == "cylinder"
    assert cavities[0].x0 == 50.0
    assert cavities[0].size_z == 5.0


def test_anomalies_string_has_priority_over_single_cavity_args():
    parser = build_parser()
    args = parser.parse_args([
        "workflow",
        "--cavity-x",
        "10",
        "--anomalies",
        "sphere:42,8.5,2.2,2.0,1.0;box:58,6,1.5,4,3,1,0.8",
        "--no-save",
    ])
    cfg = build_road_void_config_from_args(args)
    cavities = cfg.to_cavities()
    assert len(cavities) == 2
    assert [c.shape for c in cavities] == ["sphere", "box"]
    assert cavities[0].x0 == 42.0


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


def test_main_numerics_demo_no_save_runs():
    completed = _run_cli("numerics-demo", "--method", "all", "--no-save")
    assert "FEM 1D 完成" in completed.stdout
    assert "SEM 1D 完成" in completed.stdout
    assert "BEM 2D 完成" in completed.stdout


def test_main_numerics_compare_no_save_runs():
    completed = _run_cli("numerics-compare", "--no-save")
    assert "numerics-compare" in completed.stdout
    assert "fdtd_arrival_time" in completed.stdout


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
    assert (outdir / "06_kinematic_wavefield.gif").exists()


def test_wavefield_save_generates_physical_key_frames():
    outdir = Path(".tmp_test_outputs/wavefield_frames")
    completed = _run_cli("wavefield", "--save", "--outdir", str(outdir))
    assert "early: 直达波刚离开震源" in completed.stdout
    assert "hit_cavity" in completed.stdout
    assert "scattered" in completed.stdout
    assert (outdir / "06_wavefield_frame_early.png").exists()
    assert (outdir / "06_wavefield_frame_hit_cavity.png").exists()
    assert (outdir / "06_wavefield_frame_scattered.png").exists()
    assert (outdir / "06_wavefield_velocity_context.png").exists()
    assert (outdir / "output_manifest.txt").exists()


def test_layered_effective_wavefield_uses_effective_velocity_in_radius():
    parser = build_parser()
    uniform_args = parser.parse_args(["wavefield", "--velocity-mode", "uniform", "--no-save"])
    layered_args = parser.parse_args(["wavefield", "--velocity-mode", "layered-effective", "--layer-velocities", "180,240,320", "--no-save"])
    uniform_cfg = build_road_void_config_from_args(uniform_args)
    layered_cfg = build_road_void_config_from_args(layered_args)
    geom = uniform_cfg.to_geometry()
    x = np.linspace(float(geom.channel_x[0]), float(geom.channel_x[-1]), 8)
    y = np.linspace(geom.fiber_y, float(geom.shot_y), 6)
    xx, yy = np.meshgrid(x, y)
    _, radius_uniform = compute_direct_wavefield_snapshot(geom, 0, uniform_cfg.effective_rayleigh_velocity(), 0.12, uniform_cfg.record.t0, xx, yy, 1.0)
    _, radius_layered = compute_direct_wavefield_snapshot(geom, 0, layered_cfg.effective_rayleigh_velocity(), 0.12, layered_cfg.record.t0, xx, yy, 1.0)
    assert layered_cfg.effective_rayleigh_velocity() != uniform_cfg.effective_rayleigh_velocity()
    assert radius_layered != radius_uniform


def test_wavefield_layered_cli_prints_velocity_mode_and_vreff():
    completed = _run_cli("wavefield", "--velocity-mode", "layered-effective", "--no-save")
    assert "velocity_mode=layered-effective" in completed.stdout
    assert "VR_eff" in completed.stdout


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
    assert "输出写入 outputs/workflow" in completed.stdout
    assert "未生成动画" in completed.stdout


def test_workflow_default_saved_png_count_is_controlled():
    outdir = Path(".tmp_test_outputs/workflow_controlled")
    completed = _run_cli(
        "workflow",
        "--save",
        "--clean-output",
        "--outdir",
        str(outdir),
        "--scan-x-step",
        "6",
        "--scan-y-step",
        "4",
        "--scan-h-step",
        "1.6",
        "--scan-vr-step",
        "40",
    )
    assert completed.returncode == 0
    pngs = sorted(p.name for p in outdir.glob("*.png"))
    assert pngs == [
        "01_geometry_3d.png",
        "01_geometry_plan_sections.png",
        "02_velocity_model.png",
        "03_forward_gather.png",
        "04_diffraction_path.png",
        "04_gather_with_curves.png",
        "05_residual_best_curve.png",
        "05_scan_score_slices.png",
    ]
    assert not (outdir / "single_shot_vs_joint.png").exists()
    assert not any(p.name.startswith("06_wavefield") for p in outdir.glob("*.png"))
    assert not (outdir / "06_kinematic_wavefield.gif").exists()
    assert (outdir / "output_manifest.txt").exists()


def test_clean_output_only_cleans_current_outdir():
    outdir = Path(".tmp_test_outputs/clean_current")
    sibling = Path(".tmp_test_outputs/clean_sibling")
    outdir.mkdir(parents=True, exist_ok=True)
    sibling.mkdir(parents=True, exist_ok=True)
    (outdir / "old.png").write_text("old", encoding="utf-8")
    (sibling / "keep.png").write_text("keep", encoding="utf-8")
    _run_cli("geometry", "--save", "--clean-output", "--outdir", str(outdir))
    assert not (outdir / "old.png").exists()
    assert (sibling / "keep.png").exists()


def test_multishot_wavefield_static_outputs_are_limited():
    outdir = Path(".tmp_test_outputs/wavefield_multishot_static")
    completed = _run_cli(
        "wavefield",
        "--wavefield-mode",
        "multi-shot",
        "--wavefield-shot-indices",
        "0,5,10",
        "--save",
        "--clean-output",
        "--outdir",
        str(outdir),
    )
    assert "multi-shot wavefield 选择炮号" in completed.stdout
    frames = sorted(outdir.glob("06_multishot_frame_shot*.png"))
    assert len(frames) == 3
    assert not (outdir / "06_multishot_wavefield.gif").exists()


def test_multishot_wavefield_animate_generates_single_gif():
    outdir = Path(".tmp_test_outputs/wavefield_multishot_gif")
    completed = _run_cli(
        "wavefield",
        "--wavefield-mode",
        "multi-shot",
        "--wavefield-shot-indices",
        "0,5",
        "--animate",
        "--frames",
        "6",
        "--save",
        "--clean-output",
        "--outdir",
        str(outdir),
    )
    assert completed.returncode == 0
    assert (outdir / "06_multishot_wavefield.gif").exists()


def test_workflow_animate_generates_step6_outputs_only_when_requested():
    outdir = Path(".tmp_test_outputs/workflow_animate")
    completed = _run_cli(
        "workflow",
        "--animate",
        "--frames",
        "4",
        "--save",
        "--clean-output",
        "--outdir",
        str(outdir),
        "--scan-x-step",
        "6",
        "--scan-y-step",
        "4",
        "--scan-h-step",
        "1.6",
        "--scan-vr-step",
        "40",
    )
    assert completed.returncode == 0
    assert (outdir / "06_wavefield_frame_early.png").exists()
    assert (outdir / "06_wavefield_frame_hit_cavity.png").exists()
    assert (outdir / "06_wavefield_frame_scattered.png").exists()
    assert (outdir / "06_kinematic_wavefield.gif").exists()
    manifest_text = (outdir / "output_manifest.txt").read_text(encoding="utf-8")
    assert "06_wavefield_frame_early.png" in manifest_text
    assert "06_kinematic_wavefield.gif" in manifest_text


def test_workflow_and_scan_share_parameter_mapping():
    parser = build_parser()
    workflow_args = parser.parse_args(["workflow", "--road-width", "30", "--cavity-depth", "2.5", "--noise-level", "0.1", "--no-save"])
    scan_args = parser.parse_args(["scan", "--road-width", "30", "--cavity-depth", "2.5", "--noise-level", "0.1", "--no-save"])
    workflow_cfg = config_from_args(workflow_args)
    scan_cfg = config_from_args(scan_args)
    assert workflow_cfg.geometry.road_width == scan_cfg.geometry.road_width == 30.0
    assert workflow_cfg.cavity.cavity_h == scan_cfg.cavity.cavity_h == 2.5
    assert workflow_cfg.noise.noise_level == scan_cfg.noise.noise_level == 0.1


def test_velocity_mode_uniform_and_layered_models_differ():
    parser = build_parser()
    uniform_args = parser.parse_args(["workflow", "--velocity-mode", "uniform", "--no-save"])
    layered_args = parser.parse_args(["workflow", "--velocity-mode", "layered-effective", "--no-save"])
    uniform_cfg = build_road_void_config_from_args(uniform_args)
    layered_cfg = build_road_void_config_from_args(layered_args)
    assert len(uniform_cfg.to_velocity_model().layers) == 1
    assert len(layered_cfg.to_velocity_model().layers) == 3
    assert layered_cfg.effective_rayleigh_velocity() != uniform_cfg.effective_rayleigh_velocity()


def test_layer_velocities_change_effective_velocity_and_times():
    parser = build_parser()
    args1 = parser.parse_args(["forward", "--velocity-mode", "layered-effective", "--layer-velocities", "180,240,320", "--no-save"])
    args2 = parser.parse_args(["forward", "--velocity-mode", "layered-effective", "--layer-velocities", "220,260,360", "--no-save"])
    cfg1 = build_road_void_config_from_args(args1)
    cfg2 = build_road_void_config_from_args(args2)
    assert cfg1.effective_rayleigh_velocity() != cfg2.effective_rayleigh_velocity()
    t1 = cfg1.to_geometry().direct_times(cfg1.effective_rayleigh_velocity())
    t2 = cfg2.to_geometry().direct_times(cfg2.effective_rayleigh_velocity())
    assert not np.allclose(t1, t2)


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
