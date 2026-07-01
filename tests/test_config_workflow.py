from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from examples.example_09_parameter_sensitivity import main as sensitivity_main
from road_void.config import ConfigError, load_config
from road_void.workflow import run_location_workflow, simulate_from_config


TEST_OUTPUT_ROOT = Path(".tmp_test_outputs")


def _tiny_config():
    cfg = load_config("configs/default_road_void.yaml")
    cfg = replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            road_width=10.0,
            road_length=30.0,
            channel_x_min=0.0,
            channel_x_max=30.0,
            channel_spacing=3.0,
            source_x_min=0.0,
            source_x_max=30.0,
            source_spacing=6.0,
            source_y=10.0,
        ),
        cavity=replace(cfg.cavity, cavity_x=15.0, cavity_y=5.0, cavity_h=1.8),
        record=replace(cfg.record, duration=0.55, random_seed=44),
        processing=replace(
            cfg.processing,
            scan_x_min=9.0,
            scan_x_max=21.0,
            scan_x_step=3.0,
            scan_y_min=2.0,
            scan_y_max=8.0,
            scan_y_step=3.0,
            scan_h_min=0.8,
            scan_h_max=3.0,
            scan_h_step=1.0,
            scan_vr_min=220.0,
            scan_vr_max=260.0,
            scan_vr_step=20.0,
            top_k=4,
        ),
    )
    return cfg


def test_default_config_loads_and_builds_geometry():
    cfg = load_config("configs/default_road_void.yaml")
    geom = cfg.to_geometry()
    assert cfg.geometry.road_width == 15.0
    assert geom.n_channels == 81
    assert geom.n_shots == 21


def test_four_and_six_lane_configs_build_geometry():
    four = load_config("configs/four_lane_demo.yaml").to_geometry()
    six = load_config("configs/six_lane_demo.yaml").to_geometry()
    assert four.road_width == 15.0
    assert six.road_width == 28.0
    assert six.n_channels > four.n_channels


def test_config_parameters_reach_forward_model():
    cfg = _tiny_config()
    dataset = simulate_from_config(cfg)
    assert dataset.config.rayleigh_velocity == cfg.velocity.rayleigh_velocity
    assert dataset.config.noise_std == cfg.noise.noise_level
    assert dataset.data.shape == (cfg.to_geometry().n_shots, cfg.to_geometry().n_times, cfg.to_geometry().n_channels)


def test_config_parameters_reach_scan_grid():
    cfg = _tiny_config()
    grid = cfg.to_scan_grid()
    assert np.isclose(grid.x[1] - grid.x[0], cfg.processing.scan_x_step)
    assert np.isclose(grid.h[1] - grid.h[0], cfg.processing.scan_h_step)


def test_no_cavity_demo_has_low_relative_confidence():
    cfg = load_config("configs/no_cavity_demo.yaml")
    cfg = replace(cfg, processing=_tiny_config().processing, geometry=_tiny_config().geometry, record=_tiny_config().record)
    result = run_location_workflow(cfg).scan_result
    assert result.confidence < 0.05


def test_parameter_changes_affect_data_or_location():
    cfg = _tiny_config()
    base = run_location_workflow(cfg)
    wider_cfg = replace(cfg, geometry=replace(cfg.geometry, road_width=14.0, source_y=14.0))
    wider_data = simulate_from_config(wider_cfg)
    noisy = simulate_from_config(replace(cfg, noise=replace(cfg.noise, noise_level=0.2)))
    assert not np.allclose(base.dataset.direct_times, wider_data.direct_times)
    assert not np.allclose(base.dataset.data, noisy.data)


def test_missing_required_config_section_has_clear_error():
    TEST_OUTPUT_ROOT.mkdir(exist_ok=True)
    bad = TEST_OUTPUT_ROOT / "bad.yaml"
    bad.write_text("geometry:\n  road_width: 15\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="缺少关键章节"):
        load_config(bad)


def test_velocity_model_uniform_and_layered_build():
    uniform = load_config("configs/default_road_void.yaml").to_velocity_model()
    layered = load_config("configs/six_lane_demo.yaml").to_velocity_model()
    assert len(uniform.layers) == 1
    assert len(layered.layers) == 3


def test_sensitivity_example_runs_small_scale():
    out = TEST_OUTPUT_ROOT / "sensitivity"
    sensitivity_main(output_dir=out, config=_tiny_config())
    assert (out / "parameter_sensitivity_results.csv").exists()
    assert (out / "noise_vs_confidence.png").exists()


@pytest.mark.parametrize("case", ["geometry", "velocity", "scan", "tutorial"])
def test_main_config_entrypoints(case):
    out = TEST_OUTPUT_ROOT / f"main_{case}"
    cmd = [
        sys.executable,
        "main.py",
        case,
        "--config",
        "configs/default_road_void.yaml",
        "--output-dir",
        str(out),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert completed.returncode == 0
