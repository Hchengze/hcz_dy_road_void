from __future__ import annotations

import argparse
import warnings

import numpy as np
import pytest

import main
from road_void.config import ConfigError, validate_config
from road_void.test_scenarios import build_matrix_config, run_lightweight_workflow_case


def test_scan_range_miss_is_diagnostic_case_not_crash():
    result = run_lightweight_workflow_case("scan_range_miss")
    evaluation = result["localization"]
    assert evaluation.best_estimate
    assert np.isfinite(evaluation.confidence_score)


def test_high_noise_and_short_record_do_not_emit_runtime_warning():
    for name in ["high_noise", "short_record"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_lightweight_workflow_case(name)
        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert np.all(np.isfinite(result["workflow"].dataset.data))


def test_low_scattering_strength_still_returns_low_confidence_result():
    cfg = build_matrix_config("default_layered_sphere")
    cavity = cfg.cavity.__class__(**{**cfg.cavity.__dict__, "scattering_strength": 0.01})
    cfg = cfg.__class__(geometry=cfg.geometry, velocity=cfg.velocity, record=cfg.record, noise=cfg.noise, processing=cfg.processing, cavity=cavity)
    validate_config(cfg)
    # 低散射强度是困难但合理的物理场景，不应在配置层被拒绝。
    assert cfg.to_cavities()[0].scattering_strength > 0


def test_invalid_layer_velocities_string_has_clear_error():
    parser = main.build_parser()
    args = parser.parse_args(["workflow", "--velocity-mode", "layered-effective", "--layer-velocities", "180,abc,320", "--no-save"])
    with pytest.raises(argparse.ArgumentTypeError, match="浮点数"):
        main.build_road_void_config_from_args(args)


def test_invalid_anomalies_string_has_clear_error():
    parser = main.build_parser()
    args = parser.parse_args(["workflow", "--anomalies", "sphere:20,abc,1.5,1.0,1.0", "--no-save"])
    cfg = main.build_road_void_config_from_args(args)
    with pytest.raises(ConfigError, match="数值"):
        cfg.to_cavities()


def test_invalid_velocity_and_radius_are_rejected():
    cfg = build_matrix_config("default_layered_sphere")
    bad_velocity = cfg.__class__(
        geometry=cfg.geometry,
        cavity=cfg.cavity,
        record=cfg.record,
        noise=cfg.noise,
        processing=cfg.processing,
        velocity=cfg.velocity.__class__(**{**cfg.velocity.__dict__, "rayleigh_velocity": 0.0}),
    )
    with pytest.raises(ConfigError, match="rayleigh_velocity"):
        validate_config(bad_velocity)

    bad_radius = cfg.__class__(
        geometry=cfg.geometry,
        velocity=cfg.velocity,
        record=cfg.record,
        noise=cfg.noise,
        processing=cfg.processing,
        cavity=cfg.cavity.__class__(**{**cfg.cavity.__dict__, "cavity_radius": -2.0}),
    )
    with pytest.raises(ConfigError, match="cavity_radius"):
        validate_config(bad_radius)


@pytest.mark.parametrize(
    "spec",
    [
        "sphere:20,6,1.8,-1.0,1.0",
        "box:20,6,1.8,3,0,1,1.0",
        "cylinder:20,6,1.8,1.0,-2,1.0",
        "ellipsoid:20,6,1.8,3,2,-1,1.0",
        "line:20,6,1.8,0,0,1.0",
    ],
)
def test_invalid_anomaly_sizes_are_rejected(spec: str):
    parser = main.build_parser()
    args = parser.parse_args(["workflow", "--anomalies", spec, "--no-save"])
    cfg = main.build_road_void_config_from_args(args)
    with pytest.raises(ConfigError, match="必须为正数"):
        cfg.to_cavities()
