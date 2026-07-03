from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import main
from road_void.config import ConfigError, validate_config
from road_void.test_scenarios import QUICK_SCENARIOS, build_matrix_config, run_lightweight_workflow_case


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run([sys.executable, "main.py", *args], check=True, capture_output=True, text=True, env=env)


def _tmp_outdir(name: str) -> Path:
    outdir = Path(".tmp_test_outputs") / name
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


@pytest.mark.parametrize("scenario_name", QUICK_SCENARIOS)
def test_quick_parameter_matrix_runs_without_python_warnings(scenario_name: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_lightweight_workflow_case(scenario_name)
    unexpected = [w for w in caught if issubclass(w.category, (RuntimeWarning, DeprecationWarning, FutureWarning, UserWarning))]
    assert unexpected == []

    cfg = result["config"]
    workflow = result["workflow"]
    survey_dataset = result["survey_dataset"]
    diffraction = result["diffraction"]
    localization = result["localization"]

    assert np.isfinite(cfg.effective_rayleigh_velocity())
    assert survey_dataset.metadata["velocity_mode"] == cfg.velocity.velocity_model_type
    assert survey_dataset.data.shape == workflow.dataset.data.shape
    assert diffraction.attribute_gather.size > 0
    assert np.all(np.isfinite(diffraction.attribute_gather))
    assert localization.best_estimate
    assert np.isfinite(localization.confidence_score)


def test_velocity_matrix_changes_effective_velocity_and_metadata():
    cfg1 = build_matrix_config("default_layered_sphere")
    cfg2 = build_matrix_config("default_layered_sphere")
    cfg2 = cfg2.__class__(
        geometry=cfg2.geometry,
        cavity=cfg2.cavity,
        record=cfg2.record,
        noise=cfg2.noise,
        processing=cfg2.processing,
        velocity=cfg2.velocity.__class__(
            **{**cfg2.velocity.__dict__, "layer_velocities": (160.0, 220.0, 360.0), "source_frequency": 60.0}
        ),
    )
    validate_config(cfg2)
    assert cfg1.velocity.velocity_model_type == "layered-effective"
    assert cfg2.velocity.velocity_model_type == "layered-effective"
    assert np.isfinite(cfg1.effective_rayleigh_velocity())
    assert np.isfinite(cfg2.effective_rayleigh_velocity())
    assert cfg1.effective_rayleigh_velocity() != cfg2.effective_rayleigh_velocity()


@pytest.mark.parametrize("shape", ["sphere", "box", "cylinder", "ellipsoid", "line", "zone"])
def test_all_supported_shapes_enter_config_and_scattering_points(shape: str):
    cfg = build_matrix_config("default_layered_sphere")
    cavity_kwargs = cfg.cavity.__dict__.copy()
    cavity_kwargs.update({"shape": shape, "anomalies": None, "size_x": 4.0, "size_y": 2.0, "size_z": 1.2})
    cfg = cfg.__class__(geometry=cfg.geometry, velocity=cfg.velocity, record=cfg.record, noise=cfg.noise, processing=cfg.processing, cavity=cfg.cavity.__class__(**cavity_kwargs))
    validate_config(cfg)
    cavities = cfg.to_cavities()
    assert cavities[0].shape == shape
    points, weights = cavities[0].scatter_points()
    assert points.shape[1] == 3
    assert points.shape[0] >= 1
    assert weights.shape[0] == points.shape[0]
    assert np.all(np.isfinite(points))
    assert np.isclose(np.sum(weights), 1.0)


def test_multi_anomaly_string_has_priority_and_reaches_report():
    result = run_lightweight_workflow_case("multi_anomaly")
    cfg = result["config"]
    assert len(cfg.to_cavities()) == 2
    assert result["survey_dataset"].metadata["anomalies"][0]["shape"] == "sphere"
    outdir = _tmp_outdir("matrix_multi_report")
    report = outdir / "report.md"
    from road_void.inversion import write_research_report

    write_research_report(cfg, result["scenario"], result["survey_dataset"], result["diffraction"], result["localization"], report)
    text = report.read_text(encoding="utf-8")
    assert "sphere" in text
    assert "box" in text


def test_workflow_output_manifest_and_no_split_wavefield_dir():
    outdir = _tmp_outdir("matrix_workflow_output")
    local_debug_before = Path("outputs/local_debug").exists()
    wavefield_before = Path("outputs/wavefield").exists()
    _run_cli(
        "workflow",
        "--save",
        "--clean-output",
        "--outdir",
        str(outdir),
        "--road-length",
        "36",
        "--channel-spacing",
        "4",
        "--source-spacing",
        "9",
        "--scan-x-step",
        "8",
        "--scan-y-step",
        "4",
        "--scan-h-step",
        "1.5",
        "--scan-vr-step",
        "40",
    )
    manifest = outdir / "output_manifest.txt"
    params = json.loads((outdir / "run_parameters.json").read_text(encoding="utf-8"))
    assert manifest.exists()
    assert (outdir / "06_wavefield_frame_scattered.png").exists()
    assert params["geometry"]["road_length"] == 36.0
    assert Path("outputs/local_debug").exists() == local_debug_before
    assert Path("outputs/wavefield").exists() == wavefield_before


def test_wavefield_plan_and_3d_no_save_smoke():
    _run_cli(
        "wavefield",
        "--wavefield-view",
        "plan",
        "--road-length",
        "36",
        "--channel-spacing",
        "4",
        "--source-spacing",
        "9",
        "--no-save",
    )
    _run_cli(
        "wavefield",
        "--wavefield-view",
        "3d",
        "--road-length",
        "36",
        "--channel-spacing",
        "4",
        "--source-spacing",
        "9",
        "--no-save",
    )


def test_command_outdir_keeps_wavefield_in_workflow_tree():
    parser = main.build_parser()
    args = parser.parse_args(["wavefield", "--save"])
    assert main.command_outdir(args, "wavefield") == Path("outputs/workflow")
