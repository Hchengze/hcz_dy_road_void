from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import main
from road_void.config import ProcessingConfig, RoadVoidConfig
from road_void.dataset import apply_das_like_response, generate_synthetic_survey_dataset, save_synthetic_survey_dataset
from road_void.diffraction import detect_diffraction_features
from road_void.elastic3d import Elastic3DConfig
from road_void.elastic_bridge import build_local_elastic_validation_case
from road_void.inversion import run_joint_localization_evaluation, write_research_report
from road_void.scenario import build_default_subsurface_scenario, make_property_section_xz, make_property_volume
from road_void.workflow import run_location_workflow


def _small_config() -> RoadVoidConfig:
    cfg = RoadVoidConfig()
    cfg = replace(
        cfg,
        processing=replace(
            cfg.processing,
            scan_x_min=34.0,
            scan_x_max=50.0,
            scan_x_step=8.0,
            scan_y_min=4.0,
            scan_y_max=14.0,
            scan_y_step=5.0,
            scan_h_min=0.8,
            scan_h_max=3.8,
            scan_h_step=1.5,
            scan_vr_min=220.0,
            scan_vr_max=260.0,
            scan_vr_step=40.0,
        ),
    )
    return cfg


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run([sys.executable, "main.py", *args], check=True, capture_output=True, text=True, env=env)


def _workspace_tmp(name: str) -> Path:
    path = Path(".tmp_test_outputs") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_scenario_model_default_layers_and_volume():
    scenario = build_default_subsurface_scenario(_small_config())
    assert len(scenario.layers) == 3
    assert scenario.layers[0].label.startswith("asphalt")
    x, z, section = make_property_section_xz(scenario, nx=20, nz=12)
    assert section.shape == (z.size, x.size)
    vx, vy, vz, volume = make_property_volume(scenario, nx=12, ny=8, nz=10)
    assert volume.shape == (vx.size, vy.size, vz.size)
    assert np.all(np.isfinite(volume))


def test_research_dataset_shape_metadata_and_labels():
    cfg = _small_config()
    workflow = run_location_workflow(cfg)
    dataset = generate_synthetic_survey_dataset(cfg, workflow)
    assert dataset.data.shape == dataset.das_like_data.shape
    assert dataset.metadata["velocity_mode"] == cfg.velocity.velocity_model_type
    assert dataset.labels["true_anomaly_centers"]
    npz, meta = save_synthetic_survey_dataset(dataset, _workspace_tmp("research_dataset"))
    assert npz.exists()
    assert meta.exists()


def test_das_like_response_is_finite_and_changes_data():
    cfg = _small_config()
    synthetic = main.simulate_from_config(cfg)
    das = apply_das_like_response(synthetic.data, synthetic.geometry, cfg)
    assert das.data.shape == synthetic.data.shape
    assert np.all(np.isfinite(das.data))
    assert not np.allclose(das.data, synthetic.data)
    assert "strain_rate_like" in das.components


def test_diffraction_attribute_and_localization_evaluation():
    cfg = _small_config()
    workflow = run_location_workflow(cfg)
    diff = detect_diffraction_features(cfg, workflow)
    evaluation = run_joint_localization_evaluation(cfg, workflow)
    assert diff.attribute_gather.shape == workflow.residual[diff.shot_index].shape
    assert np.all(np.isfinite(diff.attribute_gather))
    assert diff.candidates
    assert evaluation.best_estimate
    assert evaluation.true_position is not None
    assert evaluation.total_location_error is not None


def test_elastic_bridge_detects_local_anomaly_bounds():
    case = build_local_elastic_validation_case(_small_config(), elastic_config=Elastic3DConfig(nx=40, ny=30, nz=24, nt=20))
    assert case.in_bounds
    assert case.mapped_cavities
    cav = case.mapped_cavities[0]
    assert 0.0 < cav.x0 < case.elastic_config.nx * case.elastic_config.dx


def test_research_report_can_be_generated():
    cfg = _small_config()
    workflow = run_location_workflow(cfg)
    scenario = build_default_subsurface_scenario(cfg)
    dataset = generate_synthetic_survey_dataset(cfg, workflow)
    diff = detect_diffraction_features(cfg, workflow)
    evaluation = run_joint_localization_evaluation(cfg, workflow)
    output = write_research_report(cfg, scenario, dataset, diff, evaluation, _workspace_tmp("research_report") / "research_report.md")
    text = output.read_text(encoding="utf-8")
    assert "合成数据摘要" in text
    assert "定位/反演结果" in text


def test_workflow_save_extra_generates_research_outputs():
    outdir = _workspace_tmp("workflow_research_extra")
    _run_cli(
        "workflow",
        "--save",
        "--save-extra",
        "--clean-output",
        "--outdir",
        str(outdir),
        "--scan-x-step",
        "8",
        "--scan-y-step",
        "5",
        "--scan-h-step",
        "1.5",
        "--scan-vr-step",
        "40",
    )
    assert (outdir / "02b_subsurface_model_xz.png").exists()
    assert (outdir / "03b_das_like_gather.png").exists()
    assert (outdir / "04b_diffraction_attribute.png").exists()
    assert (outdir / "05b_localization_error_summary.png").exists()
    assert (outdir / "research_report.md").exists()


def test_default_workflow_outputs_remain_controlled():
    outdir = _workspace_tmp("workflow_research_default")
    _run_cli(
        "workflow",
        "--save",
        "--clean-output",
        "--outdir",
        str(outdir),
        "--scan-x-step",
        "8",
        "--scan-y-step",
        "5",
        "--scan-h-step",
        "1.5",
        "--scan-vr-step",
        "40",
    )
    assert (outdir / "01_geometry_plan_sections.png").exists()
    assert not (outdir / "02b_subsurface_model_xz.png").exists()
    assert not (outdir / "research_report.md").exists()


def test_elastic_validate_cli_no_save_runs():
    completed = _run_cli("elastic-validate", "--no-save", "--elastic-nt", "40")
    assert "elastic-validate" in completed.stdout
    assert "局部坐标原点" in completed.stdout
