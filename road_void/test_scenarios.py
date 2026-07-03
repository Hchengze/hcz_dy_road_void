"""参数组合回归测试场景。

这个模块不是新的算法入口，而是把“以后每轮修改都应覆盖哪些典型参数组合”
集中放在一个地方。测试和 ``tools/check_workflow_matrix.py`` 都复用这里的矩阵，
避免场景定义散落在多个测试文件里。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .config import CavityConfig, GeometryConfig, NoiseConfig, ProcessingConfig, RecordConfig, RoadVoidConfig, VelocityConfig, validate_config
from .dataset import generate_synthetic_survey_dataset
from .diffraction import detect_diffraction_features
from .inversion import run_joint_localization_evaluation
from .scenario import build_default_subsurface_scenario
from .workflow import run_location_workflow


@dataclass(frozen=True)
class MatrixScenario:
    """一个轻量 workflow 回归场景。

    overrides 按配置章节组织，字段名与 ``RoadVoidConfig`` 中 dataclass 字段一致。
    这里默认使用小几何、小扫描网格，目的是快速暴露参数同步、NaN/inf 和 warning 问题，
    不用于判断科研结论优劣。
    """

    name: str
    expected: str
    overrides: dict[str, dict[str, Any]]
    slow: bool = False


BASE_OVERRIDES: dict[str, dict[str, Any]] = {
    "geometry": {
        "road_width": 12.0,
        "road_length": 36.0,
        "channel_x_max": 36.0,
        "source_x_max": 36.0,
        "channel_spacing": 4.0,
        "source_spacing": 9.0,
    },
    "record": {
        "sampling_rate": 400.0,
        "duration": 0.45,
        "random_seed": 2027,
    },
    "velocity": {
        "velocity_model_type": "layered-effective",
        "rayleigh_velocity": 240.0,
        "source_frequency": 30.0,
        "layer_depths": (0.4, 1.5, 4.0),
        "layer_velocities": (180.0, 240.0, 320.0),
    },
    "cavity": {
        "shape": "sphere",
        "cavity_x": 20.0,
        "cavity_y": 6.0,
        "cavity_h": 1.8,
        "cavity_radius": 1.2,
        "scattering_strength": 1.0,
        "anomalies": None,
    },
    "noise": {
        "noise_level": 0.03,
        "traffic_noise_level": 0.015,
    },
    "processing": {
        "scan_x_min": 12.0,
        "scan_x_max": 28.0,
        "scan_x_step": 8.0,
        "scan_y_min": 2.0,
        "scan_y_max": 10.0,
        "scan_y_step": 4.0,
        "scan_h_min": 0.6,
        "scan_h_max": 3.6,
        "scan_h_step": 1.5,
        "scan_vr_min": 200.0,
        "scan_vr_max": 280.0,
        "scan_vr_step": 40.0,
        "top_k": 5,
    },
}


SCENARIO_MATRIX: list[MatrixScenario] = [
    MatrixScenario("default_layered_sphere", "正常运行，无项目代码 warning。", {}),
    MatrixScenario("uniform_sphere", "uniform 速度模式正常运行。", {"velocity": {"velocity_model_type": "uniform"}}),
    MatrixScenario(
        "layered_cylinder",
        "圆柱异常体进入 scenario/dataset/diffraction/inversion。",
        {"cavity": {"shape": "cylinder", "size_z": 3.0, "cavity_radius": 1.0}},
    ),
    MatrixScenario(
        "multi_anomaly",
        "多异常体字符串优先，并进入报告/metadata。",
        {"cavity": {"anomalies": "sphere:18,6,1.6,1.2,1.0;box:28,5,1.4,3,2,1,0.8"}},
    ),
    MatrixScenario(
        "scan_range_miss",
        "扫描范围不覆盖异常体时不崩溃，结果应解释为低可信或失败场景。",
        {"processing": {"scan_x_min": 0.0, "scan_x_max": 8.0}},
    ),
    MatrixScenario(
        "high_noise",
        "高噪声降低稳定性，但不应产生 RuntimeWarning 或 NaN。",
        {"noise": {"noise_level": 0.25, "traffic_noise_level": 0.12}},
    ),
    MatrixScenario(
        "layered_box",
        "box 形状轻量回归。",
        {"cavity": {"shape": "box", "size_x": 3.0, "size_y": 2.0, "size_z": 1.2}},
        slow=True,
    ),
    MatrixScenario(
        "ellipsoid",
        "ellipsoid 形状轻量回归。",
        {"cavity": {"shape": "ellipsoid", "size_x": 3.0, "size_y": 1.8, "size_z": 1.0}},
        slow=True,
    ),
    MatrixScenario("line", "line 形状轻量回归。", {"cavity": {"shape": "line", "size_x": 8.0, "azimuth": 0.0}}, slow=True),
    MatrixScenario("zone", "zone 作为条带异常轻量回归。", {"cavity": {"shape": "zone", "size_x": 8.0, "azimuth": 20.0}}, slow=True),
    MatrixScenario("short_record", "短记录长度不应导致数组空崩溃。", {"record": {"duration": 0.22}}, slow=True),
    MatrixScenario("wide_road", "宽道路几何同步到正演和扫描。", {"geometry": {"road_width": 24.0}, "cavity": {"cavity_y": 10.0}}, slow=True),
]


QUICK_SCENARIOS = tuple(s.name for s in SCENARIO_MATRIX if not s.slow)
EXTENDED_SCENARIOS = tuple(s.name for s in SCENARIO_MATRIX)


def scenario_by_name(name: str) -> MatrixScenario:
    for scenario in SCENARIO_MATRIX:
        if scenario.name == name:
            return scenario
    raise KeyError(f"未知矩阵场景: {name}")


def build_matrix_config(name: str) -> RoadVoidConfig:
    """构建矩阵场景配置，并立即做基础合法性检查。"""

    scenario = scenario_by_name(name)
    merged = _merge_overrides(BASE_OVERRIDES, scenario.overrides)
    cfg = RoadVoidConfig(
        geometry=GeometryConfig(**merged["geometry"]),
        record=RecordConfig(**merged["record"]),
        velocity=VelocityConfig(**merged["velocity"]),
        cavity=CavityConfig(**merged["cavity"]),
        noise=NoiseConfig(**merged["noise"]),
        processing=ProcessingConfig(**merged["processing"]),
    )
    validate_config(cfg)
    return cfg


def run_lightweight_workflow_case(name: str) -> dict[str, Any]:
    """运行轻量主链，用于测试参数组合是否能协同通过。

    返回对象包含 scenario、dataset、diffraction、localization 等关键结果。这里不画图、
    不写 outputs/workflow，避免测试逻辑污染主工作流输出。
    """

    cfg = build_matrix_config(name)
    scenario = build_default_subsurface_scenario(cfg)
    workflow = run_location_workflow(cfg)
    survey_dataset = generate_synthetic_survey_dataset(cfg, workflow)
    diffraction = detect_diffraction_features(cfg, workflow)
    localization = run_joint_localization_evaluation(cfg, workflow)
    _assert_finite_case(workflow.dataset.data, "synthetic data")
    _assert_finite_case(survey_dataset.das_like_data, "DAS-like data")
    _assert_finite_case(diffraction.attribute_gather, "diffraction attribute")
    return {
        "name": name,
        "config": cfg,
        "scenario": scenario,
        "workflow": workflow,
        "survey_dataset": survey_dataset,
        "diffraction": diffraction,
        "localization": localization,
    }


def _merge_overrides(base: dict[str, dict[str, Any]], override: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = {section: values.copy() for section, values in base.items()}
    for section, values in override.items():
        merged.setdefault(section, {}).update(values)
    return merged


def _assert_finite_case(array: np.ndarray, label: str) -> None:
    if array.size == 0:
        raise ValueError(f"{label} 为空，矩阵场景不应生成空数组。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 包含 NaN 或 inf。")
