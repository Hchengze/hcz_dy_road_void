"""多炮联合定位评估与科研报告。

当前定位/反演仍以已有 joint scan 为主。本模块不实现完整 FWI，而是把
best estimate、真值、误差、置信度和不确定性整理成科研记录。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import RoadVoidConfig
from .dataset import ResearchSurveyDataset
from .diffraction import DiffractionDetectionResult
from .scenario import RoadSubsurfaceScenario
from .workflow import WorkflowResult


@dataclass(frozen=True)
class LocalizationEvaluation:
    """多炮联合定位评估结果。"""

    best_estimate: dict[str, float]
    true_position: dict[str, float] | None
    location_error_x: float | None
    location_error_y: float | None
    location_error_depth: float | None
    total_location_error: float | None
    confidence_score: float
    uncertainty_bounds: dict[str, tuple[float, float]]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_joint_localization_evaluation(config: RoadVoidConfig, workflow: WorkflowResult) -> LocalizationEvaluation:
    """根据 joint scan 输出真值对比和误差指标。"""

    best = workflow.scan_result.best
    cavities = config.to_cavities()
    true = None
    dx = dy = dh = total = None
    if cavities:
        c = cavities[0]
        true = {"x": c.x0, "y": c.y0, "depth": c.h}
        dx = best.x0 - c.x0
        dy = best.y0 - c.y0
        dh = best.h - c.h
        total = float(np.sqrt(dx**2 + dy**2 + dh**2))
    return LocalizationEvaluation(
        best_estimate={"x": best.x0, "y": best.y0, "depth": best.h, "velocity": best.velocity, "score": best.score},
        true_position=true,
        location_error_x=None if dx is None else float(dx),
        location_error_y=None if dy is None else float(dy),
        location_error_depth=None if dh is None else float(dh),
        total_location_error=total,
        confidence_score=float(workflow.scan_result.confidence),
        uncertainty_bounds=workflow.scan_result.uncertainty,
        notes="当前评估基于运动学 joint scan。x 通常更稳定，y-h 在单侧 DAS 孔径下仍可能耦合。",
    )


def plot_localization_error_summary(
    evaluation: LocalizationEvaluation,
    output: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制定位误差柱状图。"""

    labels = ["x error", "y error", "depth error", "total"]
    values = [
        evaluation.location_error_x or 0.0,
        evaluation.location_error_y or 0.0,
        evaluation.location_error_depth or 0.0,
        evaluation.total_location_error or 0.0,
    ]
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    plt.axhline(0.0, color="k", lw=0.8)
    plt.ylabel("误差 (m)")
    plt.title(f"05b 定位误差摘要；confidence={evaluation.confidence_score:.4f}")
    _finish(output, save, show, dpi)


def plot_uncertainty_summary(
    evaluation: LocalizationEvaluation,
    output: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制 x/y/h/VR 不确定性范围。"""

    keys = list(evaluation.uncertainty_bounds.keys())
    centers = []
    widths = []
    for key in keys:
        lo, hi = evaluation.uncertainty_bounds[key]
        centers.append(0.5 * (lo + hi))
        widths.append(hi - lo)
    plt.figure(figsize=(8, 4.8))
    plt.bar(keys, widths, color="tab:purple")
    plt.ylabel("范围宽度")
    plt.title("05c 不确定性范围宽度：反映受限孔径下的非唯一性")
    for i, c in enumerate(centers):
        plt.text(i, widths[i], f"center={c:.2f}", ha="center", va="bottom", fontsize=8)
    _finish(output, save, show, dpi)


def write_research_report(
    config: RoadVoidConfig,
    scenario: RoadSubsurfaceScenario,
    dataset: ResearchSurveyDataset,
    diffraction: DiffractionDetectionResult,
    evaluation: LocalizationEvaluation,
    output: str | Path,
) -> Path:
    """写出 workflow research report。"""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Research-grade synthetic road-void workflow report",
        "",
        "## 1. 当前模型参数",
        f"- road_width = {config.geometry.road_width} m",
        f"- road_length = {config.geometry.road_length} m",
        f"- velocity_mode = {config.velocity.velocity_model_type}",
        f"- VR = {config.velocity.rayleigh_velocity:.2f} m/s",
        f"- VR_eff = {config.effective_rayleigh_velocity():.2f} m/s",
        "",
        "## 2. 地下层状结构",
    ]
    for layer in scenario.layers:
        lines.append(f"- {layer.label}: thickness={layer.thickness} m, Vp={layer.vp} m/s, Vs={layer.vs} m/s, rho={layer.rho} kg/m3")
    lines.extend(["", "## 3. 异常体真值"])
    for item in scenario.anomalies:
        lines.append(f"- {item.label}: shape={item.shape}, x={item.x}, y={item.y}, depth={item.depth}, strength={item.scattering_strength}")
    lines.extend(
        [
            "",
            "## 4. 合成数据摘要",
            f"- data shape = {dataset.data.shape} = shot x time x channel",
            f"- DAS-like data shape = {dataset.das_like_data.shape}",
            f"- gauge_length = {dataset.metadata['das_like_response']['gauge_length']} m",
            "",
            "## 5. 绕射识别结果",
            f"- attribute shot_index = {diffraction.shot_index}",
            f"- candidate count = {len(diffraction.candidates)}",
        ]
    )
    for cand in diffraction.candidates[:5]:
        lines.append(f"  - x={cand['candidate_x']:.2f}, y={cand['candidate_y']:.2f}, h={cand['candidate_depth']:.2f}, score={cand['score']:.4g}")
    lines.extend(
        [
            "",
            "## 6. 定位/反演结果",
            f"- best_estimate = {evaluation.best_estimate}",
            f"- true_position = {evaluation.true_position}",
            f"- total_location_error = {evaluation.total_location_error}",
            f"- confidence_score = {evaluation.confidence_score:.4f}",
            f"- uncertainty_bounds = {evaluation.uncertainty_bounds}",
            "",
            "## 7. 当前限制",
            "- 当前主线仍是三维运动学/属性正演与 joint scan，不是完整弹性 FWI。",
            "- DAS-like response 是沿光纤方向差分/平滑近似，不是真实仪器响应。",
            "- 单侧 DAS + 对侧锤击下 y-h 耦合仍然明显，深度应解释为范围。",
            "",
            "## 8. 下一步建议",
            "- 用 elastic-validate 做局部全波场 sanity check。",
            "- 增加 FK/扇形滤波或预测滤波做绕射增强对比。",
            "- 在可靠 forward solver 和残差定义基础上，再推进 FWI refinement。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _finish(output: str | Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
