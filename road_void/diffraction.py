"""绕射/散射特征识别属性。

这里不引入机器学习，也不假装能自动“确诊空洞”。目标是把已有 residual、
包络能量、局部相干和 S-D-G 理论曲线扫描结果整理成可解释的候选异常体
证据，用于科研级合成数据闭环评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .config import RoadVoidConfig
from .processing import envelope
from .scan import Candidate
from .workflow import WorkflowResult


FloatArray = NDArray[np.float64]


@dataclass
class DiffractionDetectionResult:
    """绕射/散射识别结果。"""

    shot_index: int
    attribute_gather: FloatArray
    coherence_gather: FloatArray
    candidates: list[dict[str, Any]]
    metadata: dict[str, Any]


def detect_diffraction_features(config: RoadVoidConfig, workflow: WorkflowResult) -> DiffractionDetectionResult:
    """从 residual 和扫描结果中提取轻量绕射/散射属性。

    属性包括：
    - envelope energy：直达波压制后残差的包络能量；
    - local coherence：沿通道方向局部相干性；
    - candidate score：已有三维 S-D-G 扫描的 top-k 候选。
    """

    geom = workflow.dataset.geometry
    cavities = config.to_cavities()
    shot_index = geom.n_shots // 2 if not cavities else min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    attr = envelope(workflow.residual)
    coherence = local_channel_coherence(workflow.residual)
    candidates = [_candidate_to_dict(c, workflow.scan_result.per_shot_score_contribution) for c in workflow.scan_result.top_candidates]
    return DiffractionDetectionResult(
        shot_index=shot_index,
        attribute_gather=attr[shot_index],
        coherence_gather=coherence[shot_index],
        candidates=candidates,
        metadata={
            "method": "direct-wave residual + envelope energy + local channel coherence + S-D-G candidate score",
            "score_method": config.processing.score_method,
            "scan_mode": config.processing.scan_mode,
            "confidence": workflow.scan_result.confidence,
            "note": "候选结果表示疑似散射/绕射异常，不等于空洞确诊。",
        },
    )


def local_channel_coherence(data: FloatArray, half_width: int = 2) -> FloatArray:
    """沿 DAS 通道方向计算简化局部相干属性。

    coherence = |sum traces| / sum |traces|，只用于突出横向连续事件；它不是
    完整 semblance 或 FK 滤波。
    """

    out = np.zeros_like(data, dtype=float)
    for ich in range(data.shape[2]):
        lo = max(0, ich - half_width)
        hi = min(data.shape[2], ich + half_width + 1)
        win = data[:, :, lo:hi]
        coherent = np.abs(np.sum(win, axis=2))
        incoherent = np.sum(np.abs(win), axis=2) + 1e-12
        out[:, :, ich] = coherent / incoherent
    return out


def plot_diffraction_attribute(
    result: DiffractionDetectionResult,
    geometry,
    output: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制残差包络能量属性图。"""

    extent = [float(geometry.channel_x[0]), float(geometry.channel_x[-1]), float(geometry.time_axis[-1]), float(geometry.time_axis[0])]
    plt.figure(figsize=(10, 5.4))
    plt.imshow(result.attribute_gather, aspect="auto", cmap="magma", extent=extent)
    plt.colorbar(label="residual envelope energy")
    plt.xlabel("DAS 通道 x (m)")
    plt.ylabel("时间 (s)")
    plt.title("04b 绕射/散射属性：直达波残差包络能量")
    _finish(output, save, show, dpi)


def plot_diffraction_candidates(
    result: DiffractionDetectionResult,
    output: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制 top-k 候选异常体评分。"""

    labels = [f"{i+1}\nx={c['candidate_x']:.1f}\ny={c['candidate_y']:.1f}\nh={c['candidate_depth']:.1f}" for i, c in enumerate(result.candidates[:8])]
    scores = [c["score"] for c in result.candidates[:8]]
    plt.figure(figsize=(9, 4.8))
    plt.bar(np.arange(len(scores)), scores, color="tab:orange")
    plt.xticks(np.arange(len(scores)), labels, fontsize=8)
    plt.ylabel("candidate score")
    plt.title("04c 绕射/散射候选体评分：来自多炮 S-D-G 曲线扫描")
    _finish(output, save, show, dpi)


def _candidate_to_dict(candidate: Candidate, per_shot: FloatArray | None) -> dict[str, Any]:
    supporting: list[int] = []
    if per_shot is not None and per_shot.size:
        supporting = [int(i) for i in np.argsort(per_shot)[-5:][::-1]]
    return {
        "candidate_x": candidate.x0,
        "candidate_y": candidate.y0,
        "candidate_depth": candidate.h,
        "velocity": candidate.velocity,
        "score": candidate.score,
        "residual": candidate.residual_rms,
        "supporting_shots": supporting,
    }


def _finish(output: str | Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

