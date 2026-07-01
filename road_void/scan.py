"""三维源-空洞-接收点绕射扫描。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import RoadGeometry
from .processing import envelope, sample_along_times


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CavityScanGrid:
    """三维绕射定位使用的候选网格。"""

    x: FloatArray
    y: FloatArray
    h: FloatArray
    velocity: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=float))
        object.__setattr__(self, "y", np.asarray(self.y, dtype=float))
        object.__setattr__(self, "h", np.asarray(self.h, dtype=float))
        object.__setattr__(self, "velocity", np.asarray(self.velocity, dtype=float))
        if min(self.x.size, self.y.size, self.h.size, self.velocity.size) == 0:
            raise ValueError("All scan-grid axes must be non-empty.")
        if np.any(self.h < 0) or np.any(self.velocity <= 0):
            raise ValueError("Depths must be non-negative and velocities positive.")


@dataclass(frozen=True)
class Candidate:
    """一个已排序扫描候选。"""

    x0: float
    y0: float
    h: float
    velocity: float
    score: float
    residual_rms: float


@dataclass(frozen=True)
class CavityScanResult:
    """绕射扫描输出结果。"""

    grid: CavityScanGrid
    scores: FloatArray
    best: Candidate
    top_candidates: list[Candidate]
    confidence: float
    uncertainty: dict[str, tuple[float, float]]


def scan_cavity_diffraction(
    data: FloatArray,
    geometry: RoadGeometry,
    grid: CavityScanGrid,
    t0: float = 0.0,
    use_envelope: bool = True,
    half_window: float = 0.014,
    top_k: int = 10,
    confidence_fraction: float = 0.92,
) -> CavityScanResult:
    """网格搜索三维瑞雷波绕射/散射中心。

    评分会沿 ``S_j -> (x0, y0, h) -> G_i`` 走时曲面采样残差记录能量。
    由于单侧 DAS 孔径受限，返回的不确定性应解释为疑似异常范围，尤其
    是 ``y0`` 和 ``h`` 方向。
    """

    attribute = envelope(data) if use_envelope else data
    scores = np.zeros((grid.x.size, grid.y.size, grid.h.size, grid.velocity.size), dtype=float)
    candidates: list[Candidate] = []

    # 对每个候选异常体位置，计算完整的 S-D-G 三维绕射走时面，
    # 然后沿该曲面采样残差能量。高分表示数据中存在与该候选几何一致的事件。
    for ix, x0 in enumerate(grid.x):
        for iy, y0 in enumerate(grid.y):
            for ih, h in enumerate(grid.h):
                point = (float(x0), float(y0), float(h))
                for iv, velocity in enumerate(grid.velocity):
                    times = geometry.diffraction_times(point, float(velocity), t0=t0)
                    energy, coherence = sample_along_times(
                        attribute,
                        geometry,
                        times,
                        half_window=half_window,
                    )
                    score = energy * (0.75 + 0.25 * coherence)
                    scores[ix, iy, ih, iv] = score
                    candidates.append(
                        Candidate(
                            x0=float(x0),
                            y0=float(y0),
                            h=float(h),
                            velocity=float(velocity),
                            score=float(score),
                            residual_rms=float("nan"),
                        )
                    )

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    # 残差计算相对较慢，只对 top-k 候选执行，避免扫描大网格时不必要的开销。
    top = [_with_residual(c, attribute, geometry, t0, half_window) for c in ranked[: max(1, top_k)]]
    best = top[0]
    second = ranked[1].score if len(ranked) > 1 else 0.0
    confidence = float((best.score - second) / (best.score + 1e-12))
    uncertainty = _uncertainty_from_scores(grid, scores, confidence_fraction)
    return CavityScanResult(
        grid=grid,
        scores=scores,
        best=best,
        top_candidates=top,
        confidence=confidence,
        uncertainty=uncertainty,
    )


def _with_residual(
    candidate: Candidate,
    attribute: FloatArray,
    geometry: RoadGeometry,
    t0: float,
    half_window: float,
) -> Candidate:
    times = geometry.diffraction_times(
        (candidate.x0, candidate.y0, candidate.h),
        candidate.velocity,
        t0=t0,
    )
    residual = _arrival_residual(attribute, geometry, times, half_window)
    return Candidate(
        x0=candidate.x0,
        y0=candidate.y0,
        h=candidate.h,
        velocity=candidate.velocity,
        score=candidate.score,
        residual_rms=float(residual),
    )


def _arrival_residual(
    attribute: FloatArray,
    geometry: RoadGeometry,
    times: FloatArray,
    half_window: float,
) -> float:
    picked: list[float] = []
    predicted: list[float] = []
    time = geometry.time_axis
    for ishot in range(geometry.n_shots):
        for ich in range(geometry.n_channels):
            mask = np.abs(time - times[ishot, ich]) <= half_window
            if not np.any(mask):
                continue
            local = attribute[ishot, mask, ich]
            picked.append(float(time[mask][np.argmax(local)]))
            predicted.append(float(times[ishot, ich]))
    if not picked:
        return float("inf")
    return float(np.sqrt(np.mean((np.asarray(picked) - np.asarray(predicted)) ** 2)))


def _uncertainty_from_scores(
    grid: CavityScanGrid,
    scores: FloatArray,
    fraction: float,
) -> dict[str, tuple[float, float]]:
    max_score = float(np.max(scores))
    mask = scores >= fraction * max_score
    if not np.any(mask):
        best_idx = np.unravel_index(int(np.argmax(scores)), scores.shape)
        mask[best_idx] = True
    # 不确定性范围来自接近最高分的候选集合。它比单点最优更适合受限孔径解释。
    ix, iy, ih, iv = np.where(mask)
    return {
        "x0": (float(np.min(grid.x[ix])), float(np.max(grid.x[ix]))),
        "y0": (float(np.min(grid.y[iy])), float(np.max(grid.y[iy]))),
        "h": (float(np.min(grid.h[ih])), float(np.max(grid.h[ih]))),
        "velocity": (float(np.min(grid.velocity[iv])), float(np.max(grid.velocity[iv]))),
    }
