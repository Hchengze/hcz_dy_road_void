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
    scan_mode: str = "joint"
    per_shot_best: list[Candidate] | None = None
    per_shot_score_contribution: FloatArray | None = None
    shot_weights: FloatArray | None = None
    consistency: dict[str, float] | None = None


def scan_cavity_diffraction(
    data: FloatArray,
    geometry: RoadGeometry,
    grid: CavityScanGrid,
    t0: float = 0.0,
    use_envelope: bool = True,
    half_window: float = 0.014,
    top_k: int = 10,
    confidence_fraction: float = 0.92,
    scan_mode: str = "joint",
    shot_index: int | None = None,
    shot_weight_mode: str = "uniform",
) -> CavityScanResult:
    """网格搜索三维瑞雷波绕射/散射中心。

    评分会沿 ``S_j -> (x0, y0, h) -> G_i`` 走时曲面采样残差记录能量。
    由于单侧 DAS 孔径受限，返回的不确定性应解释为疑似异常范围，尤其
    是 ``y0`` 和 ``h`` 方向。
    """

    if scan_mode not in {"joint", "single-shot", "compare"}:
        raise ValueError("scan_mode must be joint, single-shot or compare.")
    if shot_weight_mode not in {"uniform", "near-cavity", "snr"}:
        raise ValueError("shot_weight_mode must be uniform, near-cavity or snr.")
    attribute = envelope(data) if use_envelope else data
    scores = np.zeros((grid.x.size, grid.y.size, grid.h.size, grid.velocity.size), dtype=float)
    per_shot_grid = np.zeros(scores.shape + (geometry.n_shots,), dtype=float)
    candidates: list[Candidate] = []
    selected_shots = _selected_shots(geometry, scan_mode, shot_index)
    base_weights = _base_shot_weights(attribute, shot_weight_mode)

    # 对每个候选异常体位置，计算完整的 S-D-G 三维绕射走时面，
    # 然后沿该曲面采样残差能量。高分表示数据中存在与该候选几何一致的事件。
    for ix, x0 in enumerate(grid.x):
        for iy, y0 in enumerate(grid.y):
            for ih, h in enumerate(grid.h):
                point = (float(x0), float(y0), float(h))
                shot_weights = _candidate_shot_weights(geometry, float(x0), base_weights, shot_weight_mode)
                for iv, velocity in enumerate(grid.velocity):
                    times = geometry.diffraction_times(point, float(velocity), t0=t0)
                    per_shot = _sample_along_times_per_shot(attribute, geometry, times, half_window)
                    per_shot_grid[ix, iy, ih, iv, :] = per_shot
                    score = _combine_shot_scores(per_shot, shot_weights, selected_shots)
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
    per_shot_best = _per_shot_best_candidates(grid, per_shot_grid)
    contribution = per_shot_grid[np.unravel_index(int(np.argmax(scores)), scores.shape)]
    consistency = _consistency_from_per_shot(per_shot_best)
    return CavityScanResult(
        grid=grid,
        scores=scores,
        best=best,
        top_candidates=top,
        confidence=confidence,
        uncertainty=uncertainty,
        scan_mode=scan_mode,
        per_shot_best=per_shot_best,
        per_shot_score_contribution=contribution,
        shot_weights=_candidate_shot_weights(geometry, best.x0, base_weights, shot_weight_mode),
        consistency=consistency,
    )


def _selected_shots(geometry: RoadGeometry, scan_mode: str, shot_index: int | None) -> np.ndarray:
    if scan_mode == "single-shot":
        idx = geometry.n_shots // 2 if shot_index is None else int(shot_index)
        if idx < 0 or idx >= geometry.n_shots:
            raise ValueError("shot_index 超出炮点范围。")
        return np.asarray([idx], dtype=int)
    return np.arange(geometry.n_shots, dtype=int)


def _base_shot_weights(attribute: FloatArray, mode: str) -> FloatArray:
    if mode == "snr":
        energy = np.sqrt(np.mean(attribute**2, axis=(1, 2)))
        return energy / (np.mean(energy) + 1e-12)
    return np.ones(attribute.shape[0], dtype=float)


def _candidate_shot_weights(
    geometry: RoadGeometry,
    x0: float,
    base_weights: FloatArray,
    mode: str,
) -> FloatArray:
    weights = base_weights.astype(float).copy()
    if mode == "near-cavity":
        sigma = max(0.18 * (geometry.channel_x[-1] - geometry.channel_x[0]), geometry.road_width, 1.0)
        weights *= np.exp(-0.5 * ((geometry.shot_x - x0) / sigma) ** 2)
    return weights / (np.mean(weights) + 1e-12)


def _combine_shot_scores(per_shot: FloatArray, weights: FloatArray, selected_shots: np.ndarray) -> float:
    local_scores = per_shot[selected_shots]
    local_weights = weights[selected_shots]
    if local_scores.size == 0:
        return 0.0
    return float(np.sum(local_scores * local_weights) / (np.sum(local_weights) + 1e-12))


def _sample_along_times_per_shot(
    attribute: FloatArray,
    geometry: RoadGeometry,
    arrival_times: FloatArray,
    half_window: float,
) -> FloatArray:
    """逐炮沿候选绕射曲线采样能量和相干性。

    返回值是每炮一个分数。joint 模式对这些分数加权平均；single-shot 模式
    只取指定炮；compare 模式返回 joint 分数，同时保留每炮最佳结果用于对比。
    """

    dt = geometry.dt
    nt = geometry.n_times
    center = np.rint(arrival_times / dt).astype(int)
    half_samples = max(0, int(round(half_window / dt)))
    offsets = np.arange(-half_samples, half_samples + 1, dtype=int)
    sample_idx = center[:, :, None] + offsets[None, None, :]
    valid = (sample_idx >= 0) & (sample_idx < nt)
    clipped = np.clip(sample_idx, 0, nt - 1)
    shot_idx = np.arange(geometry.n_shots)[:, None, None]
    channel_idx = np.arange(geometry.n_channels)[None, :, None]
    sampled = attribute[shot_idx, clipped, channel_idx]
    sampled = np.where(valid, sampled, 0.0)
    max_idx = np.argmax(np.abs(sampled), axis=2)
    gathered = np.take_along_axis(sampled, max_idx[:, :, None], axis=2).squeeze(axis=2)
    valid_trace = np.any(valid, axis=2)
    amp = np.abs(gathered) * valid_trace
    count = np.maximum(np.sum(valid_trace, axis=1), 1)
    energy = np.sum(amp, axis=1) / count
    signed_sum = np.abs(np.sum(gathered * valid_trace, axis=1)) / count
    mean_abs = np.sum(amp, axis=1) / count
    coherence = signed_sum / (mean_abs + 1e-12)
    return energy * (0.75 + 0.25 * coherence)


def _per_shot_best_candidates(grid: CavityScanGrid, per_shot_grid: FloatArray) -> list[Candidate]:
    best: list[Candidate] = []
    for ishot in range(per_shot_grid.shape[-1]):
        idx = np.unravel_index(int(np.argmax(per_shot_grid[..., ishot])), per_shot_grid[..., ishot].shape)
        best.append(
            Candidate(
                x0=float(grid.x[idx[0]]),
                y0=float(grid.y[idx[1]]),
                h=float(grid.h[idx[2]]),
                velocity=float(grid.velocity[idx[3]]),
                score=float(per_shot_grid[idx + (ishot,)]),
                residual_rms=float("nan"),
            )
        )
    return best


def _consistency_from_per_shot(candidates: list[Candidate]) -> dict[str, float]:
    if not candidates:
        return {"x_std": float("nan"), "y_std": float("nan"), "h_std": float("nan")}
    return {
        "x_std": float(np.std([c.x0 for c in candidates])),
        "y_std": float(np.std([c.y0 for c in candidates])),
        "h_std": float(np.std([c.h for c in candidates])),
    }


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
