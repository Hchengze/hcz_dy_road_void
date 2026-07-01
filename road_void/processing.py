"""直达波拟合与绕射增强使用的处理工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import RoadGeometry
from .wavelets import hammer_pulse, ricker


FloatArray = NDArray[np.float64]


def bandpass(data: FloatArray, dt: float, fmin: float, fmax: float, order: int = 4) -> FloatArray:
    """沿时间轴应用 Butterworth 带通滤波。"""

    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception as exc:  # pragma: no cover - 仅在缺少 scipy 时触发
        raise ImportError("bandpass requires scipy.signal.") from exc
    nyq = 0.5 / dt
    if not 0 < fmin < fmax < nyq:
        raise ValueError("Require 0 < fmin < fmax < Nyquist.")
    sos = butter(order, [fmin / nyq, fmax / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=1)


def trace_normalize(data: FloatArray, percentile: float = 95.0, eps: float = 1e-9) -> FloatArray:
    """按稳健振幅分位数归一化每个炮-道记录。"""

    scale = np.percentile(np.abs(data), percentile, axis=1, keepdims=True)
    return data / np.maximum(scale, eps)


def envelope(data: FloatArray) -> FloatArray:
    """返回沿时间轴计算的解析信号包络。"""

    try:
        from scipy.signal import hilbert
    except Exception as exc:  # pragma: no cover
        raise ImportError("envelope requires scipy.signal.") from exc
    return np.abs(hilbert(data, axis=1))


@dataclass(frozen=True)
class VelocityFit:
    """直达波速度拟合结果。"""

    velocity: float
    t0: float
    residual_rms: float
    n_picks: int


def pick_direct_arrivals(
    data: FloatArray,
    geometry: RoadGeometry,
    velocity_hint: float,
    t0_hint: float = 0.0,
    search_half_width: float = 0.045,
) -> FloatArray:
    """在三维几何预测附近用包络极大值拾取直达波到时。"""

    if velocity_hint <= 0:
        raise ValueError("velocity_hint must be positive.")
    env = envelope(data)
    pred = geometry.direct_times(velocity_hint, t0_hint)
    picks = np.full(pred.shape, np.nan, dtype=float)
    time = geometry.time_axis
    for ishot in range(geometry.n_shots):
        for ich in range(geometry.n_channels):
            mask = np.abs(time - pred[ishot, ich]) <= search_half_width
            if not np.any(mask):
                continue
            local_idx = np.argmax(env[ishot, mask, ich])
            picks[ishot, ich] = time[mask][local_idx]
    return picks


def fit_direct_velocity(arrival_times: FloatArray, geometry: RoadGeometry) -> VelocityFit:
    """由直达波拾取拟合 ``t = t0 + distance_3d / VR``。"""

    distances = geometry.source_receiver_distances()
    valid = np.isfinite(arrival_times)
    if np.count_nonzero(valid) < 3:
        raise ValueError("At least three valid arrival picks are required.")
    # 线性化形式为 t = t0 + distance * slowness。
    # 注意 distance 是三维源检距，因此拟合结果才对应道路横向孔径下的等效 VR。
    x = distances[valid]
    y = arrival_times[valid]
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    t0 = float(coef[0])
    slowness = float(coef[1])
    if slowness <= 0:
        raise ValueError("Fitted slowness is non-positive.")
    pred = design @ coef
    residual_rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    return VelocityFit(velocity=1.0 / slowness, t0=t0, residual_rms=residual_rms, n_picks=int(y.size))


def mute_direct_wave(
    data: FloatArray,
    geometry: RoadGeometry,
    velocity: float,
    t0: float = 0.0,
    half_width: float = 0.045,
    taper: float = 0.015,
) -> FloatArray:
    """用平滑时间 mute 压制直达瑞雷波。"""

    pred = geometry.direct_times(velocity, t0)
    time = geometry.time_axis
    muted = data.copy()
    for ishot in range(geometry.n_shots):
        dist = np.abs(time[:, None] - pred[ishot][None, :])
        weight = np.ones_like(muted[ishot])
        inner = dist <= half_width
        ramp = (dist > half_width) & (dist < half_width + taper)
        weight[inner] = 0.0
        weight[ramp] = 0.5 - 0.5 * np.cos(np.pi * (dist[ramp] - half_width) / taper)
        muted[ishot] *= weight
    return muted


def subtract_direct_wave_template(
    data: FloatArray,
    geometry: RoadGeometry,
    velocity: float,
    t0: float = 0.0,
    frequency: float = 35.0,
    wavelet: str = "ricker",
    fit_half_width: float = 0.035,
) -> FloatArray:
    """逐道减去最佳拟合直达波模板。

    这是面向合成数据和触发较好的主动源数据的一阶模型减法。当浅部散射
    事件只比直达波晚几毫秒时，它通常比宽时窗 mute 更不容易误伤绕射波。
    """

    if wavelet == "ricker":
        wavelet_t, wavelet_amp = ricker(frequency, geometry.dt)
    elif wavelet == "hammer":
        wavelet_t, wavelet_amp = hammer_pulse(frequency, geometry.dt)
    else:
        raise ValueError("wavelet must be either 'ricker' or 'hammer'.")
    direct = geometry.direct_times(velocity, t0)
    residual = data.copy()
    nt = geometry.n_times
    time = geometry.time_axis
    for ishot in range(geometry.n_shots):
        for ich in range(geometry.n_channels):
            # 逐道生成理论直达波模板，再在直达波邻域内拟合振幅。
            # 这样比宽 mute 更温和，适合散射波紧贴直达波的浅层异常场景。
            template = np.zeros(nt, dtype=float)
            samples = (direct[ishot, ich] + wavelet_t) / geometry.dt
            lower = np.floor(samples).astype(int)
            frac = samples - lower
            valid = (lower >= 0) & (lower + 1 < nt)
            if not np.any(valid):
                continue
            lo = lower[valid]
            template[lo] += wavelet_amp[valid] * (1.0 - frac[valid])
            template[lo + 1] += wavelet_amp[valid] * frac[valid]
            fit_mask = np.abs(time - direct[ishot, ich]) <= fit_half_width
            denom = float(np.dot(template[fit_mask], template[fit_mask]))
            if denom <= 1e-12:
                continue
            amp = float(np.dot(data[ishot, fit_mask, ich], template[fit_mask]) / denom)
            residual[ishot, :, ich] -= amp * template
    return residual


def sample_along_times(
    attribute: FloatArray,
    geometry: RoadGeometry,
    arrival_times: FloatArray,
    half_window: float = 0.012,
) -> tuple[float, float]:
    """在候选走时面附近采样能量和有符号相干性。"""

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
    # 在理论走时附近取局部最大值，允许合成波形存在少量相位偏移或采样误差。
    max_idx = np.argmax(np.abs(sampled), axis=2)
    gathered = np.take_along_axis(sampled, max_idx[:, :, None], axis=2).squeeze(axis=2)
    valid_trace = np.any(valid, axis=2)
    if not np.any(valid_trace):
        return 0.0, 0.0
    sig = gathered[valid_trace]
    amp = np.abs(sig)
    energy = float(np.mean(amp))
    coherence = float(abs(np.mean(sig)) / (np.mean(np.abs(sig)) + 1e-12))
    return energy, coherence
