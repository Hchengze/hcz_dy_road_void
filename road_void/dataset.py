"""科研级合成 survey dataset 与 DAS-like 响应近似。

本模块把已有运动学正演结果整理成结构化数据集，并加入轻量 DAS-like
处理：沿光纤方向的 gauge length 平滑/差分、相干交通噪声、随机脉冲噪声
和通道耦合变化。它是合成数据研究接口，不是真实 DAS 解调器。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .config import RoadVoidConfig
from .forward import SyntheticDataset
from .geometry import RoadGeometry
from .workflow import WorkflowResult, simulate_from_config


FloatArray = NDArray[np.float64]


@dataclass
class DASLikeResult:
    """DAS-like 近似响应。

    strain_rate_like 由沿 x 方向的有限差分/平滑得到，只表示沿光纤方向
    应变率响应的合成近似；真实 DAS 还受 gauge length、光纤耦合、方向、
    解调方式和埋深影响。
    """

    data: FloatArray
    components: dict[str, FloatArray]
    gauge_length: float


@dataclass
class ResearchSurveyDataset:
    """科研级合成道路空洞 survey dataset。"""

    shots: FloatArray
    receivers: FloatArray
    time_axis: FloatArray
    data: FloatArray
    das_like_data: FloatArray
    labels: dict[str, Any]
    metadata: dict[str, Any]
    noise_components: dict[str, FloatArray]


def generate_synthetic_survey_dataset(
    config: RoadVoidConfig,
    workflow: WorkflowResult | None = None,
    *,
    gauge_length: float = 4.0,
) -> ResearchSurveyDataset:
    """生成结构化 synthetic survey dataset。

    若 workflow 已经运行，则复用其中的 dataset，避免重复正演；否则内部
    调用 simulate_from_config。返回对象既包含原始运动学数据，也包含
    DAS-like 近似响应和真值标签。
    """

    synthetic = workflow.dataset if workflow is not None else simulate_from_config(config)
    geom = synthetic.geometry
    das = apply_das_like_response(synthetic.data, geom, config, gauge_length=gauge_length)
    labels = _build_labels(config, synthetic)
    metadata = _build_metadata(config, synthetic, gauge_length)
    return ResearchSurveyDataset(
        shots=geom.shot_xyz.copy(),
        receivers=geom.channel_xyz.copy(),
        time_axis=geom.time_axis.copy(),
        data=synthetic.data.copy(),
        das_like_data=das.data,
        labels=labels,
        metadata=metadata,
        noise_components=das.components,
    )


def apply_das_like_response(
    data: FloatArray,
    geometry: RoadGeometry,
    config: RoadVoidConfig,
    *,
    gauge_length: float = 4.0,
) -> DASLikeResult:
    """把合成振幅转换为 DAS-like 沿光纤方向响应。

    当前做法：
    1. 沿 channel/x 方向做有限长度差分，近似 ``strain_rate_xx``；
    2. 用 gauge length 对通道方向做滑动平滑；
    3. 叠加相干交通噪声和少量随机脉冲噪声；
    4. 保留弱耦合/坏道造成的通道增益变化。

    这不是真实 DAS 仪器响应，只是用于合成数据鲁棒性测试的近似。
    """

    spacing = float(np.median(np.diff(geometry.channel_x))) if geometry.n_channels > 1 else 1.0
    half = max(1, int(round(0.5 * gauge_length / max(spacing, 1e-6))))
    padded = np.pad(data, ((0, 0), (0, 0), (half, half)), mode="edge")
    strain_rate = (padded[:, :, 2 * half :] - padded[:, :, : -2 * half]) / max(2 * half * spacing, 1e-6)
    smoothed = _moving_average_channels(strain_rate, max(1, 2 * half + 1))

    rng = np.random.default_rng(config.record.random_seed)
    traffic = _coherent_traffic_noise(geometry, smoothed.shape, config.noise.traffic_noise_level, rng)
    impulsive = _impulsive_noise(smoothed.shape, 0.15 * config.noise.noise_level, rng)
    coupling = _weak_coupling_segments(geometry.n_channels, config.noise.weak_coupling_fraction, rng)
    das_like = (smoothed + traffic + impulsive) * coupling[None, None, :]
    return DASLikeResult(
        data=das_like.astype(float),
        components={
            "strain_rate_like": smoothed.astype(float),
            "traffic_coherent_noise": traffic.astype(float),
            "random_impulsive_noise": impulsive.astype(float),
            "coupling_gain": coupling.astype(float),
        },
        gauge_length=gauge_length,
    )


def save_synthetic_survey_dataset(dataset: ResearchSurveyDataset, outdir: str | Path) -> tuple[Path, Path]:
    """保存 npz 数据和 JSON metadata。"""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "synthetic_dataset.npz"
    json_path = outdir / "synthetic_dataset_metadata.json"
    np.savez_compressed(
        npz_path,
        shots=dataset.shots,
        receivers=dataset.receivers,
        time_axis=dataset.time_axis,
        data=dataset.data,
        das_like_data=dataset.das_like_data,
        direct_wave_time_curves=np.asarray(dataset.labels["direct_wave_time_curves"]),
        expected_diffraction_time_curves=np.asarray(dataset.labels["expected_diffraction_time_curves"]),
    )
    json_path.write_text(json.dumps(_to_jsonable(dataset.metadata | {"labels": dataset.labels}), ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, json_path


def plot_das_like_gather(
    dataset: ResearchSurveyDataset,
    shot_index: int,
    output: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制 DAS-like gather。"""

    gather = dataset.das_like_data[shot_index]
    clip = np.percentile(np.abs(gather), 98.5)
    x = dataset.receivers[:, 0]
    extent = [float(x[0]), float(x[-1]), float(dataset.time_axis[-1]), float(dataset.time_axis[0])]
    plt.figure(figsize=(10, 5.4))
    plt.imshow(gather, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip, extent=extent)
    plt.colorbar(label="DAS-like strain-rate 振幅")
    plt.xlabel("DAS 通道 x (m)")
    plt.ylabel("时间 (s)")
    plt.title("03b DAS-like gather：沿光纤方向差分/平滑近似，不是真实仪器响应")
    _finish(output, save, show, dpi)


def plot_noise_components(
    dataset: ResearchSurveyDataset,
    output: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制 DAS-like 响应中几个噪声/耦合分量的 RMS 摘要。"""

    plt.figure(figsize=(8.5, 4.8))
    for name, comp in dataset.noise_components.items():
        if comp.ndim == 3:
            rms = np.sqrt(np.mean(comp**2, axis=(0, 1)))
            plt.plot(dataset.receivers[:, 0], rms, label=name)
        elif comp.ndim == 1:
            plt.plot(dataset.receivers[:, 0], comp, label=name)
    plt.xlabel("DAS 通道 x (m)")
    plt.ylabel("RMS / gain")
    plt.title("03c DAS-like 噪声与耦合分量摘要")
    plt.legend(fontsize=8)
    _finish(output, save, show, dpi)


def _build_labels(config: RoadVoidConfig, synthetic: SyntheticDataset) -> dict[str, Any]:
    cavities = synthetic.cavities
    return {
        "true_anomaly_centers": [(c.x0, c.y0, c.h) for c in cavities],
        "true_anomaly_shapes": [c.shape for c in cavities],
        "true_anomaly_depths": [c.h for c in cavities],
        "direct_wave_time_curves": synthetic.direct_times if synthetic.direct_times is not None else np.empty((0, 0)),
        "expected_diffraction_time_curves": synthetic.diffraction_times,
        "scan_grid_recommendation": {
            "x": [config.processing.scan_x_min, config.processing.scan_x_max, config.processing.scan_x_step],
            "y": [config.processing.scan_y_min, config.processing.scan_y_max, config.processing.scan_y_step],
            "h": [config.processing.scan_h_min, config.processing.scan_h_max, config.processing.scan_h_step],
        },
    }


def _build_metadata(config: RoadVoidConfig, synthetic: SyntheticDataset, gauge_length: float) -> dict[str, Any]:
    geom = synthetic.geometry
    return {
        "geometry": {
            "road_width": config.geometry.road_width,
            "road_length": config.geometry.road_length,
            "n_shots": geom.n_shots,
            "n_channels": geom.n_channels,
            "sampling_rate": config.record.sampling_rate,
            "duration": config.record.duration,
        },
        "velocity_mode": config.velocity.velocity_model_type,
        "rayleigh_velocity": config.velocity.rayleigh_velocity,
        "effective_rayleigh_velocity": config.effective_rayleigh_velocity(),
        "layer_model": {
            "layer_depths": config.velocity.layer_depths,
            "layer_velocities": config.velocity.layer_velocities,
            "sensitivity_depth_factor": config.velocity.sensitivity_depth_factor,
        },
        "anomalies": [
            {"shape": c.shape, "x": c.x0, "y": c.y0, "depth": c.h, "radius": c.radius, "strength": c.scattering_strength}
            for c in synthetic.cavities
        ],
        "noise_settings": {
            "noise_level": config.noise.noise_level,
            "traffic_noise_level": config.noise.traffic_noise_level,
            "weak_coupling_fraction": config.noise.weak_coupling_fraction,
            "bad_channel_fraction": config.noise.bad_channel_fraction,
        },
        "das_like_response": {
            "gauge_length": gauge_length,
            "approximation": "沿光纤 x 方向有限长度差分/平滑的 strain-rate-like 响应，不是真实 DAS 仪器响应。",
        },
    }


def _moving_average_channels(data: FloatArray, width: int) -> FloatArray:
    if width <= 1:
        return data
    kernel = np.ones(width, dtype=float) / width
    out = np.empty_like(data)
    pad = width // 2
    padded = np.pad(data, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    for i in range(data.shape[2]):
        out[:, :, i] = np.tensordot(padded[:, :, i : i + width], kernel, axes=([2], [0]))
    return out


def _coherent_traffic_noise(shape_geom: RoadGeometry, shape: tuple[int, int, int], level: float, rng: np.random.Generator) -> FloatArray:
    if level <= 0:
        return np.zeros(shape, dtype=float)
    nshot, nt, nch = shape
    t = shape_geom.time_axis
    x = shape_geom.channel_x
    noise = np.zeros(shape, dtype=float)
    for ishot in range(nshot):
        freq = rng.uniform(3.0, 9.0)
        phase = rng.uniform(0, 2 * np.pi)
        spatial = np.cos(2 * np.pi * (x - x.min()) / max(float(np.ptp(x)), 1.0) + phase)
        wave = np.sin(2 * np.pi * freq * t + phase)
        noise[ishot] = level * 0.35 * wave[:, None] * spatial[None, :]
    return noise


def _impulsive_noise(shape: tuple[int, int, int], level: float, rng: np.random.Generator) -> FloatArray:
    spikes = np.zeros(shape, dtype=float)
    if level <= 0:
        return spikes
    n = max(1, int(0.00025 * np.prod(shape)))
    idx = tuple(rng.integers(0, s, size=n) for s in shape)
    spikes[idx] = rng.normal(0.0, 8.0 * level, size=n)
    return spikes


def _weak_coupling_segments(n_channels: int, fraction: float, rng: np.random.Generator) -> FloatArray:
    gain = np.ones(n_channels, dtype=float)
    nseg = max(1, int(round(3 * fraction * n_channels / 10))) if fraction > 0 else 0
    for _ in range(nseg):
        start = int(rng.integers(0, max(n_channels, 1)))
        length = int(rng.integers(3, max(4, n_channels // 8)))
        gain[start : min(n_channels, start + length)] *= rng.uniform(0.25, 0.65)
    return gain


def _finish(output: str | Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > 2000:
            return {"shape": list(value.shape), "note": "large array saved in synthetic_dataset.npz"}
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value
