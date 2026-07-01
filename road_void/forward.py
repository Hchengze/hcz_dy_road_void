"""道路空洞实验的三维等效瑞雷面波正演。

这是运动学和属性层面的原型，不是三维弹性 FDTD 求解器。它合成直达
瑞雷波、源-空洞-接收点散射波、几何扩散、近似衰减、城市交通类噪声
以及 DAS 通道质量变化，用于在真实数据或更重的波动方程正演之前测试
单侧 DAS + 对侧锤击的道路几何可行性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .anomaly import Cavity
from .geometry import RoadGeometry
from .wavelets import hammer_pulse, ricker


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ForwardModelConfig:
    """合成等效瑞雷面波数据的控制参数。"""

    rayleigh_velocity: float = 240.0
    t0: float = 0.02
    source_frequency: float = 35.0
    wavelet: str = "ricker"
    direct_amplitude: float = 1.0
    coda_amplitude: float = 0.12
    attenuation_q: float = 0.004
    noise_std: float = 0.03
    traffic_noise_std: float = 0.015
    weak_coupling_fraction: float = 0.06
    bad_channel_fraction: float = 0.02
    coupling_variation: float = 0.08
    random_seed: int | None = 2027


@dataclass
class SyntheticDataset:
    """合成 shot gather 及其元数据。

    数据形状为 ``n_shots x n_times x n_channels``。
    """

    data: FloatArray
    geometry: RoadGeometry
    config: ForwardModelConfig
    cavities: list[Cavity] = field(default_factory=list)
    direct_times: FloatArray | None = None
    diffraction_times: list[FloatArray] = field(default_factory=list)
    channel_gain: FloatArray | None = None


class RayleighKinematicForwardModel:
    """生成三维运动学瑞雷面波 DAS shot gather。"""

    def __init__(self, geometry: RoadGeometry, config: ForwardModelConfig | None = None) -> None:
        self.geometry = geometry
        self.config = config or ForwardModelConfig()

    def simulate(self, cavities: Sequence[Cavity] | None = None) -> SyntheticDataset:
        """模拟无空洞、单空洞或多空洞场景的 shot gather。"""

        cavities = list(cavities or [])
        geom = self.geometry
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        data = np.zeros((geom.n_shots, geom.n_times, geom.n_channels), dtype=float)

        # 1. 先加入直达瑞雷波：它是后续速度拟合和直达波压制的基准事件。
        wavelet_t, wavelet_amp = self._make_wavelet()
        direct_times = geom.direct_times(cfg.rayleigh_velocity, cfg.t0)
        sr_dist = geom.source_receiver_distances()
        direct_amp = cfg.direct_amplitude / np.sqrt(np.maximum(sr_dist, 1.0))
        direct_amp *= np.exp(-cfg.attenuation_q * sr_dist)
        self._add_events(data, direct_times, direct_amp, wavelet_t, wavelet_amp)

        if cfg.coda_amplitude > 0:
            self._add_coda(data, direct_times, direct_amp, rng)

        # 2. 再加入空洞/异常体散射波。每个异常体都用一个有效散射中心表示，
        #    这不是完整空腔散射场，而是用于验证三维绕射定位的工程近似。
        diffraction_time_list: list[FloatArray] = []
        for cavity in cavities:
            td = geom.diffraction_times(cavity.xyz, cfg.rayleigh_velocity, cfg.t0)
            diffraction_time_list.append(td)
            sd, dg = self._scatterer_distances(cavity)
            path = sd + dg
            lateral_offset = np.abs(geom.shot_x[:, None] - cavity.x0) + np.abs(geom.channel_x[None, :] - cavity.x0)
            amp = cavity.scattering_strength / np.sqrt(np.maximum(path, 1.0))
            amp *= np.exp(-cfg.attenuation_q * path)
            amp *= np.exp(-0.5 * (lateral_offset / max(2.0 * cavity.radius + geom.road_width, 1.0)) ** 2)
            shifted_t = wavelet_t + 0.25 / cfg.source_frequency
            scattered = cavity.tail_strength * self._minimum_phase_tail(wavelet_amp)
            self._add_events(data, td, amp, shifted_t, scattered)
            self._apply_shadow(data, cavity, direct_times)

        # 3. 最后加入通道增益差异和噪声，使合成数据更接近真实 DAS 采集状态。
        channel_gain = self._make_channel_gains(rng)
        data *= channel_gain[None, None, :]
        self._add_noise(data, rng)
        return SyntheticDataset(
            data=data,
            geometry=geom,
            config=cfg,
            cavities=cavities,
            direct_times=direct_times,
            diffraction_times=diffraction_time_list,
            channel_gain=channel_gain,
        )

    def _make_wavelet(self) -> tuple[FloatArray, FloatArray]:
        cfg = self.config
        if cfg.wavelet == "ricker":
            return ricker(cfg.source_frequency, self.geometry.dt)
        if cfg.wavelet == "hammer":
            return hammer_pulse(cfg.source_frequency, self.geometry.dt)
        raise ValueError("wavelet must be either 'ricker' or 'hammer'.")

    def _add_events(
        self,
        data: FloatArray,
        arrival_times: FloatArray,
        amplitudes: FloatArray,
        wavelet_t: FloatArray,
        wavelet_amp: FloatArray,
    ) -> None:
        """用线性插值在非整数采样到时处加入子波事件。"""

        dt = self.geometry.dt
        nt = self.geometry.n_times
        for ishot in range(arrival_times.shape[0]):
            for ich in range(arrival_times.shape[1]):
                samples = (arrival_times[ishot, ich] + wavelet_t) / dt
                lower = np.floor(samples).astype(int)
                frac = samples - lower
                valid = (lower >= 0) & (lower + 1 < nt)
                if not np.any(valid):
                    continue
                vals = amplitudes[ishot, ich] * wavelet_amp[valid]
                lo = lower[valid]
                np.add.at(data[ishot, :, ich], lo, vals * (1.0 - frac[valid]))
                np.add.at(data[ishot, :, ich], lo + 1, vals * frac[valid])

    def _minimum_phase_tail(self, wavelet_amp: FloatArray) -> FloatArray:
        tail = wavelet_amp.copy()
        n = tail.size
        taper = np.linspace(1.0, 0.45, n)
        tail *= taper
        return tail

    def _add_coda(
        self,
        data: FloatArray,
        direct_times: FloatArray,
        direct_amp: FloatArray,
        rng: np.random.Generator,
    ) -> None:
        wavelet_t, wavelet_amp = ricker(max(10.0, self.config.source_frequency * 0.55), self.geometry.dt)
        for delay, scale in [(0.045, 0.7), (0.085, 0.45), (0.14, 0.25)]:
            jitter = rng.normal(0.0, 0.006, size=direct_times.shape)
            self._add_events(
                data,
                direct_times + delay + jitter,
                self.config.coda_amplitude * scale * direct_amp,
                wavelet_t,
                wavelet_amp,
            )

    def _scatterer_distances(self, cavity: Cavity) -> tuple[FloatArray, FloatArray]:
        point = np.asarray(cavity.xyz, dtype=float)
        sd = np.linalg.norm(self.geometry.shot_xyz - point[None, :], axis=-1)[:, None]
        dg = np.linalg.norm(self.geometry.channel_xyz - point[None, :], axis=-1)[None, :]
        return sd, dg

    def _apply_shadow(self, data: FloatArray, cavity: Cavity, direct_times: FloatArray) -> None:
        if cavity.attenuation_strength <= 0:
            return
        geom = self.geometry
        # 阴影效应用来表达空洞附近直达波能量可能降低的现象。
        # 它只是一种属性级近似，不代表严格的透射/反射系数计算。
        x_weight = np.exp(-0.5 * ((geom.channel_x[None, :] - cavity.x0) / max(2.0 * cavity.radius, 1.0)) ** 2)
        shot_weight = np.exp(-0.5 * ((geom.shot_x[:, None] - cavity.x0) / max(3.0 * cavity.radius, 1.0)) ** 2)
        shadow = cavity.attenuation_strength * x_weight * shot_weight
        half_width = max(0.025, 1.5 / self.config.source_frequency)
        time = geom.time_axis
        for ishot in range(geom.n_shots):
            mask = np.abs(time[:, None] - direct_times[ishot][None, :]) <= half_width
            data[ishot] *= 1.0 - shadow[ishot][None, :] * mask

    def _make_channel_gains(self, rng: np.random.Generator) -> FloatArray:
        geom = self.geometry
        cfg = self.config
        gain = rng.lognormal(mean=0.0, sigma=cfg.coupling_variation, size=geom.n_channels)
        n_weak = int(round(cfg.weak_coupling_fraction * geom.n_channels))
        n_bad = int(round(cfg.bad_channel_fraction * geom.n_channels))
        if n_weak > 0:
            weak_idx = rng.choice(geom.n_channels, size=n_weak, replace=False)
            gain[weak_idx] *= rng.uniform(0.25, 0.65, size=n_weak)
        if n_bad > 0:
            bad_idx = rng.choice(geom.n_channels, size=n_bad, replace=False)
            gain[bad_idx] *= rng.uniform(0.0, 0.08, size=n_bad)
        return gain.astype(float)

    def _add_noise(self, data: FloatArray, rng: np.random.Generator) -> None:
        cfg = self.config
        if cfg.noise_std > 0:
            data += rng.normal(0.0, cfg.noise_std, size=data.shape)
        if cfg.traffic_noise_std > 0:
            nt = self.geometry.n_times
            t = self.geometry.time_axis
            for ishot in range(self.geometry.n_shots):
                phase = rng.uniform(0, 2 * np.pi)
                traffic = np.sin(2 * np.pi * rng.uniform(4.0, 10.0) * t + phase)
                traffic += 0.5 * np.sin(2 * np.pi * rng.uniform(12.0, 18.0) * t + 0.3 * phase)
                spatial = rng.normal(1.0, 0.15, size=self.geometry.n_channels)
                data[ishot] += cfg.traffic_noise_std * traffic[:nt, None] * spatial[None, :]
