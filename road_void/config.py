"""项目统一参数配置。

本模块把“道路几何、空洞参数、速度频率、采样记录、噪声耦合、处理扫描”
集中管理起来。配置文件推荐使用 YAML，并允许中文注释；为了保持轻量，
代码会优先使用 PyYAML，若环境中没有 PyYAML，则使用内置的简化 YAML
解析器读取本项目配置模板。
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import ast
from pathlib import Path
from typing import Any

import numpy as np

from .anomaly import Cavity
from .forward import ForwardModelConfig
from .geometry import RoadGeometry
from .scan import CavityScanGrid
from .velocity import LayeredRayleighVelocityModel, VelocityLayer


class ConfigError(ValueError):
    """配置文件内容不完整或参数不合理时抛出的错误。"""


@dataclass(frozen=True)
class GeometryConfig:
    """道路与观测几何参数。

    road_width 是本项目最关键的几何参数之一，单位为 m。它控制锤击线与
    光纤线之间的横向偏移，也就是单侧 DAS + 对侧锤击的观测孔径。道路
    越宽，传播路径越长，普通锤击能量越弱，横向位置 y0 与深度 h 的
    耦合通常也越明显。
    """

    road_width: float = 15.0
    road_length: float = 80.0
    fiber_y: float = 0.0
    source_y: float | None = None
    channel_x_min: float = 0.0
    channel_x_max: float = 80.0
    channel_spacing: float = 1.0
    source_x_min: float = 0.0
    source_x_max: float = 80.0
    source_spacing: float = 4.0
    fiber_depth: float = 0.0
    source_depth: float = 0.0


@dataclass(frozen=True)
class CavityConfig:
    """空洞/异常体参数。

    当前空洞是“有效散射体”，不是完整真实边界。cavity_x、cavity_y、
    cavity_h 主要控制三维绕射走时；scattering_strength、attenuation_strength
    和 tail_strength 主要控制散射事件能量、直达波阴影和可见性。
    """

    enable_cavity: bool = True
    cavity_x: float = 42.0
    cavity_y: float = 8.5
    cavity_h: float = 2.2
    cavity_radius: float = 2.0
    scattering_strength: float = 1.0
    attenuation_strength: float = 0.25
    tail_strength: float = 1.0
    label: str = "疑似空洞"
    shape: str = "sphere"
    size_x: float | None = None
    size_y: float | None = None
    size_z: float | None = None
    azimuth: float = 0.0
    anomalies: str | None = None


@dataclass(frozen=True)
class VelocityConfig:
    """速度、频率与震源子波参数。

    rayleigh_velocity 是某一频带内的等效瑞雷波速度，单位 m/s。它可由
    直达波三维几何拟合、DAS-MASW 或交通噪声频散估计。速度偏大时，
    扫描可能倾向更深解释；速度偏小时，深度解释可能偏浅。
    """

    rayleigh_velocity: float = 240.0
    velocity_model_type: str = "uniform"
    layer_depths: tuple[float, ...] = (0.5, 2.0, 6.0)
    layer_velocities: tuple[float, ...] = (320.0, 260.0, 220.0)
    sensitivity_depth_factor: float = 0.5
    source_frequency: float = 35.0
    wavelet_type: str = "ricker"
    bandpass_freqmin: float = 10.0
    bandpass_freqmax: float = 90.0
    multi_band: tuple[tuple[float, float], ...] = ((20.0, 40.0), (40.0, 80.0), (80.0, 120.0))


@dataclass(frozen=True)
class RecordConfig:
    """采样与记录参数。

    sampling_rate 需高于锤击主频和带宽的两倍以上，否则高频成分会混叠。
    duration 必须足够长，避免直达波、绕射波或尾波被记录窗口截断。
    """

    sampling_rate: float = 1000.0
    duration: float = 1.0
    t0: float = 0.02
    stack_count: int = 1
    random_seed: int | None = 2027


@dataclass(frozen=True)
class NoiseConfig:
    """噪声、耦合与干扰参数。"""

    noise_level: float = 0.03
    traffic_noise_level: float = 0.015
    bad_channel_fraction: float = 0.02
    weak_coupling_fraction: float = 0.06
    coupling_variation: float = 0.08
    direct_wave_strength: float = 1.0
    diffraction_strength: float = 1.0


@dataclass(frozen=True)
class ProcessingConfig:
    """处理和扫描参数。

    扫描范围和步长会同时影响定位精度与计算量。scan_h_step 越小，深度
    搜索越细，但计算越慢；步长过大时，真实深度附近可能没有被采样到。
    """

    bandpass_freqmin: float = 10.0
    bandpass_freqmax: float = 90.0
    direct_wave_mute_width: float = 0.04
    direct_wave_subtraction_enable: bool = True
    scan_x_min: float = 32.0
    scan_x_max: float = 52.0
    scan_x_step: float = 1.0
    scan_y_min: float = 3.0
    scan_y_max: float = 14.0
    scan_y_step: float = 1.0
    scan_h_min: float = 0.8
    scan_h_max: float = 4.0
    scan_h_step: float = 0.4
    scan_vr_min: float = 220.0
    scan_vr_max: float = 260.0
    scan_vr_step: float = 10.0
    score_method: str = "envelope"
    top_k: int = 8
    uncertainty_threshold: float = 0.92
    scan_mode: str = "joint"
    shot_index: int | None = None
    shot_weight_mode: str = "uniform"


@dataclass(frozen=True)
class RoadVoidConfig:
    """道路空洞 DAS 原型完整配置。"""

    geometry: GeometryConfig = GeometryConfig()
    cavity: CavityConfig = CavityConfig()
    velocity: VelocityConfig = VelocityConfig()
    record: RecordConfig = RecordConfig()
    noise: NoiseConfig = NoiseConfig()
    processing: ProcessingConfig = ProcessingConfig()

    def to_geometry(self) -> RoadGeometry:
        """根据配置构建 RoadGeometry。"""

        g = self.geometry
        dt = 1.0 / self.record.sampling_rate
        channel_x = np.arange(g.channel_x_min, g.channel_x_max + 0.5 * g.channel_spacing, g.channel_spacing)
        source_x = np.arange(g.source_x_min, g.source_x_max + 0.5 * g.source_spacing, g.source_spacing)
        return RoadGeometry(
            road_width=g.road_width,
            channel_x=channel_x,
            shot_x=source_x,
            dt=dt,
            t_max=self.record.duration,
            fiber_y=g.fiber_y,
            shot_y=g.road_width if g.source_y is None else g.source_y,
            receiver_z=g.fiber_depth,
            source_z=g.source_depth,
        )

    def to_cavities(self) -> list[Cavity]:
        """根据配置构建空洞/异常体列表。"""

        c = self.cavity
        if not c.enable_cavity:
            return []
        if c.anomalies:
            return _parse_anomaly_specs(
                c.anomalies,
                attenuation_strength=c.attenuation_strength,
                tail_strength=c.tail_strength,
                diffraction_strength=self.noise.diffraction_strength,
            )
        return [
            Cavity(
                x0=c.cavity_x,
                y0=c.cavity_y,
                h=c.cavity_h,
                radius=c.cavity_radius,
                scattering_strength=c.scattering_strength * self.noise.diffraction_strength,
                attenuation_strength=c.attenuation_strength,
                tail_strength=c.tail_strength,
                label=c.label,
                shape=c.shape,
                size_x=c.size_x,
                size_y=c.size_y,
                size_z=c.size_z,
                azimuth=c.azimuth,
            )
        ]

    def effective_rayleigh_velocity(self, reference_velocity: float | None = None) -> float:
        """返回当前正演/扫描实际使用的等效速度。

        ``uniform`` 直接返回参考速度。``layered-effective`` 会根据主频、波长
        和层状模型计算 ``VR_eff``，使层状速度能够进入当前运动学走时。
        """

        ref = self.velocity.rayleigh_velocity if reference_velocity is None else float(reference_velocity)
        mode = self.velocity.velocity_model_type
        if mode == "uniform":
            return float(ref)
        if mode == "layered-effective":
            return self.to_velocity_model().effective_velocity(
                ref,
                self.velocity.source_frequency,
                self.velocity.sensitivity_depth_factor,
            )
        raise ConfigError("velocity_model_type 只能是 uniform 或 layered-effective。")

    def to_forward_config(self) -> ForwardModelConfig:
        """根据配置构建正演参数。"""

        return ForwardModelConfig(
            rayleigh_velocity=self.effective_rayleigh_velocity(),
            t0=self.record.t0,
            source_frequency=self.velocity.source_frequency,
            wavelet=self.velocity.wavelet_type,
            direct_amplitude=self.noise.direct_wave_strength,
            coda_amplitude=0.05 * self.cavity.tail_strength,
            attenuation_q=0.004,
            noise_std=self.noise.noise_level,
            traffic_noise_std=self.noise.traffic_noise_level,
            weak_coupling_fraction=self.noise.weak_coupling_fraction,
            bad_channel_fraction=self.noise.bad_channel_fraction,
            coupling_variation=self.noise.coupling_variation,
            random_seed=self.record.random_seed,
        )

    def to_scan_grid(self) -> CavityScanGrid:
        """根据配置构建三维绕射扫描网格。"""

        p = self.processing
        velocity_axis = _arange_inclusive(p.scan_vr_min, p.scan_vr_max, p.scan_vr_step)
        if self.velocity.velocity_model_type == "layered-effective":
            velocity_axis = np.asarray([self.effective_rayleigh_velocity(v) for v in velocity_axis], dtype=float)
        return CavityScanGrid(
            x=_arange_inclusive(p.scan_x_min, p.scan_x_max, p.scan_x_step),
            y=_arange_inclusive(p.scan_y_min, p.scan_y_max, p.scan_y_step),
            h=_arange_inclusive(p.scan_h_min, p.scan_h_max, p.scan_h_step),
            velocity=velocity_axis,
        )

    def to_velocity_model(self) -> LayeredRayleighVelocityModel:
        """构建用于展示的等效瑞雷波速度模型。

        注意：uniform 模式返回常速模型；layered-effective 模式会同时用于
        计算 VR_eff，但它仍是轻量等效走时模型，不是弹性波正演。
        """

        if self.velocity.velocity_model_type == "uniform":
            depth = max(self.velocity.layer_depths[-1], self.processing.scan_h_max, 6.0)
            return LayeredRayleighVelocityModel(
                layers=(VelocityLayer(0.0, float(depth), self.velocity.rayleigh_velocity, "均匀等效瑞雷速度"),)
            )
        if self.velocity.velocity_model_type != "layered-effective":
            raise ConfigError("velocity_model_type 只能是 uniform 或 layered-effective。")
        depths = tuple(float(v) for v in self.velocity.layer_depths)
        velocities = tuple(float(v) for v in self.velocity.layer_velocities)
        if len(depths) != len(velocities):
            raise ConfigError("layer_depths 和 layer_velocities 长度必须一致。")
        top = 0.0
        layers: list[VelocityLayer] = []
        for idx, (bottom, velocity) in enumerate(zip(depths, velocities)):
            if bottom <= top:
                raise ConfigError("layer_depths 必须严格递增。")
            layers.append(VelocityLayer(top, bottom, velocity, f"第 {idx + 1} 层"))
            top = bottom
        return LayeredRayleighVelocityModel(tuple(layers))


def load_config(path: str | Path | None = None) -> RoadVoidConfig:
    """读取配置文件；path 为 None 时返回内置默认配置。"""

    if path is None:
        cfg = RoadVoidConfig()
        validate_config(cfg)
        return cfg
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    data = _load_yaml_like(path)
    required = {"geometry", "cavity", "velocity", "record", "noise", "processing"}
    missing = required - set(data)
    if missing:
        raise ConfigError(f"配置文件缺少关键章节: {', '.join(sorted(missing))}")
    cfg = RoadVoidConfig(
        geometry=_build_dataclass(GeometryConfig, data["geometry"]),
        cavity=_build_dataclass(CavityConfig, data["cavity"]),
        velocity=_build_dataclass(VelocityConfig, data["velocity"]),
        record=_build_dataclass(RecordConfig, data["record"]),
        noise=_build_dataclass(NoiseConfig, data["noise"]),
        processing=_build_dataclass(ProcessingConfig, data["processing"]),
    )
    validate_config(cfg)
    return cfg


def apply_overrides(
    config: RoadVoidConfig,
    road_width: float | None = None,
    cavity_depth: float | None = None,
    noise_level: float | None = None,
    rayleigh_velocity: float | None = None,
) -> RoadVoidConfig:
    """应用常用命令行参数覆盖。"""

    cfg = config
    if road_width is not None:
        cfg = replace(cfg, geometry=replace(cfg.geometry, road_width=road_width, source_y=road_width))
    if cavity_depth is not None:
        cfg = replace(cfg, cavity=replace(cfg.cavity, cavity_h=cavity_depth))
    if noise_level is not None:
        cfg = replace(cfg, noise=replace(cfg.noise, noise_level=noise_level))
    if rayleigh_velocity is not None:
        cfg = replace(cfg, velocity=replace(cfg.velocity, rayleigh_velocity=rayleigh_velocity))
    validate_config(cfg)
    return cfg


def validate_config(config: RoadVoidConfig) -> None:
    """检查关键参数，尽早给出清晰错误提示。"""

    g, c, v, r, n, p = config.geometry, config.cavity, config.velocity, config.record, config.noise, config.processing
    if g.road_width <= 0 or g.road_length <= 0:
        raise ConfigError("road_width 和 road_length 必须为正数。")
    if g.channel_spacing <= 0 or g.source_spacing <= 0:
        raise ConfigError("channel_spacing 和 source_spacing 必须为正数。")
    if g.channel_x_max <= g.channel_x_min or g.source_x_max <= g.source_x_min:
        raise ConfigError("通道和震源 x 范围必须满足 max > min。")
    if c.cavity_h < 0 or c.cavity_radius <= 0:
        raise ConfigError("cavity_h 必须非负，cavity_radius 必须为正数。")
    if v.velocity_model_type not in {"uniform", "layered-effective"}:
        raise ConfigError("velocity_model_type 只能是 uniform 或 layered-effective。")
    if len(v.layer_depths) != len(v.layer_velocities):
        raise ConfigError("layer_depths 和 layer_velocities 长度必须一致。")
    if any(depth <= 0 for depth in v.layer_depths):
        raise ConfigError("layer_depths 必须为正数。")
    if any(v2 <= 0 for v2 in v.layer_velocities):
        raise ConfigError("layer_velocities 必须全部为正数。")
    if any(b <= a for a, b in zip(v.layer_depths, v.layer_depths[1:])):
        raise ConfigError("layer_depths 必须严格递增。")
    if v.sensitivity_depth_factor <= 0:
        raise ConfigError("sensitivity_depth_factor 必须为正数。")
    if v.rayleigh_velocity <= 0 or v.source_frequency <= 0:
        raise ConfigError("rayleigh_velocity 和 source_frequency 必须为正数。")
    if r.sampling_rate <= 2.5 * v.source_frequency:
        raise ConfigError("sampling_rate 过低，可能导致锤击主频混叠。")
    if r.duration <= 0:
        raise ConfigError("duration 必须为正数。")
    if not 0 <= n.bad_channel_fraction < 1 or not 0 <= n.weak_coupling_fraction < 1:
        raise ConfigError("坏道比例和弱耦合比例必须位于 [0, 1) 区间。")
    if n.noise_level < 0 or n.traffic_noise_level < 0:
        raise ConfigError("噪声强度不能为负。")
    for name in ("scan_x_step", "scan_y_step", "scan_h_step", "scan_vr_step"):
        if getattr(p, name) <= 0:
            raise ConfigError(f"{name} 必须为正数。")
    if p.scan_x_max <= p.scan_x_min or p.scan_y_max <= p.scan_y_min or p.scan_h_max <= p.scan_h_min:
        raise ConfigError("扫描范围必须满足 max > min。")
    if p.scan_mode not in {"joint", "single-shot", "compare"}:
        raise ConfigError("scan_mode 必须是 joint、single-shot 或 compare。")
    if p.shot_weight_mode not in {"uniform", "near-cavity", "snr"}:
        raise ConfigError("shot_weight_mode 必须是 uniform、near-cavity 或 snr。")


def _parse_anomaly_specs(
    text: str,
    attenuation_strength: float,
    tail_strength: float,
    diffraction_strength: float,
) -> list[Cavity]:
    """解析命令行多异常体字符串。

    示例：``sphere:42,8.5,2.2,2.0,1.0;box:58,6,1.5,4,3,1,0.8``。
    这种格式适合本地研究原型快速改参数，避免重新引入 YAML 主入口。
    """

    anomalies: list[Cavity] = []
    for idx, raw_item in enumerate(part.strip() for part in text.split(";")):
        if not raw_item:
            continue
        if ":" not in raw_item:
            raise ConfigError(f"异常体格式缺少 shape:values: {raw_item}")
        shape, values_text = raw_item.split(":", 1)
        shape = shape.strip().lower()
        try:
            values = [float(v.strip()) for v in values_text.split(",") if v.strip()]
        except ValueError as exc:
            raise ConfigError(f"异常体参数必须是数值: {raw_item}") from exc
        label = f"{shape}-{idx + 1}"
        if shape == "sphere":
            if len(values) != 5:
                raise ConfigError("sphere 格式为 sphere:x,y,h,radius,strength。")
            x, y, h, radius, strength = values
            if radius <= 0:
                raise ConfigError("sphere radius 必须为正数。")
            anomalies.append(_make_cavity(shape, x, y, h, radius, strength, attenuation_strength, tail_strength, diffraction_strength, label))
        elif shape == "box":
            if len(values) != 7:
                raise ConfigError("box 格式为 box:x,y,h,size_x,size_y,size_z,strength。")
            x, y, h, sx, sy, sz, strength = values
            if sx <= 0 or sy <= 0 or sz <= 0:
                raise ConfigError("box size_x/size_y/size_z 必须为正数。")
            anomalies.append(_make_cavity(shape, x, y, h, max(sx, sy, sz) / 2.0, strength, attenuation_strength, tail_strength, diffraction_strength, label, sx, sy, sz))
        elif shape == "cylinder":
            if len(values) != 6:
                raise ConfigError("cylinder 格式为 cylinder:x,y,h,radius,height,strength。")
            x, y, h, radius, height, strength = values
            if radius <= 0 or height <= 0:
                raise ConfigError("cylinder radius/height 必须为正数。")
            anomalies.append(_make_cavity(shape, x, y, h, radius, strength, attenuation_strength, tail_strength, diffraction_strength, label, radius, radius, height))
        elif shape == "ellipsoid":
            if len(values) != 7:
                raise ConfigError("ellipsoid 格式为 ellipsoid:x,y,h,size_x,size_y,size_z,strength。")
            x, y, h, sx, sy, sz, strength = values
            if sx <= 0 or sy <= 0 or sz <= 0:
                raise ConfigError("ellipsoid size_x/size_y/size_z 必须为正数。")
            anomalies.append(_make_cavity(shape, x, y, h, max(sx, sy, sz) / 2.0, strength, attenuation_strength, tail_strength, diffraction_strength, label, sx, sy, sz))
        elif shape in {"line", "zone"}:
            if len(values) != 6:
                raise ConfigError("line 格式为 line:x,y,h,length,azimuth,strength。")
            x, y, h, length, azimuth, strength = values
            if length <= 0:
                raise ConfigError("line/zone length 必须为正数。")
            anomalies.append(_make_cavity(shape, x, y, h, length / 2.0, strength, attenuation_strength, tail_strength, diffraction_strength, label, length, None, None, azimuth))
        else:
            raise ConfigError(f"未知异常体形状: {shape}")
    if not anomalies:
        raise ConfigError("--anomalies 未解析到有效异常体。")
    return anomalies


def _make_cavity(
    shape: str,
    x: float,
    y: float,
    h: float,
    radius: float,
    strength: float,
    attenuation_strength: float,
    tail_strength: float,
    diffraction_strength: float,
    label: str,
    size_x: float | None = None,
    size_y: float | None = None,
    size_z: float | None = None,
    azimuth: float = 0.0,
) -> Cavity:
    return Cavity(
        x0=x,
        y0=y,
        h=h,
        radius=max(radius, 0.1),
        scattering_strength=strength * diffraction_strength,
        attenuation_strength=attenuation_strength,
        tail_strength=tail_strength,
        label=label,
        shape=shape,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        azimuth=azimuth,
    )


def _arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    return np.arange(start, stop + 0.5 * step, step, dtype=float)


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{cls.__name__} 包含未知参数: {', '.join(sorted(unknown))}")
    values = {}
    for f in fields(cls):
        if f.name in data:
            values[f.name] = _normalize_sequence(data[f.name])
    return cls(**values)


def _normalize_sequence(value: Any) -> Any:
    if isinstance(value, list):
        if value and all(isinstance(item, list) for item in value):
            return tuple(tuple(float(v) for v in item) for item in value)
        if value and all(isinstance(item, (int, float)) for item in value):
            return tuple(float(v) for v in value)
    return value


def _load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise ConfigError("配置文件顶层必须是映射结构。")
        return loaded
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """读取本项目配置模板使用的简化 YAML 语法。"""

    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" "):
                key = line.rstrip(":").strip()
                current = {}
                root[key] = current
                continue
            if current is None or ":" not in line:
                raise ConfigError(f"无法解析配置行: {raw_line.strip()}")
            key, value = line.strip().split(":", 1)
            current[key.strip()] = _parse_scalar(value.strip())
    return root


def _parse_scalar(text: str) -> Any:
    if text in ("", "null", "None"):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text.startswith("[") and text.endswith("]"):
        return _parse_list(text)
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_list(text: str) -> list[Any]:
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        content = text[1:-1].strip()
        return [] if not content else [_parse_scalar(part.strip()) for part in content.split(",")]
    if not isinstance(value, list):
        raise ConfigError(f"列表参数解析失败: {text}")
    return value
