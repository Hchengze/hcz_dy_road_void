"""从 argparse/本地调试参数构建 ``RoadVoidConfig`` 的唯一主路径。"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .config import (
    CavityConfig,
    GeometryConfig,
    NoiseConfig,
    ProcessingConfig,
    RecordConfig,
    RoadVoidConfig,
    VelocityConfig,
)


def parse_float_list(text: str) -> tuple[float, ...]:
    """解析命令行中的逗号分隔浮点数列表。"""

    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析浮点数列表: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("浮点数列表不能为空。")
    return values


def parse_int_list(text: str) -> list[int]:
    """解析逗号分隔整数列表；空字符串返回空列表。"""

    if not text:
        return []
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析整数列表: {text}") from exc


def build_road_void_config_from_args(args: Any) -> RoadVoidConfig:
    """把 argparse 或 VSCode LOCAL 参数统一转换为 ``RoadVoidConfig``。

    本项目要求所有主流程都走同一条参数链：

    ``LOCAL_OUTPUT/LOCAL_WORKFLOW 或 argparse`` -> ``Namespace`` -> ``RoadVoidConfig``。
    """

    road_length = getattr(args, "road_length", 80.0)
    road_width = getattr(args, "road_width", 15.0)
    duration = getattr(args, "duration", 1.0)
    return RoadVoidConfig(
        geometry=GeometryConfig(
            road_width=road_width,
            road_length=road_length,
            source_y=road_width,
            channel_x_min=0.0,
            channel_x_max=road_length,
            channel_spacing=getattr(args, "channel_spacing", 1.0),
            source_x_min=0.0,
            source_x_max=road_length,
            source_spacing=getattr(args, "source_spacing", 4.0),
            fiber_depth=getattr(args, "fiber_depth", 0.0),
            source_depth=getattr(args, "source_depth", 0.0),
        ),
        cavity=CavityConfig(
            enable_cavity=not getattr(args, "no_cavity", False),
            cavity_x=getattr(args, "cavity_x", 42.0),
            cavity_y=getattr(args, "cavity_y", 8.5),
            cavity_h=getattr(args, "cavity_depth", 2.2),
            cavity_radius=getattr(args, "cavity_radius", 2.0),
            scattering_strength=getattr(args, "scattering_strength", 1.0),
            attenuation_strength=getattr(args, "attenuation_strength", 0.25),
            tail_strength=getattr(args, "tail_strength", 1.0),
            shape=getattr(args, "cavity_shape", "sphere"),
            size_x=getattr(args, "cavity_size_x", None),
            size_y=getattr(args, "cavity_size_y", None),
            size_z=getattr(args, "cavity_size_z", None),
            azimuth=getattr(args, "cavity_azimuth", 0.0),
            anomalies=getattr(args, "anomalies", None),
        ),
        velocity=VelocityConfig(
            rayleigh_velocity=getattr(args, "rayleigh_velocity", 240.0),
            velocity_model_type=getattr(args, "velocity_mode", "uniform"),
            layer_depths=parse_float_list(getattr(args, "layer_depths", "0.4,1.5,4.0")),
            layer_velocities=parse_float_list(getattr(args, "layer_velocities", "180,240,320")),
            sensitivity_depth_factor=getattr(args, "sensitivity_depth_factor", 0.5),
            source_frequency=getattr(args, "source_frequency", 35.0),
            wavelet_type=getattr(args, "wavelet", "ricker"),
        ),
        record=RecordConfig(
            sampling_rate=getattr(args, "sampling_rate", 1000.0),
            duration=duration,
            t0=getattr(args, "t0", 0.02),
            random_seed=getattr(args, "random_seed", 2027),
        ),
        noise=NoiseConfig(
            noise_level=getattr(args, "noise_level", 0.03),
            traffic_noise_level=getattr(args, "traffic_noise_level", 0.015),
            bad_channel_fraction=getattr(args, "bad_channel_fraction", 0.02),
            weak_coupling_fraction=getattr(args, "weak_coupling_fraction", 0.06),
            coupling_variation=getattr(args, "coupling_variation", 0.08),
            diffraction_strength=getattr(args, "scattering_strength", 1.0),
        ),
        processing=ProcessingConfig(
            scan_x_min=getattr(args, "scan_x_min", 32.0),
            scan_x_max=getattr(args, "scan_x_max", 52.0),
            scan_x_step=getattr(args, "scan_x_step", 1.0),
            scan_y_min=getattr(args, "scan_y_min", 3.0),
            scan_y_max=getattr(args, "scan_y_max", min(14.0, road_width - 1.0)),
            scan_y_step=getattr(args, "scan_y_step", 1.0),
            scan_h_min=getattr(args, "scan_h_min", 0.8),
            scan_h_max=getattr(args, "scan_h_max", 4.0),
            scan_h_step=getattr(args, "scan_h_step", 0.4),
            scan_vr_min=getattr(args, "scan_vr_min", 220.0),
            scan_vr_max=getattr(args, "scan_vr_max", 260.0),
            scan_vr_step=getattr(args, "scan_vr_step", 10.0),
            score_method=getattr(args, "score_method", "envelope"),
            top_k=getattr(args, "top_k", 8),
            uncertainty_threshold=getattr(args, "uncertainty_threshold", 0.92),
            direct_wave_mute_width=getattr(args, "direct_wave_mute_width", 0.04),
            scan_mode=getattr(args, "scan_mode", "joint"),
            shot_index=getattr(args, "shot_index", None),
            shot_weight_mode=getattr(args, "shot_weight_mode", "uniform"),
        ),
    )


def config_from_args(args: Any) -> RoadVoidConfig:
    """兼容旧测试/旧脚本的别名。"""

    return build_road_void_config_from_args(args)


def validate_config_consistency(cfg: RoadVoidConfig) -> None:
    """打印配置一致性 warning。

    严重参数错误仍由 ``RoadVoidConfig`` 自身校验处理；这里主要提示研究原型里常见的
    “异常体不在扫描范围内”“记录时长可能不足”等问题。
    """

    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()
    g = cfg.geometry
    p = cfg.processing
    if geom.n_channels < 3:
        print("警告：DAS 通道数过少，绕射曲线识别和扫描结果不可靠。")
    if geom.n_shots < 2:
        print("警告：炮点数过少，多炮联合约束基本失效。")
    if abs(float(geom.shot_y) - (g.source_y if g.source_y is not None else g.road_width)) > 1e-9:
        print("警告：RoadGeometry 中的炮线 y 与配置 source_y/road_width 不一致。")
    if abs(float(geom.fiber_y) - g.fiber_y) > 1e-9:
        print("警告：RoadGeometry 中的光纤 y 与配置 fiber_y 不一致。")

    for idx, cav in enumerate(cavities, start=1):
        prefix = f"警告：异常体 {idx}({cav.shape})"
        if cav.x0 < -0.05 * g.road_length or cav.x0 > 1.05 * g.road_length:
            print(f"{prefix} x={cav.x0:.2f} m 明显超出道路长度 [0, {g.road_length:.2f}]。")
        if cav.y0 < min(g.fiber_y, geom.shot_y) or cav.y0 > max(g.fiber_y, geom.shot_y):
            print(f"{prefix} y={cav.y0:.2f} m 不在光纤-炮线横向孔径内。")
        if cav.h <= 0:
            print(f"{prefix} depth/h={cav.h:.2f} m 非正。")
        if cav.radius <= 0:
            print(f"{prefix} radius={cav.radius:.2f} m 非正。")
        if not (p.scan_x_min <= cav.x0 <= p.scan_x_max):
            print("警告：当前扫描 x 范围没有覆盖设置的异常体位置，扫描结果可能找不到目标。")
        if not (p.scan_y_min <= cav.y0 <= p.scan_y_max):
            print("警告：当前扫描 y 范围没有覆盖设置的异常体位置，横向定位可能偏离目标。")
        if not (p.scan_h_min <= cav.h <= p.scan_h_max):
            print("警告：当前扫描 h 范围没有覆盖设置的异常体深度，深度扫描可能找不到目标。")

    vr_eff = cfg.effective_rayleigh_velocity()
    max_sg = float(np.max(geom.source_receiver_distances())) if geom.n_shots and geom.n_channels else 0.0
    needed = cfg.record.t0 + max_sg / vr_eff + 0.08
    if cfg.record.duration < needed:
        print(f"警告：记录长度 duration={cfg.record.duration:.3f}s 可能不足，估计至少需要约 {needed:.3f}s。")


def print_key_parameters(cfg: RoadVoidConfig, command: str | None = None) -> None:
    """打印当前完整参数摘要，便于 VSCode 输出窗口核对。"""

    cavities = cfg.to_cavities()
    vr_eff = cfg.effective_rayleigh_velocity()
    print("-" * 64)
    if command:
        print(f"当前运行模式：{command}")
    primary = "none" if not cavities else f"{cavities[0].shape}@({cavities[0].x0:.1f},{cavities[0].y0:.1f},{cavities[0].h:.1f})"
    print(
        "关键参数："
        f"W={cfg.geometry.road_width:.1f} m, L={cfg.geometry.road_length:.1f} m, "
        f"velocity-mode={cfg.velocity.velocity_model_type}, VR_eff={vr_eff:.1f} m/s, "
        f"anomalies={len(cavities)}, primary={primary}, scan-mode={cfg.processing.scan_mode}, "
        f"noise={cfg.noise.noise_level:.3f}"
    )
    print("道路与观测几何：")
    print(f"  road_width = {cfg.geometry.road_width:.2f} m, road_length = {cfg.geometry.road_length:.2f} m")
    print(f"  channel_spacing = {cfg.geometry.channel_spacing:.2f} m, source_spacing = {cfg.geometry.source_spacing:.2f} m")
    print("异常体：")
    if not cavities:
        print("  无异常体（--no-cavity）。")
    else:
        for idx, cav in enumerate(cavities, start=1):
            print(
                f"  {idx}) shape={cav.shape}, x={cav.x0:.2f}, y={cav.y0:.2f}, depth={cav.h:.2f}, "
                f"radius={cav.radius:.2f}, size=({cav.size_x},{cav.size_y},{cav.size_z}), "
                f"azimuth={cav.azimuth:.1f}, strength={cav.scattering_strength:.2f}"
            )
    wavelength = cfg.velocity.rayleigh_velocity / cfg.velocity.source_frequency
    print("速度与频率：")
    print(f"  velocity_mode = {cfg.velocity.velocity_model_type}")
    print(f"  VR = {cfg.velocity.rayleigh_velocity:.2f} m/s, VR_eff = {vr_eff:.2f} m/s")
    print(f"  source_frequency = {cfg.velocity.source_frequency:.2f} Hz, lambda=VR/f = {wavelength:.2f} m")
    if cfg.velocity.velocity_model_type == "layered-effective":
        print(f"  layer_depths = {cfg.velocity.layer_depths}, layer_velocities = {cfg.velocity.layer_velocities}")
    print("扫描范围：")
    print(f"  x=[{cfg.processing.scan_x_min:.2f}, {cfg.processing.scan_x_max:.2f}], step={cfg.processing.scan_x_step:.2f}")
    print(f"  y=[{cfg.processing.scan_y_min:.2f}, {cfg.processing.scan_y_max:.2f}], step={cfg.processing.scan_y_step:.2f}")
    print(f"  h=[{cfg.processing.scan_h_min:.2f}, {cfg.processing.scan_h_max:.2f}], step={cfg.processing.scan_h_step:.2f}")
    print(f"  scan_mode={cfg.processing.scan_mode}, shot_weight_mode={cfg.processing.shot_weight_mode}, noise={cfg.noise.noise_level:.3f}")
    print("-" * 64)


def prepare_road_void_config(args: Any, command: str | None = None) -> RoadVoidConfig:
    """构建、检查并回显配置；所有主流程入口都应调用这个函数。"""

    cfg = build_road_void_config_from_args(args)
    print_key_parameters(cfg, command or getattr(args, "command", None))
    validate_config_consistency(cfg)
    return cfg


def velocity_plot_info(cfg: RoadVoidConfig) -> dict[str, object]:
    """整理速度图标题/图注需要的参数，确保图件和正演扫描使用同一配置。"""

    vr = cfg.velocity.rayleigh_velocity
    f0 = cfg.velocity.source_frequency
    return {
        "velocity_mode": cfg.velocity.velocity_model_type,
        "rayleigh_velocity": vr,
        "effective_velocity": cfg.effective_rayleigh_velocity(),
        "source_frequency": f0,
        "wavelength": vr / f0,
        "sensitivity_depth_factor": cfg.velocity.sensitivity_depth_factor,
        "layer_depths": cfg.velocity.layer_depths,
        "layer_velocities": cfg.velocity.layer_velocities,
    }


def select_wavefield_shots(args: Any, geom: Any, cavities: list[Any]) -> list[int]:
    """根据 wavefield 参数选择单炮或多炮索引。"""

    n = geom.n_shots
    if n <= 0:
        return []
    mode = getattr(args, "wavefield_mode", "single-shot")
    if mode == "single-shot":
        if getattr(args, "wavefield_shot_index", None) is not None:
            return [max(0, min(n - 1, int(args.wavefield_shot_index)))]
        if cavities:
            return [min(range(n), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))]
        return [n // 2]
    explicit = parse_int_list(getattr(args, "wavefield_shot_indices", ""))
    if explicit:
        selected = [idx for idx in explicit if 0 <= idx < n]
    else:
        step = max(1, int(getattr(args, "wavefield_shot_step", 5)))
        selected = list(range(0, n, step))
    max_shots = max(1, int(getattr(args, "wavefield_max_shots", 5)))
    return selected[:max_shots]
