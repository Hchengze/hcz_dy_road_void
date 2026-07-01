"""小尺度三维弹性波全波形正演原型。

本模块用于研究和教学展示，不是工业级 3D elastic FDTD。实现采用
三维各向同性 velocity-stress 一阶方程、二阶/四阶前后向配对差分、
简化自由表面和 sponge/experimental-cpml 吸收边界。为保持代码短小，
变量以同尺寸 NumPy 数组存储；它表达 velocity-stress 有限差分的核心
物理关系，但当前仍不是严格工业级交错网格、完整 CPML、GPU/MPI 或
生产级数值精度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from numpy.typing import NDArray

from .anomaly import Cavity


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Elastic3DConfig:
    """三维弹性波小模型参数。"""

    nx: int = 56
    ny: int = 36
    nz: int = 28
    dx: float = 0.5
    dy: float = 0.5
    dz: float = 0.5
    dt: float = 0.00012
    nt: int = 420
    source_frequency: float = 60.0
    source_amplitude: float = 1.0e5
    vs_scale: float = 1.0
    space_order: int = 2
    abc: str = "sponge"
    record_component: str = "vz"
    gauge_length: float = 1.0
    sponge_width: int = 6
    sponge_strength: float = 0.018
    free_surface: bool = True
    with_anomaly: bool = True
    anomaly_x: float = 14.0
    anomaly_y: float = 8.5
    anomaly_z: float = 2.5
    anomaly_radius: float = 1.5
    random_seed: int = 2027


@dataclass
class Elastic3DModel:
    """三维弹性模型参数。"""

    vp: FloatArray
    vs: FloatArray
    rho: FloatArray
    lam: FloatArray
    mu: FloatArray
    dx: float
    dy: float
    dz: float


@dataclass
class Elastic3DResult:
    """三维弹性正演输出。"""

    config: Elastic3DConfig
    model: Elastic3DModel
    gather: FloatArray
    receiver_x: FloatArray
    record_component: str = "vz"
    snapshots: dict[str, FloatArray] = field(default_factory=dict)
    cfl: float = 0.0


def build_elastic3d_model(config: Elastic3DConfig, cavities: list[Cavity] | None = None) -> Elastic3DModel:
    """构建三层 Vp/Vs/rho 模型，并可加入低速低密度异常体。"""

    nx, ny, nz = config.nx, config.ny, config.nz
    z = np.arange(nz, dtype=float) * config.dz
    vp_depth = np.where(z < 1.0, 520.0, np.where(z < 4.0, 720.0, 900.0))
    vs_depth = np.where(z < 1.0, 220.0, np.where(z < 4.0, 320.0, 420.0))
    rho_depth = np.where(z < 1.0, 1800.0, np.where(z < 4.0, 1900.0, 2050.0))
    vp = np.tile(vp_depth[None, None, :], (nx, ny, 1)).astype(float)
    vs = np.tile((config.vs_scale * vs_depth)[None, None, :], (nx, ny, 1)).astype(float)
    rho = np.tile(rho_depth[None, None, :], (nx, ny, 1)).astype(float)

    if config.with_anomaly:
        cavs = cavities or [Cavity(config.anomaly_x, config.anomaly_y, config.anomaly_z, radius=config.anomaly_radius, shape="sphere")]
        xx, yy, zz = _coordinate_grids(config)
        for cav in cavs:
            mask = _elastic_anomaly_mask(cav, xx, yy, zz)
            vp[mask] *= 0.42
            vs[mask] *= 0.35
            rho[mask] *= 0.55

    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu
    model = Elastic3DModel(vp=vp, vs=vs, rho=rho, lam=lam, mu=mu, dx=config.dx, dy=config.dy, dz=config.dz)
    validate_elastic_model(model)
    return model


def validate_elastic_model(model: Elastic3DModel) -> None:
    """做最基本的物理 sanity check，避免错误模型静默进入正演。"""

    if not np.all(np.isfinite(model.vp)) or not np.all(np.isfinite(model.vs)) or not np.all(np.isfinite(model.rho)):
        raise ValueError("elastic3d 模型包含非有限数值。")
    if np.any(model.vp <= 0) or np.any(model.vs <= 0):
        raise ValueError("elastic3d 模型速度必须为正。")
    if np.any(model.rho <= 0):
        raise ValueError("elastic3d 模型密度必须为正。")
    if np.any(model.vp <= model.vs):
        raise ValueError("elastic3d 要求 Vp > Vs > 0。")
    if np.any(model.mu <= 0):
        raise ValueError("elastic3d 剪切模量 mu 必须为正。")
    if np.any(model.lam <= 0):
        raise ValueError("elastic3d 当前默认模型要求 lambda > 0，请检查 Vp/Vs/rho。")


def check_cfl(config: Elastic3DConfig, model: Elastic3DModel, limit: float = 0.45) -> float:
    """检查三维显式有限差分稳定性。

    这里使用保守 CFL 数：
    ``vmax * dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)``。超过 ``limit`` 时直接
    报错，避免生成看似有图但数值已经爆炸的结果。
    """

    vmax = float(np.max(model.vp))
    cfl = vmax * config.dt * np.sqrt(1.0 / config.dx**2 + 1.0 / config.dy**2 + 1.0 / config.dz**2)
    if cfl >= limit:
        raise ValueError(
            f"CFL 不稳定: cfl={cfl:.3f} >= {limit:.2f}。请减小 dt 或增大 dx/dy/dz。"
        )
    return float(cfl)


def run_elastic3d(config: Elastic3DConfig | None = None, cavities: list[Cavity] | None = None) -> Elastic3DResult:
    """运行小尺度三维弹性波 velocity-stress FDTD。"""

    cfg = config or Elastic3DConfig()
    _validate_elastic_config(cfg)
    model = build_elastic3d_model(cfg, cavities)
    cfl = check_cfl(cfg, model)
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    vx = np.zeros((nx, ny, nz), dtype=float)
    vy = np.zeros_like(vx)
    vz = np.zeros_like(vx)
    sxx = np.zeros_like(vx)
    syy = np.zeros_like(vx)
    szz = np.zeros_like(vx)
    sxy = np.zeros_like(vx)
    sxz = np.zeros_like(vx)
    syz = np.zeros_like(vx)
    ux = np.zeros_like(vx) if cfg.record_component == "strain_xx" else None
    damping = _absorbing_mask(cfg)

    src_y = max(cfg.sponge_width + 3, ny - cfg.sponge_width - 3)
    rec_y = min(cfg.sponge_width + 2, ny - cfg.sponge_width - 4)
    src = (nx // 2, src_y, 2)
    rec_x_idx = np.arange(4, nx - 4, 2, dtype=int)
    rec_z = 2
    gather = np.zeros((cfg.nt, rec_x_idx.size), dtype=float)
    receiver_x = rec_x_idx * cfg.dx
    snapshot_steps = {
        "wavefield_snapshot_early": max(1, cfg.nt // 4),
        "wavefield_snapshot_mid": max(2, cfg.nt // 2),
        "wavefield_snapshot_late": max(3, 3 * cfg.nt // 4),
    }
    snapshots: dict[str, FloatArray] = {}
    source = _ricker_series(cfg.nt, cfg.dt, cfg.source_frequency)

    for it in range(cfg.nt):
        dvx_dx = _ddx_f(vx, cfg.dx, cfg.space_order)
        dvy_dy = _ddy_f(vy, cfg.dy, cfg.space_order)
        dvz_dz = _ddz_f(vz, cfg.dz, cfg.space_order)
        sxx += cfg.dt * ((model.lam + 2 * model.mu) * dvx_dx + model.lam * (dvy_dy + dvz_dz))
        syy += cfg.dt * ((model.lam + 2 * model.mu) * dvy_dy + model.lam * (dvx_dx + dvz_dz))
        szz += cfg.dt * ((model.lam + 2 * model.mu) * dvz_dz + model.lam * (dvx_dx + dvy_dy))
        sxy += cfg.dt * model.mu * (_ddy_f(vx, cfg.dy, cfg.space_order) + _ddx_f(vy, cfg.dx, cfg.space_order))
        sxz += cfg.dt * model.mu * (_ddz_f(vx, cfg.dz, cfg.space_order) + _ddx_f(vz, cfg.dx, cfg.space_order))
        syz += cfg.dt * model.mu * (_ddz_f(vy, cfg.dz, cfg.space_order) + _ddy_f(vz, cfg.dy, cfg.space_order))

        if cfg.free_surface:
            szz[:, :, 0] = 0.0
            sxz[:, :, 0] = 0.0
            syz[:, :, 0] = 0.0

        fx = _ddx_b(sxx, cfg.dx, cfg.space_order) + _ddy_b(sxy, cfg.dy, cfg.space_order) + _ddz_b(sxz, cfg.dz, cfg.space_order)
        fy = _ddx_b(sxy, cfg.dx, cfg.space_order) + _ddy_b(syy, cfg.dy, cfg.space_order) + _ddz_b(syz, cfg.dz, cfg.space_order)
        fz = _ddx_b(sxz, cfg.dx, cfg.space_order) + _ddy_b(syz, cfg.dy, cfg.space_order) + _ddz_b(szz, cfg.dz, cfg.space_order)
        vx += cfg.dt * fx / model.rho
        vy += cfg.dt * fy / model.rho
        vz += cfg.dt * fz / model.rho
        # 垂向锤击力源。这里加载到速度场，等效于局部竖向脉冲力。
        vz[src] += cfg.dt * cfg.source_amplitude * source[it] / model.rho[src]

        vx *= damping
        vy *= damping
        vz *= damping
        sxx *= damping
        syy *= damping
        szz *= damping
        sxy *= damping
        sxz *= damping
        syz *= damping
        if ux is not None:
            ux += cfg.dt * vx
            ux *= damping

        gather[it, :] = _record_receivers(cfg, vx, vz, ux, rec_x_idx, rec_y, rec_z)
        for name, step in snapshot_steps.items():
            if it == step:
                # x-z 剖面，显示通过震源 y 的垂向速度场。
                snapshots[name] = vz[:, src[1], :].copy()

    return Elastic3DResult(cfg, model, gather, receiver_x, cfg.record_component, snapshots, cfl)


def plot_elastic3d_outputs(
    result: Elastic3DResult,
    outdir: str | Path,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """保存/显示 elastic3d 速度模型、快照和 gather。"""

    outdir = Path(outdir)
    _plot_velocity_slice(result, outdir / "velocity_model_slice.png", save, show, dpi)
    for name, snapshot in result.snapshots.items():
        _plot_snapshot(result, snapshot, outdir / f"{name}.png", save, show, dpi)
    component_output = outdir / f"elastic3d_gather_{result.record_component}.png"
    _plot_gather(result, component_output, save, show, dpi)
    if result.record_component == "vz":
        _plot_gather(result, outdir / "elastic3d_gather.png", save, False, dpi)


def animate_elastic3d_wavefield(
    result: Elastic3DResult,
    outdir: str | Path,
    save: bool = True,
    show: bool = False,
    fps: int = 8,
) -> None:
    """用已保存的少量快照生成教学 GIF。"""

    if not result.snapshots:
        return
    outdir = Path(outdir)
    frames = list(result.snapshots.values())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    vmax = max(float(np.max(np.abs(frame))) for frame in frames) + 1e-12
    image = ax.imshow(frames[0].T, origin="upper", aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x 网格")
    ax.set_ylabel("z 网格")
    ax.set_title("三维弹性波全波形有限差分原型，小尺度教学/研究模型")

    def update(i: int) -> list[object]:
        image.set_data(frames[i].T)
        ax.set_title(f"三维弹性波全波形有限差分原型，小尺度教学/研究模型；frame {i + 1}")
        return [image]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        anim.save(outdir / "elastic3d_wavefield.gif", writer=animation.PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_abc_comparison(
    config: Elastic3DConfig,
    outdir: str | Path,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """输出 sponge 与 experimental cpml-like 的短时对比图。

    这个函数只用于人工 sanity check：比较两种边界阻尼下接收记录的整体
    能量随时间变化。它不能证明 CPML 严格正确。
    """

    short_nt = min(config.nt, 360)
    base = config.__dict__.copy()
    base.update({"nt": short_nt, "record_component": "vz"})
    sponge = run_elastic3d(Elastic3DConfig(**{**base, "abc": "sponge"}))
    cpml = run_elastic3d(Elastic3DConfig(**{**base, "abc": "cpml"}))
    t = np.arange(short_nt, dtype=float) * config.dt
    sponge_energy = np.sqrt(np.mean(sponge.gather**2, axis=1))
    cpml_energy = np.sqrt(np.mean(cpml.gather**2, axis=1))
    plt.figure(figsize=(7.4, 4.6))
    plt.plot(t, sponge_energy, label="sponge")
    plt.plot(t, cpml_energy, label="experimental cpml-like")
    plt.xlabel("时间 (s)")
    plt.ylabel("接收记录 RMS")
    plt.title("elastic3d 边界吸收 sanity check：sponge vs experimental cpml-like（非严格 CPML 证明）")
    plt.legend()
    _finish_elastic_figure(Path(outdir) / "abc_compare_sponge_vs_cpml.png", save, show, dpi)


def _coordinate_grids(config: Elastic3DConfig) -> tuple[FloatArray, FloatArray, FloatArray]:
    x = np.arange(config.nx, dtype=float)[:, None, None] * config.dx
    y = np.arange(config.ny, dtype=float)[None, :, None] * config.dy
    z = np.arange(config.nz, dtype=float)[None, None, :] * config.dz
    return np.broadcast_to(x, (config.nx, config.ny, config.nz)), np.broadcast_to(y, (config.nx, config.ny, config.nz)), np.broadcast_to(z, (config.nx, config.ny, config.nz))


def _elastic_anomaly_mask(cavity: Cavity, xx: FloatArray, yy: FloatArray, zz: FloatArray) -> FloatArray:
    shape = cavity.shape.lower()
    if shape in {"sphere", "cylinder"}:
        r = cavity.size_x or cavity.radius
        if shape == "cylinder":
            height = cavity.size_z or 2 * cavity.radius
            radial = (xx - cavity.x0) ** 2 + (yy - cavity.y0) ** 2 <= r**2
            vertical = np.abs(zz - cavity.h) <= 0.5 * height
            return radial & vertical
        return (xx - cavity.x0) ** 2 + (yy - cavity.y0) ** 2 + (zz - cavity.h) ** 2 <= r**2
    if shape == "box":
        sx = cavity.size_x or 2 * cavity.radius
        sy = cavity.size_y or 2 * cavity.radius
        sz = cavity.size_z or cavity.radius
        return (np.abs(xx - cavity.x0) <= sx / 2) & (np.abs(yy - cavity.y0) <= sy / 2) & (np.abs(zz - cavity.h) <= sz / 2)
    if shape == "ellipsoid":
        sx = cavity.size_x or 2 * cavity.radius
        sy = cavity.size_y or 1.4 * cavity.radius
        sz = cavity.size_z or cavity.radius
        return ((xx - cavity.x0) / (sx / 2)) ** 2 + ((yy - cavity.y0) / (sy / 2)) ** 2 + ((zz - cavity.h) / (sz / 2)) ** 2 <= 1.0
    if shape in {"line", "zone"}:
        theta = np.deg2rad(cavity.azimuth)
        along = (xx - cavity.x0) * np.cos(theta) + (yy - cavity.y0) * np.sin(theta)
        cross = -(xx - cavity.x0) * np.sin(theta) + (yy - cavity.y0) * np.cos(theta)
        length = cavity.size_x or 3 * cavity.radius
        width = cavity.size_y or (1.5 * cavity.radius if shape == "zone" else 0.5)
        height = cavity.size_z or 0.8
        return (np.abs(along) <= length / 2) & (np.abs(cross) <= width / 2) & (np.abs(zz - cavity.h) <= height / 2)
    return np.zeros_like(xx, dtype=bool)


def _validate_elastic_config(config: Elastic3DConfig) -> None:
    if config.space_order not in {2, 4}:
        raise ValueError("elastic-space-order 目前只支持 2 或 4。")
    if config.abc not in {"sponge", "cpml"}:
        raise ValueError("elastic-abc 目前只支持 sponge 或 cpml。")
    if config.record_component not in {"vz", "vx", "strain_xx", "strain_rate_xx"}:
        raise ValueError("elastic-record-component 只能是 vz、vx、strain_xx 或 strain_rate_xx。")
    if min(config.nx, config.ny, config.nz) < 12:
        raise ValueError("elastic3d 网格太小，建议每个方向至少 12 个网格。")
    if config.dx <= 0 or config.dy <= 0 or config.dz <= 0 or config.dt <= 0:
        raise ValueError("dx/dy/dz/dt 必须为正。")
    if config.vs_scale <= 0:
        raise ValueError("vs_scale 必须为正。")
    if config.gauge_length <= 0:
        raise ValueError("elastic-gauge-length 必须为正。")
    if config.source_amplitude <= 0:
        raise ValueError("震源幅值必须为正，过大可能导致数值饱和。")


def _absorbing_mask(config: Elastic3DConfig) -> FloatArray:
    """返回边界阻尼因子。

    ``sponge`` 是简单指数海绵层。``cpml`` 当前是 experimental CPML-like
    多项式阻尼，只模仿 PML 中“边界阻尼随深度增强”的思想，没有实现完整
    CPML 辅助记忆变量，因此文档和控制台都必须称为实验选项。
    """

    if config.abc == "cpml":
        return _cpml_like_mask(config)
    return _sponge_mask(config)


def _sponge_mask(config: Elastic3DConfig) -> FloatArray:
    mask = np.ones((config.nx, config.ny, config.nz), dtype=float)
    width = max(int(config.sponge_width), 1)
    for axis, n in enumerate((config.nx, config.ny, config.nz)):
        idx = np.arange(n, dtype=float)
        dist = np.minimum(idx, n - 1 - idx)
        if axis == 2 and config.free_surface:
            dist = np.minimum(n - 1 - idx, width)
        damp_1d = np.ones(n, dtype=float)
        edge = dist < width
        damp_1d[edge] = np.exp(-config.sponge_strength * ((width - dist[edge]) / width) ** 2)
        shape = [1, 1, 1]
        shape[axis] = n
        mask *= damp_1d.reshape(shape)
    return mask


def _cpml_like_mask(config: Elastic3DConfig) -> FloatArray:
    mask = np.ones((config.nx, config.ny, config.nz), dtype=float)
    width = max(int(config.sponge_width), 1)
    vmax_hint = 900.0
    sigma_max = 3.0 * vmax_hint * np.log(1000.0) / max(width * min(config.dx, config.dy, config.dz), 1e-6)
    for axis, n in enumerate((config.nx, config.ny, config.nz)):
        idx = np.arange(n, dtype=float)
        dist = np.minimum(idx, n - 1 - idx)
        if axis == 2 and config.free_surface:
            dist = np.minimum(n - 1 - idx, width)
        sigma = np.zeros(n, dtype=float)
        edge = dist < width
        sigma[edge] = sigma_max * ((width - dist[edge]) / width) ** 2
        damp_1d = np.exp(-sigma * config.dt)
        shape = [1, 1, 1]
        shape[axis] = n
        mask *= damp_1d.reshape(shape)
    return mask


def _record_receivers(
    config: Elastic3DConfig,
    vx: FloatArray,
    vz: FloatArray,
    ux: FloatArray | None,
    rec_x_idx: NDArray[np.integer],
    rec_y: int,
    rec_z: int,
) -> FloatArray:
    """记录检波器/DAS 近似响应。

    ``vz`` 和 ``vx`` 是普通速度分量；``strain_rate_xx`` 用
    ``dvx/dx`` 近似沿 x 方向光纤的 DAS 应变率；``strain_xx`` 用积分位移
    ``ux`` 做有限 gauge length 差分。真实 DAS 还受光纤方向、gauge length、
    耦合和解调方式影响，这里只是数值原型。
    """

    if config.record_component == "vz":
        return vz[rec_x_idx, rec_y, rec_z]
    if config.record_component == "vx":
        return vx[rec_x_idx, rec_y, rec_z]
    field = vx if config.record_component == "strain_rate_xx" else ux
    if field is None:
        raise ValueError("strain_xx 记录需要位移积分场 ux。")
    half = max(1, int(round(0.5 * config.gauge_length / config.dx)))
    left = np.clip(rec_x_idx - half, 0, config.nx - 1)
    right = np.clip(rec_x_idx + half, 0, config.nx - 1)
    length = np.maximum((right - left) * config.dx, config.dx)
    return (field[right, rec_y, rec_z] - field[left, rec_y, rec_z]) / length


def _ddx_f(a: FloatArray, dx: float, order: int = 2) -> FloatArray:
    return _diff_forward(a, dx, axis=0, order=order)


def _ddy_f(a: FloatArray, dy: float, order: int = 2) -> FloatArray:
    return _diff_forward(a, dy, axis=1, order=order)


def _ddz_f(a: FloatArray, dz: float, order: int = 2) -> FloatArray:
    return _diff_forward(a, dz, axis=2, order=order)


def _ddx_b(a: FloatArray, dx: float, order: int = 2) -> FloatArray:
    return _diff_backward(a, dx, axis=0, order=order)


def _ddy_b(a: FloatArray, dy: float, order: int = 2) -> FloatArray:
    return _diff_backward(a, dy, axis=1, order=order)


def _ddz_b(a: FloatArray, dz: float, order: int = 2) -> FloatArray:
    return _diff_backward(a, dz, axis=2, order=order)


def _diff_forward(a: FloatArray, spacing: float, axis: int, order: int) -> FloatArray:
    """前向导数。

    二阶模板为 ``f[i+1]-f[i]``。四阶模板使用常见交错差分系数：
    ``9/8*(f[i+1]-f[i]) - 1/24*(f[i+2]-f[i-1])``，边界附近自动退回二阶。
    当前变量仍以同尺寸数组存储，因此这是“交错模板风格”的近似，而不是
    严格变量错位存储。
    """

    moved = np.moveaxis(a, axis, 0)
    out = np.zeros_like(moved)
    out[:-1] = (moved[1:] - moved[:-1]) / spacing
    if order == 4 and moved.shape[0] >= 4:
        out[1:-2] = ((9.0 / 8.0) * (moved[2:-1] - moved[1:-2]) - (1.0 / 24.0) * (moved[3:] - moved[:-3])) / spacing
    return np.moveaxis(out, 0, axis)


def _diff_backward(a: FloatArray, spacing: float, axis: int, order: int) -> FloatArray:
    """后向导数，四阶时与前向模板配对。"""

    moved = np.moveaxis(a, axis, 0)
    out = np.zeros_like(moved)
    out[1:] = (moved[1:] - moved[:-1]) / spacing
    if order == 4 and moved.shape[0] >= 4:
        out[2:-1] = ((9.0 / 8.0) * (moved[2:-1] - moved[1:-2]) - (1.0 / 24.0) * (moved[3:] - moved[:-3])) / spacing
    return np.moveaxis(out, 0, axis)


def _ricker_series(nt: int, dt: float, frequency: float) -> FloatArray:
    t = np.arange(nt, dtype=float) * dt
    t0 = 1.0 / frequency
    arg = np.pi * frequency * (t - t0)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


def _plot_velocity_slice(result: Elastic3DResult, output: Path, save: bool, show: bool, dpi: int) -> None:
    cfg = result.config
    y = cfg.ny // 2
    plt.figure(figsize=(7.5, 4.8))
    plt.imshow(result.model.vp[:, y, :].T, origin="upper", aspect="auto", extent=[0, cfg.nx * cfg.dx, cfg.nz * cfg.dz, 0], cmap="viridis")
    plt.colorbar(label="Vp (m/s)")
    plt.xlabel("x (m)")
    plt.ylabel("z 深度 (m)")
    plt.title("三维弹性波全波形有限差分原型，小尺度教学/研究模型：Vp 切片")
    _finish_elastic_figure(output, save, show, dpi)


def _plot_snapshot(result: Elastic3DResult, snapshot: FloatArray, output: Path, save: bool, show: bool, dpi: int) -> None:
    cfg = result.config
    vmax = float(np.percentile(np.abs(snapshot), 99.5)) + 1e-12
    plt.figure(figsize=(7.5, 4.8))
    plt.imshow(snapshot.T, origin="upper", aspect="auto", extent=[0, cfg.nx * cfg.dx, cfg.nz * cfg.dz, 0], cmap="seismic", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="vz (m/s)")
    plt.xlabel("x (m)")
    plt.ylabel("z 深度 (m)")
    plt.title("三维弹性波全波形有限差分原型，小尺度教学/研究模型：vz 快照")
    _finish_elastic_figure(output, save, show, dpi)


def _plot_gather(result: Elastic3DResult, output: Path, save: bool, show: bool, dpi: int) -> None:
    cfg = result.config
    vmax = max(float(np.max(np.abs(result.gather))), 1e-12)
    plt.figure(figsize=(8.2, 4.8))
    plt.imshow(result.gather, origin="upper", aspect="auto", extent=[result.receiver_x[0], result.receiver_x[-1], cfg.nt * cfg.dt, 0], cmap="seismic", vmin=-vmax, vmax=vmax)
    label = _component_label(result.record_component)
    plt.colorbar(label=label)
    plt.xlabel("接收线 x (m)")
    plt.ylabel("时间 (s)")
    plt.title(f"三维弹性波全波形有限差分原型，小尺度教学/研究模型：{result.record_component} 接收记录")
    _finish_elastic_figure(output, save, show, dpi)


def _component_label(component: str) -> str:
    return {
        "vz": "vz (m/s)",
        "vx": "vx (m/s)",
        "strain_rate_xx": "近似 DAS strain_rate_xx (1/s)",
        "strain_xx": "近似 DAS strain_xx",
    }.get(component, component)


def _finish_elastic_figure(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
