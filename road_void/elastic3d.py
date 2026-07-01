"""小尺度三维弹性波全波形正演原型。

本模块用于研究和教学展示，不是工业级 3D elastic FDTD。实现采用
三维各向同性 velocity-stress 一阶方程、前/后向配对差分、简化自由
表面和 sponge 吸收边界。为保持代码短小，变量以同尺寸 NumPy 数组存储；
它表达 velocity-stress 有限差分的核心物理关系，但不追求高阶严格交错
网格、CPML、GPU/MPI 或生产级数值精度。
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
    vs = np.tile(vs_depth[None, None, :], (nx, ny, 1)).astype(float)
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
    return Elastic3DModel(vp=vp, vs=vs, rho=rho, lam=lam, mu=mu, dx=config.dx, dy=config.dy, dz=config.dz)


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
    sponge = _sponge_mask(cfg)

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
        dvx_dx = _ddx_f(vx, cfg.dx)
        dvy_dy = _ddy_f(vy, cfg.dy)
        dvz_dz = _ddz_f(vz, cfg.dz)
        sxx += cfg.dt * ((model.lam + 2 * model.mu) * dvx_dx + model.lam * (dvy_dy + dvz_dz))
        syy += cfg.dt * ((model.lam + 2 * model.mu) * dvy_dy + model.lam * (dvx_dx + dvz_dz))
        szz += cfg.dt * ((model.lam + 2 * model.mu) * dvz_dz + model.lam * (dvx_dx + dvy_dy))
        sxy += cfg.dt * model.mu * (_ddy_f(vx, cfg.dy) + _ddx_f(vy, cfg.dx))
        sxz += cfg.dt * model.mu * (_ddz_f(vx, cfg.dz) + _ddx_f(vz, cfg.dx))
        syz += cfg.dt * model.mu * (_ddz_f(vy, cfg.dz) + _ddy_f(vz, cfg.dy))

        if cfg.free_surface:
            szz[:, :, 0] = 0.0
            sxz[:, :, 0] = 0.0
            syz[:, :, 0] = 0.0

        fx = _ddx_b(sxx, cfg.dx) + _ddy_b(sxy, cfg.dy) + _ddz_b(sxz, cfg.dz)
        fy = _ddx_b(sxy, cfg.dx) + _ddy_b(syy, cfg.dy) + _ddz_b(syz, cfg.dz)
        fz = _ddx_b(sxz, cfg.dx) + _ddy_b(syz, cfg.dy) + _ddz_b(szz, cfg.dz)
        vx += cfg.dt * fx / model.rho
        vy += cfg.dt * fy / model.rho
        vz += cfg.dt * fz / model.rho
        # 垂向锤击力源。这里加载到速度场，等效于局部竖向脉冲力。
        vz[src] += cfg.dt * cfg.source_amplitude * source[it] / model.rho[src]

        vx *= sponge
        vy *= sponge
        vz *= sponge
        sxx *= sponge
        syy *= sponge
        szz *= sponge
        sxy *= sponge
        sxz *= sponge
        syz *= sponge

        gather[it, :] = vz[rec_x_idx, rec_y, rec_z]
        for name, step in snapshot_steps.items():
            if it == step:
                # x-z 剖面，显示通过震源 y 的垂向速度场。
                snapshots[name] = vz[:, src[1], :].copy()

    return Elastic3DResult(cfg, model, gather, receiver_x, snapshots, cfl)


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
    _plot_gather(result, outdir / "elastic3d_gather.png", save, show, dpi)


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


def _ddx_f(a: FloatArray, dx: float) -> FloatArray:
    out = np.zeros_like(a)
    out[:-1, :, :] = (a[1:, :, :] - a[:-1, :, :]) / dx
    return out


def _ddy_f(a: FloatArray, dy: float) -> FloatArray:
    out = np.zeros_like(a)
    out[:, :-1, :] = (a[:, 1:, :] - a[:, :-1, :]) / dy
    return out


def _ddz_f(a: FloatArray, dz: float) -> FloatArray:
    out = np.zeros_like(a)
    out[:, :, :-1] = (a[:, :, 1:] - a[:, :, :-1]) / dz
    return out


def _ddx_b(a: FloatArray, dx: float) -> FloatArray:
    out = np.zeros_like(a)
    out[1:, :, :] = (a[1:, :, :] - a[:-1, :, :]) / dx
    return out


def _ddy_b(a: FloatArray, dy: float) -> FloatArray:
    out = np.zeros_like(a)
    out[:, 1:, :] = (a[:, 1:, :] - a[:, :-1, :]) / dy
    return out


def _ddz_b(a: FloatArray, dz: float) -> FloatArray:
    out = np.zeros_like(a)
    out[:, :, 1:] = (a[:, :, 1:] - a[:, :, :-1]) / dz
    return out


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
    plt.colorbar(label="vz (m/s)")
    plt.xlabel("接收线 x (m)")
    plt.ylabel("时间 (s)")
    plt.title("三维弹性波全波形有限差分原型，小尺度教学/研究模型：接收记录")
    _finish_elastic_figure(output, save, show, dpi)


def _finish_elastic_figure(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
