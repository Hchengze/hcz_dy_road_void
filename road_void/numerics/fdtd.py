"""FDTD 路线说明与轻量对比辅助。

当前项目中真正的 FDTD 原型仍在 ``road_void.elastic3d``。本文件不复制
elastic3d，只提供文字说明和小型摘要函数，便于把 FDTD 与 FEM/BEM/SEM
放在同一 numerics 目录下理解。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .validation import check_array_finite, check_energy_not_exploding


FloatArray = NDArray[np.float64]


@dataclass
class FDTD1DResult:
    """1D 标量波 FDTD benchmark 输出。"""

    x: FloatArray
    time: FloatArray
    snapshots: FloatArray
    receiver_trace: FloatArray
    source_index: int
    receiver_index: int
    cfl: float


def describe_fdtd_route() -> str:
    """返回 FDTD 路线说明。"""

    return (
        "FDTD/finite-difference 路线使用规则网格和差分模板更新波场。"
        "本项目的 road_void.elastic3d 属于小尺度三维 velocity-stress FDTD 原型，"
        "适合教学和规则网格波场 sanity check；复杂曲面边界会有 stair-step 近似，"
        "后续可继续发展严格 staggered-grid、完整 CPML 和 DAS 应变响应。"
    )


def compare_fdtd_to_kinematic() -> dict[str, str]:
    """返回 FDTD 与运动学正演的概念对比。"""

    return {
        "kinematic": "直接用 t_direct/t_diff 构造事件，速度快，适合扫描定位和参数敏感性。",
        "fdtd": "显式更新波场变量，更接近全波场，但计算量更大，默认不替代 workflow。",
        "relationship": "两者是不同层级工具：运动学模型做快速验证，FDTD 做小模型物理 sanity check。",
    }


def run_fdtd1d_wave_demo(
    n_points: int = 201,
    length: float = 100.0,
    velocity: float = 300.0,
    duration: float = 0.24,
    dt: float | None = None,
    source_position: float | None = None,
    receiver_position: float | None = None,
    source_frequency: float = 35.0,
    save: bool = False,
    show: bool = False,
    outdir: str | Path = "outputs/numerics",
    dpi: int = 180,
) -> FDTD1DResult:
    """运行 1D 均匀标量波方程 FDTD benchmark。

    方程为 ``u_tt = c^2 u_xx + f``。这里使用二阶中心差分：

    ``u_next = 2*u - u_prev + (c*dt/dx)^2 * (u[i+1]-2u[i]+u[i-1]) + dt^2*f``。

    该函数只是为了和 FEM/SEM 的 1D 教学算例做同题对比，不替代
    ``road_void.elastic3d``，也不是道路空洞主正演。
    """

    if n_points < 5:
        raise ValueError("FDTD 1D 至少需要 5 个网格点。")
    if length <= 0 or velocity <= 0 or duration <= 0 or source_frequency <= 0:
        raise ValueError("length、velocity、duration、source_frequency 必须为正。")
    x = np.linspace(0.0, length, n_points, dtype=float)
    dx = float(x[1] - x[0])
    dt = float(dt or 0.45 * dx / velocity)
    cfl = velocity * dt / dx
    if cfl >= 1.0:
        raise ValueError(f"FDTD 1D CFL={cfl:.3f} >= 1，显式格式不稳定，请减小 dt 或增大 dx。")
    n_steps = max(8, int(duration / dt) + 1)
    time = np.arange(n_steps, dtype=float) * dt
    source_position = 0.25 * length if source_position is None else source_position
    receiver_position = 0.75 * length if receiver_position is None else receiver_position
    source_index = int(np.argmin(np.abs(x - source_position)))
    receiver_index = int(np.argmin(np.abs(x - receiver_position)))

    u_prev = np.zeros(n_points, dtype=float)
    u = np.zeros(n_points, dtype=float)
    receiver_trace = np.zeros(n_steps, dtype=float)
    snapshot_ids = np.linspace(0, n_steps - 1, 5, dtype=int)
    snapshots = []
    source = _ricker(time, source_frequency)
    coeff = cfl**2

    for it in range(n_steps):
        lap = np.zeros_like(u)
        lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        u_next = 2.0 * u - u_prev + coeff * lap
        u_next[source_index] += dt**2 * source[it]
        # 固定端边界。benchmark 只比较首次到时，避免后期边界反射主导。
        u_next[0] = 0.0
        u_next[-1] = 0.0
        receiver_trace[it] = u[receiver_index]
        if it in snapshot_ids:
            snapshots.append(u.copy())
        u_prev, u = u, u_next

    result = FDTD1DResult(x, time, np.asarray(snapshots), receiver_trace, source_index, receiver_index, cfl)
    check_array_finite(result.snapshots, "fdtd snapshots")
    check_energy_not_exploding(result.receiver_trace, growth_limit=1.0e6, name="fdtd receiver trace")
    _plot_fdtd_result(result, Path(outdir), save, show, dpi)
    return result


def _ricker(time: FloatArray, frequency: float) -> FloatArray:
    t0 = 1.5 / frequency
    arg = np.pi * frequency * (time - t0)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


def _plot_fdtd_result(result: FDTD1DResult, outdir: Path, save: bool, show: bool, dpi: int) -> None:
    if not save and not show:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.8))
    for snap in result.snapshots:
        plt.plot(result.x, snap, lw=1.2)
    plt.axvline(result.x[result.source_index], color="tab:red", ls="--", label="点源")
    plt.axvline(result.x[result.receiver_index], color="tab:green", ls=":", label="接收点")
    plt.xlabel("x (m)")
    plt.ylabel("u")
    plt.title(f"FDTD 1D 标量波 benchmark：波场快照，CFL={result.cfl:.2f}")
    plt.legend()
    _finish(outdir / "fdtd1d_wavefield.png", save, show, dpi)

    plt.figure(figsize=(8.5, 4.2))
    plt.plot(result.time, result.receiver_trace)
    plt.xlabel("时间 (s)")
    plt.ylabel("u(receiver)")
    plt.title("FDTD 1D 标量波 benchmark：接收记录")
    _finish(outdir / "fdtd1d_receiver_trace.png", save, show, dpi)


def _finish(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
