"""1D 标量波方程 FEM 教学原型。

本模块实现线性有限元的最小波动算例：

    M u_tt + K u = f

其中 ``M`` 是质量矩阵，``K`` 是刚度矩阵，``u`` 是位移型标量波场。
它不是弹性波 FEM，也没有二维/三维非结构网格；目的只是让项目内有一个
可读、可运行、可测试的 FEM 入门原型。
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
class FEM1DResult:
    """1D FEM 波动算例输出。"""

    x: FloatArray
    time: FloatArray
    mass: FloatArray
    stiffness: FloatArray
    snapshots: FloatArray
    receiver_trace: FloatArray
    source_index: int
    receiver_index: int


def assemble_fem1d_matrices(n_nodes: int, length: float, velocity: float = 1.0, density: float = 1.0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """组装 1D 线性单元质量矩阵和刚度矩阵。

    对长度为 ``h`` 的线性单元，局部矩阵为：

    ``Me = rho*h/6 * [[2, 1], [1, 2]]``

    ``Ke = rho*c^2/h * [[1, -1], [-1, 1]]``

    这里 ``c`` 是标量波速度，单位可理解为 m/s。
    """

    if n_nodes < 3:
        raise ValueError("FEM 1D 至少需要 3 个节点。")
    if length <= 0 or velocity <= 0 or density <= 0:
        raise ValueError("length、velocity、density 必须为正。")
    x = np.linspace(0.0, length, n_nodes, dtype=float)
    h = length / (n_nodes - 1)
    mass = np.zeros((n_nodes, n_nodes), dtype=float)
    stiffness = np.zeros_like(mass)
    me = density * h / 6.0 * np.asarray([[2.0, 1.0], [1.0, 2.0]])
    ke = density * velocity**2 / h * np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    for e in range(n_nodes - 1):
        idx = np.asarray([e, e + 1])
        mass[np.ix_(idx, idx)] += me
        stiffness[np.ix_(idx, idx)] += ke
    return x, mass, stiffness


def run_fem1d_wave_demo(
    n_nodes: int = 101,
    length: float = 100.0,
    velocity: float = 300.0,
    duration: float = 0.28,
    dt: float | None = None,
    source_position: float | None = None,
    receiver_position: float | None = None,
    source_frequency: float = 35.0,
    save: bool = False,
    show: bool = False,
    outdir: str | Path = "outputs/numerics",
    dpi: int = 180,
) -> FEM1DResult:
    """运行 1D 标量波 FEM 教学算例。

    时间推进使用质量集总后的显式中心差分。边界采用固定端 ``u=0``，
    所以这里只适合短时教学传播，不用于真实道路空洞模拟。
    """

    x, mass, stiffness = assemble_fem1d_matrices(n_nodes, length, velocity)
    dx = x[1] - x[0]
    dt = float(dt or 0.35 * dx / velocity)
    n_steps = max(8, int(duration / dt) + 1)
    time = np.arange(n_steps, dtype=float) * dt
    source_position = 0.25 * length if source_position is None else source_position
    receiver_position = 0.75 * length if receiver_position is None else receiver_position
    source_index = int(np.argmin(np.abs(x - source_position)))
    receiver_index = int(np.argmin(np.abs(x - receiver_position)))
    lumped_mass = np.sum(mass, axis=1)
    u_prev = np.zeros(n_nodes, dtype=float)
    u = np.zeros(n_nodes, dtype=float)
    receiver_trace = np.zeros(n_steps, dtype=float)
    snapshot_ids = np.linspace(0, n_steps - 1, 5, dtype=int)
    snapshots = []
    source = _ricker(time, source_frequency)

    for it in range(n_steps):
        force = np.zeros(n_nodes, dtype=float)
        force[source_index] = source[it]
        accel = (force - stiffness @ u) / lumped_mass
        u_next = 2.0 * u - u_prev + dt**2 * accel
        # 固定端边界条件，表达两端位移为零。
        u_next[0] = 0.0
        u_next[-1] = 0.0
        receiver_trace[it] = u[receiver_index]
        if it in snapshot_ids:
            snapshots.append(u.copy())
        u_prev, u = u, u_next

    result = FEM1DResult(x, time, mass, stiffness, np.asarray(snapshots), receiver_trace, source_index, receiver_index)
    check_array_finite(result.snapshots, "fem snapshots")
    check_energy_not_exploding(result.receiver_trace, growth_limit=1.0e6, name="fem receiver trace")
    _plot_fem_result(result, Path(outdir), save, show, dpi)
    return result


def _ricker(time: FloatArray, frequency: float) -> FloatArray:
    t0 = 1.5 / frequency
    arg = np.pi * frequency * (time - t0)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


def _plot_fem_result(result: FEM1DResult, outdir: Path, save: bool, show: bool, dpi: int) -> None:
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
    plt.title("FEM 1D 标量波教学原型：波场快照")
    plt.legend()
    _finish(outdir / "fem1d_wavefield.png", save, show, dpi)

    plt.figure(figsize=(8.5, 4.2))
    plt.plot(result.time, result.receiver_trace)
    plt.xlabel("时间 (s)")
    plt.ylabel("u(receiver)")
    plt.title("FEM 1D 标量波教学原型：接收记录")
    _finish(outdir / "fem1d_receiver_trace.png", save, show, dpi)


def _finish(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
