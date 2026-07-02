"""1D 谱元/谱元素 SEM 教学原型。

SEM 可以看成高阶 FEM：单元内部使用较高阶 Lagrange 多项式，并常把节点放在
Gauss-Lobatto-Legendre (GLL/LGL) 点上。使用同一组 GLL 点做插值和积分时，
质量矩阵会接近对角形式，适合显式波动方程推进。

本文件只做 1D 标量波传播演示，不是 SPECFEM3D，也不是三维弹性 SEM。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import legendre
from numpy.typing import NDArray

from .validation import check_array_finite, check_energy_not_exploding


FloatArray = NDArray[np.float64]


@dataclass
class SEM1DResult:
    """1D SEM 波动算例输出。"""

    x: FloatArray
    time: FloatArray
    gll_nodes: FloatArray
    mass_diag: FloatArray
    stiffness: FloatArray
    snapshots: FloatArray
    receiver_trace: FloatArray
    source_index: int
    receiver_index: int


def gll_nodes_weights(order: int) -> tuple[FloatArray, FloatArray]:
    """生成 [-1, 1] 上的 GLL 节点和权重。

    GLL 节点包括端点和 Legendre 多项式导数的根。权重公式：
    ``w_i = 2 / [N(N+1) P_N(x_i)^2]``。
    """

    if order < 2:
        raise ValueError("SEM 教学原型要求 order >= 2。")
    coeff = np.zeros(order + 1)
    coeff[-1] = 1.0
    dcoeff = legendre.legder(coeff)
    interior = np.sort(legendre.legroots(dcoeff).real)
    nodes = np.concatenate(([-1.0], interior, [1.0])).astype(float)
    pn = legendre.legval(nodes, coeff)
    weights = 2.0 / (order * (order + 1) * pn**2)
    return nodes, weights.astype(float)


def lagrange_derivative_matrix(nodes: FloatArray) -> FloatArray:
    """构造 Lagrange 基函数在节点上的导数矩阵 D_ij = l'_j(x_i)。"""

    n = nodes.size
    bary = np.ones(n, dtype=float)
    for j in range(n):
        for k in range(n):
            if j != k:
                bary[j] /= nodes[j] - nodes[k]
    dmat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                dmat[i, j] = bary[j] / bary[i] / (nodes[i] - nodes[j])
        dmat[i, i] = -np.sum(dmat[i])
    return dmat


def run_sem1d_wave_demo(
    n_elements: int = 8,
    order: int = 4,
    length: float = 100.0,
    velocity: float = 300.0,
    duration: float = 0.24,
    dt: float | None = None,
    source_frequency: float = 35.0,
    save: bool = False,
    show: bool = False,
    outdir: str | Path = "outputs/numerics",
    dpi: int = 180,
) -> SEM1DResult:
    """运行 1D 标量波 SEM 教学算例。"""

    if n_elements < 2:
        raise ValueError("SEM 至少需要两个谱元。")
    xi, w = gll_nodes_weights(order)
    dxi = lagrange_derivative_matrix(xi)
    n_local = order + 1
    n_global = n_elements * order + 1
    x = np.zeros(n_global, dtype=float)
    mass_diag = np.zeros(n_global, dtype=float)
    stiffness = np.zeros((n_global, n_global), dtype=float)
    elem_len = length / n_elements
    ke_ref = dxi.T @ np.diag(w) @ dxi

    for e in range(n_elements):
        x_left = e * elem_len
        local_x = x_left + 0.5 * elem_len * (xi + 1.0)
        ids = np.arange(e * order, e * order + n_local)
        x[ids] = local_x
        mass_diag[ids] += 0.5 * elem_len * w
        stiffness[np.ix_(ids, ids)] += (2.0 / elem_len) * velocity**2 * ke_ref

    dx_min = float(np.min(np.diff(np.unique(x))))
    dt = float(dt or 0.18 * dx_min / velocity)
    n_steps = max(8, int(duration / dt) + 1)
    time = np.arange(n_steps, dtype=float) * dt
    source_index = int(np.argmin(np.abs(x - 0.25 * length)))
    receiver_index = int(np.argmin(np.abs(x - 0.75 * length)))
    u_prev = np.zeros(n_global, dtype=float)
    u = np.zeros(n_global, dtype=float)
    receiver_trace = np.zeros(n_steps, dtype=float)
    snapshot_ids = np.linspace(0, n_steps - 1, 5, dtype=int)
    snapshots = []
    source = _ricker(time, source_frequency)

    for it in range(n_steps):
        force = np.zeros(n_global, dtype=float)
        force[source_index] = source[it]
        accel = (force - stiffness @ u) / mass_diag
        u_next = 2.0 * u - u_prev + dt**2 * accel
        u_next[0] = 0.0
        u_next[-1] = 0.0
        receiver_trace[it] = u[receiver_index]
        if it in snapshot_ids:
            snapshots.append(u.copy())
        u_prev, u = u, u_next

    result = SEM1DResult(x, time, xi, mass_diag, stiffness, np.asarray(snapshots), receiver_trace, source_index, receiver_index)
    check_array_finite(result.snapshots, "sem snapshots")
    check_energy_not_exploding(result.receiver_trace, growth_limit=1.0e6, name="sem receiver trace")
    _plot_sem_result(result, Path(outdir), save, show, dpi)
    return result


def _ricker(time: FloatArray, frequency: float) -> FloatArray:
    t0 = 1.5 / frequency
    arg = np.pi * frequency * (time - t0)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


def _plot_sem_result(result: SEM1DResult, outdir: Path, save: bool, show: bool, dpi: int) -> None:
    if not save and not show:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.8))
    for snap in result.snapshots:
        order = np.argsort(result.x)
        plt.plot(result.x[order], snap[order], lw=1.2)
    plt.scatter(result.x, np.zeros_like(result.x), s=8, c="0.45", label="SEM GLL 全局节点")
    plt.axvline(result.x[result.source_index], color="tab:red", ls="--", label="点源")
    plt.axvline(result.x[result.receiver_index], color="tab:green", ls=":", label="接收点")
    plt.xlabel("x (m)")
    plt.ylabel("u")
    plt.title("SEM 1D 标量波教学原型：波场快照")
    plt.legend()
    _finish(outdir / "sem1d_wavefield.png", save, show, dpi)

    plt.figure(figsize=(8.5, 4.2))
    plt.plot(result.time, result.receiver_trace)
    plt.xlabel("时间 (s)")
    plt.ylabel("u(receiver)")
    plt.title("SEM 1D 标量波教学原型：接收记录")
    _finish(outdir / "sem1d_receiver_trace.png", save, show, dpi)


def _finish(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
