"""FDTD/FEM/SEM 统一 1D 标量波 benchmark。

本模块把三个低维教学原型放到同一个物理问题下：

    u_tt = c^2 u_xx + f

三种方法使用相同的长度、速度、震源、接收点、时间步长和记录时长。
它的目的不是证明哪种方法“更好”，而是给项目一个可复现的 sanity
check：主要到时应接近，记录不能爆炸，差异指标应为有限值。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .fdtd import FDTD1DResult, run_fdtd1d_wave_demo
from .fem import FEM1DResult, run_fem1d_wave_demo
from .sem import SEM1DResult, run_sem1d_wave_demo
from .validation import check_array_finite, compare_traces_l2, estimate_arrival_time


FloatArray = NDArray[np.float64]


@dataclass
class Wave1DCompareResult:
    """统一 benchmark 输出。"""

    fdtd: FDTD1DResult
    fem: FEM1DResult
    sem: SEM1DResult
    metrics: dict[str, float]


def compare_1d_wave_methods(
    length: float = 100.0,
    velocity: float = 300.0,
    duration: float = 0.24,
    dt: float = 0.0005,
    source_position: float = 25.0,
    receiver_position: float = 75.0,
    source_frequency: float = 35.0,
    outdir: str | Path = "outputs/numerics",
    save: bool = False,
    show: bool = False,
    dpi: int = 180,
) -> Wave1DCompareResult:
    """运行 FDTD/FEM/SEM 三种 1D 标量波算例并对比。

    参数单位均为 SI：长度 m、速度 m/s、时间 s、频率 Hz。固定端边界会在
    后期产生反射，因此 benchmark 主要关注首次到时和短时记录。
    """

    if not (0.0 < source_position < length and 0.0 < receiver_position < length):
        raise ValueError("source_position 和 receiver_position 必须位于 (0, length) 内。")
    if dt <= 0 or duration <= 0:
        raise ValueError("dt 和 duration 必须为正。")

    # 使用相近的空间分辨率：FDTD/FEM 为等距节点，SEM 用高阶谱元节点。
    fdtd = run_fdtd1d_wave_demo(
        n_points=201,
        length=length,
        velocity=velocity,
        duration=duration,
        dt=dt,
        source_position=source_position,
        receiver_position=receiver_position,
        source_frequency=source_frequency,
        save=False,
        show=False,
        outdir=outdir,
        dpi=dpi,
    )
    fem = run_fem1d_wave_demo(
        n_nodes=201,
        length=length,
        velocity=velocity,
        duration=duration,
        dt=dt,
        source_position=source_position,
        receiver_position=receiver_position,
        source_frequency=source_frequency,
        save=False,
        show=False,
        outdir=outdir,
        dpi=dpi,
    )
    sem = run_sem1d_wave_demo(
        n_elements=20,
        order=4,
        length=length,
        velocity=velocity,
        duration=duration,
        dt=dt,
        source_position=source_position,
        receiver_position=receiver_position,
        source_frequency=source_frequency,
        save=False,
        show=False,
        outdir=outdir,
        dpi=dpi,
    )

    traces = _align_traces(fdtd.receiver_trace, fem.receiver_trace, sem.receiver_trace)
    fdtd_trace, fem_trace, sem_trace = traces
    for name, trace in zip(("fdtd", "fem", "sem"), traces):
        check_array_finite(trace, f"{name} receiver trace")
    metrics = {
        "fdtd_arrival_time": estimate_arrival_time(fdtd_trace, dt),
        "fem_arrival_time": estimate_arrival_time(fem_trace, dt),
        "sem_arrival_time": estimate_arrival_time(sem_trace, dt),
        "fdtd_peak_amplitude": float(np.max(np.abs(fdtd_trace))),
        "fem_peak_amplitude": float(np.max(np.abs(fem_trace))),
        "sem_peak_amplitude": float(np.max(np.abs(sem_trace))),
        "fem_vs_fdtd_l2": compare_traces_l2(_normalize(fdtd_trace), _normalize(fem_trace)),
        "sem_vs_fdtd_l2": compare_traces_l2(_normalize(fdtd_trace), _normalize(sem_trace)),
        "theoretical_arrival_time": abs(receiver_position - source_position) / velocity + 1.5 / source_frequency,
    }
    if save or show:
        _plot_compare_outputs(fdtd, fem, sem, traces, metrics, Path(outdir), save, show, dpi)
    if save:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with (Path(outdir) / "compare_1d_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    return Wave1DCompareResult(fdtd=fdtd, fem=fem, sem=sem, metrics=metrics)


def _align_traces(*traces: FloatArray) -> tuple[FloatArray, ...]:
    n = min(len(trace) for trace in traces)
    return tuple(np.asarray(trace[:n], dtype=float) for trace in traces)


def _normalize(trace: FloatArray) -> FloatArray:
    scale = float(np.max(np.abs(trace)))
    if scale <= 0:
        return np.asarray(trace, dtype=float)
    return np.asarray(trace, dtype=float) / scale


def _plot_compare_outputs(
    fdtd: FDTD1DResult,
    fem: FEM1DResult,
    sem: SEM1DResult,
    traces: tuple[FloatArray, FloatArray, FloatArray],
    metrics: dict[str, float],
    outdir: Path,
    save: bool,
    show: bool,
    dpi: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(traces[0])
    time = fdtd.time[:n]
    plt.figure(figsize=(9.2, 4.6))
    plt.plot(time, _normalize(traces[0]), label=f"FDTD, t_arr={metrics['fdtd_arrival_time']:.4f}s")
    plt.plot(time, _normalize(traces[1]), label=f"FEM, t_arr={metrics['fem_arrival_time']:.4f}s")
    plt.plot(time, _normalize(traces[2]), label=f"SEM, t_arr={metrics['sem_arrival_time']:.4f}s")
    plt.axvline(metrics["theoretical_arrival_time"], color="k", ls="--", lw=1.0, label="理论主能量到时近似")
    plt.xlabel("时间 (s)")
    plt.ylabel("归一化接收振幅")
    plt.title("FDTD/FEM/SEM 1D 标量波 benchmark：接收记录对比")
    plt.legend()
    _finish(outdir / "compare_1d_traces.png", save, show, dpi)

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.2), sharex=True)
    for ax, name, x, snapshots in [
        (axes[0], "FDTD", fdtd.x, fdtd.snapshots),
        (axes[1], "FEM", fem.x, fem.snapshots),
        (axes[2], "SEM", sem.x, sem.snapshots),
    ]:
        order = np.argsort(x)
        for snap in snapshots:
            ax.plot(x[order], snap[order], lw=1.0)
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("x (m)")
    fig.suptitle("FDTD/FEM/SEM 1D 标量波 benchmark：波场快照对比")
    _finish(outdir / "compare_1d_wavefields.png", save, show, dpi)


def _finish(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
