"""FWI 最小原型：只实现 L2 misfit 曲线，不实现完整伴随梯度。

完整三维弹性 FWI 需要严格正演、伴随方程、梯度构造、步长搜索和模型更新。
本模块只用于教学演示：改变一个标量参数 ``vs_scale``，重复运行小尺度
``elastic3d`` 正演，计算 ``J(m)=0.5||d_cal-d_obs||^2``。因此它应被称为
``fwi-demo`` 或误差函数实验，而不是完整 FWI。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .elastic3d import Elastic3DConfig, Elastic3DResult, run_elastic3d


FloatArray = NDArray[np.float64]


@dataclass
class FWIDemoResult:
    """FWI-demo 输出。"""

    vs_scales: FloatArray
    misfits: FloatArray
    observed: Elastic3DResult
    initial: Elastic3DResult
    best_vs_scale: float


def run_fwi_misfit_demo(
    vs_scales: tuple[float, ...] = (0.86, 0.92, 0.98, 1.0, 1.04),
    observed_vs_scale: float = 1.0,
    initial_vs_scale: float = 0.9,
    config: Elastic3DConfig | None = None,
) -> FWIDemoResult:
    """运行一维 ``Vs`` 缩放误差函数演示。

    这里没有伴随梯度，也没有模型更新；只是用多个候选 ``vs_scale`` 生成
    synthetic gather，与目标 gather 做 L2 misfit 对比。这个曲线可以帮助
    理解 FWI 的“数据残差驱动模型更新”思想。
    """

    base = config or Elastic3DConfig(nx=30, ny=22, nz=16, nt=90, record_component="strain_rate_xx")
    observed = run_elastic3d(_replace_config(base, vs_scale=observed_vs_scale))
    initial = run_elastic3d(_replace_config(base, vs_scale=initial_vs_scale))
    misfits = []
    for scale in vs_scales:
        synthetic = run_elastic3d(_replace_config(base, vs_scale=scale))
        residual = synthetic.gather - observed.gather
        misfits.append(0.5 * float(np.sum(residual**2)))
    misfit_array = np.asarray(misfits, dtype=float)
    best = float(vs_scales[int(np.argmin(misfit_array))])
    return FWIDemoResult(np.asarray(vs_scales, dtype=float), misfit_array, observed, initial, best)


def plot_fwi_demo_outputs(
    result: FWIDemoResult,
    outdir: str | Path,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制 misfit 曲线和目标/初始合成记录对比。"""

    outdir = Path(outdir)
    _plot_misfit_curve(result, outdir / "misfit_curve.png", save, show, dpi)
    _plot_observed_vs_initial(result, outdir / "observed_vs_synthetic_gather.png", save, show, dpi)


def _replace_config(config: Elastic3DConfig, **updates: object) -> Elastic3DConfig:
    data = config.__dict__.copy()
    data.update(updates)
    return Elastic3DConfig(**data)


def _plot_misfit_curve(result: FWIDemoResult, output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(result.vs_scales, result.misfits, "o-", lw=1.8)
    plt.axvline(result.best_vs_scale, color="tab:red", ls="--", label=f"最小 misfit: Vs scale={result.best_vs_scale:.2f}")
    plt.xlabel("Vs 缩放因子")
    plt.ylabel("L2 misfit")
    plt.title("FWI 最小原型：一维 Vs 缩放误差函数曲线（非完整伴随 FWI）")
    plt.legend()
    _finish(output, save, show, dpi)


def _plot_observed_vs_initial(result: FWIDemoResult, output: Path, save: bool, show: bool, dpi: int) -> None:
    obs = result.observed.gather
    ini = result.initial.gather
    vmax = max(float(np.max(np.abs(obs))), float(np.max(np.abs(ini))), 1e-12)
    extent = [
        float(result.observed.receiver_x[0]),
        float(result.observed.receiver_x[-1]),
        result.observed.config.nt * result.observed.config.dt,
        0.0,
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True, constrained_layout=True)
    images = [
        (obs, "目标数据 observed"),
        (ini, "初始模型 synthetic"),
        (ini - obs, "残差 synthetic-observed"),
    ]
    for ax, (data, title) in zip(axes, images):
        im = ax.imshow(data, origin="upper", aspect="auto", extent=extent, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("接收线 x (m)")
    axes[0].set_ylabel("时间 (s)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, label=result.observed.record_component)
    fig.suptitle("FWI 最小原型：目标记录、初始合成记录与残差")
    _finish(output, save, show, dpi)


def _finish(output: Path, save: bool, show: bool, dpi: int) -> None:
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
