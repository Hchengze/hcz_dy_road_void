"""数值方法共用 sanity check 工具。

这些函数刻意保持简单，服务于教学原型：检查数组是否有限、能量是否
明显爆炸、估计到时、比较两条记录。它们不是严格数值分析证明。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def check_array_finite(array: FloatArray, name: str = "array") -> None:
    """检查数组没有 NaN/inf。

    参数
    ----
    array:
        任意数值数组。
    name:
        报错时显示的物理量名称。
    """

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 包含 NaN 或 inf。")


def check_energy_not_exploding(signal: FloatArray, growth_limit: float = 1.0e8, name: str = "signal") -> None:
    """检查能量没有非物理爆炸。

    这里用 ``sum(signal**2)`` 作为能量代理。教学算例中震源会注入能量，
    因此不能要求能量单调衰减；但若能量超过 ``growth_limit``，通常说明
    CFL、边界或源幅值出了问题。
    """

    check_array_finite(signal, name)
    energy = float(np.sum(np.asarray(signal, dtype=float) ** 2))
    if energy > growth_limit:
        raise ValueError(f"{name} 能量过大: {energy:.3e} > {growth_limit:.3e}，疑似数值爆炸。")


def estimate_arrival_time(trace: FloatArray, dt: float, threshold_fraction: float = 0.2) -> float:
    """用阈值法估计一条记录的首次到时。

    ``threshold_fraction`` 是相对最大振幅的比例。若记录全零，返回 ``nan``。
    """

    check_array_finite(trace, "trace")
    amp = np.abs(trace)
    peak = float(np.max(amp))
    if peak <= 0.0:
        return float("nan")
    idx = int(np.argmax(amp >= threshold_fraction * peak))
    return idx * dt


def compare_traces_l2(trace_a: FloatArray, trace_b: FloatArray) -> float:
    """返回两条记录的 L2 差异。"""

    a = np.asarray(trace_a, dtype=float)
    b = np.asarray(trace_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"trace shape 不一致: {a.shape} != {b.shape}")
    check_array_finite(a, "trace_a")
    check_array_finite(b, "trace_b")
    return float(np.sqrt(np.sum((a - b) ** 2)))
