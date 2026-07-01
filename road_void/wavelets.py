"""运动学锤击正演模型使用的震源子波。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def ricker(frequency: float, dt: float, duration: float | None = None) -> tuple[FloatArray, FloatArray]:
    """返回零相位 Ricker 子波及其居中时间轴。"""

    if frequency <= 0 or dt <= 0:
        raise ValueError("frequency and dt must be positive.")
    if duration is None:
        duration = 6.0 / frequency
    n = int(np.ceil(duration / dt))
    if n % 2 == 0:
        n += 1
    t = (np.arange(n) - n // 2) * dt
    pf = np.pi * frequency * t
    w = (1.0 - 2.0 * pf**2) * np.exp(-(pf**2))
    return t, w.astype(float)


def hammer_pulse(frequency: float, dt: float, duration: float | None = None) -> tuple[FloatArray, FloatArray]:
    """返回近似锤击激发的因果短脉冲。"""

    if duration is None:
        duration = 4.0 / frequency
    n = max(3, int(np.ceil(duration / dt)))
    t = np.arange(n) * dt
    envelope = np.exp(-frequency * 3.5 * t)
    pulse = np.sin(2.0 * np.pi * frequency * t) * envelope
    pulse -= np.mean(pulse)
    peak = np.max(np.abs(pulse))
    if peak > 0:
        pulse /= peak
    centered_t = t - 0.35 / frequency
    return centered_t.astype(float), pulse.astype(float)
