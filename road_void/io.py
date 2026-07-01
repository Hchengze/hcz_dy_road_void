"""为后续真实道路 DAS 数据接入预留的接口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .geometry import RoadGeometry
from .processing import bandpass, trace_normalize


FloatArray = NDArray[np.float64]


@dataclass
class RealDASData:
    """真实或外部输入道路 DAS 数据容器。"""

    data: FloatArray
    dt: float
    channel_x: FloatArray
    channel_y: FloatArray
    shot_x: FloatArray | None = None
    shot_y: FloatArray | None = None
    trigger_times: FloatArray | None = None
    metadata: Mapping[str, object] | None = None


def load_numpy_das(path: str | Path, dt: float) -> FloatArray:
    """从 ``.npy`` 或 ``.npz`` 加载 DAS 数据。

    期望形状为 ``time x channel`` 或 ``shot x time x channel``。二维输入
    可以在补充炮点元数据后由下游流程提升为三维数据。
    """

    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        first_key = arr.files[0]
        data = np.asarray(arr[first_key], dtype=float)
    else:
        data = np.asarray(arr, dtype=float)
    if data.ndim not in (2, 3):
        raise ValueError("DAS data must be time x channel or shot x time x channel.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    return data


def geometry_from_real_metadata(
    road_width: float,
    channel_x: FloatArray,
    shot_x: FloatArray,
    dt: float,
    t_max: float,
    fiber_y: float = 0.0,
    shot_y: float | None = None,
) -> RoadGeometry:
    """将实测通道和锤击点位置映射到原型几何。"""

    return RoadGeometry(
        road_width=road_width,
        channel_x=np.asarray(channel_x, dtype=float),
        shot_x=np.asarray(shot_x, dtype=float),
        dt=dt,
        t_max=t_max,
        fiber_y=fiber_y,
        shot_y=road_width if shot_y is None else shot_y,
    )


def preprocess_real_das(
    data: FloatArray,
    dt: float,
    fmin: float | None = None,
    fmax: float | None = None,
    normalize: bool = True,
) -> FloatArray:
    """真实 DAS 数据的基础预处理占位接口。

    真实解释仍需完成通道位置标定、gauge length 检查、耦合质量控制、
    触发时间校正和场地特异性干扰核查。
    """

    out = np.asarray(data, dtype=float)
    if out.ndim == 2:
        # 真实 DAS 有时先以 time x channel 形式交付；这里先补一个 shot 维度，
        # 后续再结合锤击触发表拆分或映射到真实炮集。
        out = out[None, :, :]
    if fmin is not None and fmax is not None:
        out = bandpass(out, dt, fmin, fmax)
    if normalize:
        out = trace_normalize(out)
    return out
