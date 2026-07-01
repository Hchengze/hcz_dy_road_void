"""单侧 DAS 接收与对侧锤击激发的三维道路几何。

该几何刻意保留非共线观测关系：DAS 通道位于道路一侧 ``fiber_y``，
锤击点位于道路另一侧 ``shot_y``。即使震源和接收点深度为零，所有
走时计算也都使用三维坐标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoadGeometry:
    """三维等效瑞雷面波建模使用的道路、震源与接收几何。"""

    road_width: float
    channel_x: FloatArray
    shot_x: FloatArray
    dt: float
    t_max: float
    fiber_y: float = 0.0
    shot_y: float | None = None
    receiver_z: float = 0.0
    source_z: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_x", np.asarray(self.channel_x, dtype=float))
        object.__setattr__(self, "shot_x", np.asarray(self.shot_x, dtype=float))
        if self.shot_y is None:
            object.__setattr__(self, "shot_y", float(self.road_width))
        if self.road_width <= 0:
            raise ValueError("road_width must be positive.")
        if self.dt <= 0 or self.t_max <= 0:
            raise ValueError("dt and t_max must be positive.")
        if self.channel_x.ndim != 1 or self.shot_x.ndim != 1:
            raise ValueError("channel_x and shot_x must be one-dimensional.")

    @classmethod
    def typical_four_lane(
        cls,
        x_min: float = 0.0,
        x_max: float = 80.0,
        channel_spacing: float = 1.0,
        shot_spacing: float = 4.0,
        dt: float = 0.001,
        t_max: float = 1.0,
    ) -> "RoadGeometry":
        """创建典型四车道道路几何，横向孔径约 15 m。"""

        return cls.from_ranges(
            road_width=15.0,
            x_min=x_min,
            x_max=x_max,
            channel_spacing=channel_spacing,
            shot_spacing=shot_spacing,
            dt=dt,
            t_max=t_max,
        )

    @classmethod
    def typical_six_lane(
        cls,
        x_min: float = 0.0,
        x_max: float = 100.0,
        channel_spacing: float = 1.0,
        shot_spacing: float = 5.0,
        dt: float = 0.001,
        t_max: float = 1.2,
    ) -> "RoadGeometry":
        """创建典型六车道道路几何，横向孔径约 28 m。"""

        return cls.from_ranges(
            road_width=28.0,
            x_min=x_min,
            x_max=x_max,
            channel_spacing=channel_spacing,
            shot_spacing=shot_spacing,
            dt=dt,
            t_max=t_max,
        )

    @classmethod
    def from_ranges(
        cls,
        road_width: float,
        x_min: float,
        x_max: float,
        channel_spacing: float,
        shot_spacing: float,
        dt: float,
        t_max: float,
        fiber_y: float = 0.0,
        shot_y: float | None = None,
    ) -> "RoadGeometry":
        """根据 x 范围生成规则震源线和接收线。"""

        if channel_spacing <= 0 or shot_spacing <= 0:
            raise ValueError("channel_spacing and shot_spacing must be positive.")
        channel_x = np.arange(x_min, x_max + 0.5 * channel_spacing, channel_spacing)
        shot_x = np.arange(x_min, x_max + 0.5 * shot_spacing, shot_spacing)
        return cls(
            road_width=road_width,
            channel_x=channel_x,
            shot_x=shot_x,
            dt=dt,
            t_max=t_max,
            fiber_y=fiber_y,
            shot_y=road_width if shot_y is None else shot_y,
        )

    @property
    def n_channels(self) -> int:
        return int(self.channel_x.size)

    @property
    def n_shots(self) -> int:
        return int(self.shot_x.size)

    @property
    def n_times(self) -> int:
        return int(np.floor(self.t_max / self.dt)) + 1

    @property
    def time_axis(self) -> FloatArray:
        return np.arange(self.n_times, dtype=float) * self.dt

    @property
    def channel_xyz(self) -> FloatArray:
        return np.column_stack(
            [
                self.channel_x,
                np.full(self.n_channels, self.fiber_y, dtype=float),
                np.full(self.n_channels, self.receiver_z, dtype=float),
            ]
        )

    @property
    def shot_xyz(self) -> FloatArray:
        return np.column_stack(
            [
                self.shot_x,
                np.full(self.n_shots, float(self.shot_y), dtype=float),
                np.full(self.n_shots, self.source_z, dtype=float),
            ]
        )

    @staticmethod
    def distance(a: ArrayLike, b: ArrayLike) -> FloatArray | float:
        """返回两个三维点或点数组之间的欧氏距离。"""

        arr_a = np.asarray(a, dtype=float)
        arr_b = np.asarray(b, dtype=float)
        return np.linalg.norm(arr_a - arr_b, axis=-1)

    def source_receiver_distances(self) -> FloatArray:
        """返回三维源检距矩阵，形状为 shots x channels。"""

        # 这里保留 y 方向的道路宽度，不把震源和接收点投影到同一条 x 测线。
        src = self.shot_xyz[:, None, :]
        rec = self.channel_xyz[None, :, :]
        return np.linalg.norm(src - rec, axis=-1)

    def direct_times(self, velocity: float, t0: float = 0.0) -> FloatArray:
        """计算所有炮点和通道的直达等效瑞雷波到时。"""

        if velocity <= 0:
            raise ValueError("velocity must be positive.")
        return t0 + self.source_receiver_distances() / velocity

    def diffraction_times(
        self,
        scatterer_xyz: Iterable[float],
        velocity: float,
        t0: float = 0.0,
    ) -> FloatArray:
        """计算所有炮点和通道的源-散射体-接收点绕射走时。"""

        if velocity <= 0:
            raise ValueError("velocity must be positive.")
        point = np.asarray(tuple(scatterer_xyz), dtype=float)
        if point.shape != (3,):
            raise ValueError("scatterer_xyz must contain exactly three coordinates.")
        # 绕射路径由两段组成：震源到异常体、异常体到 DAS 通道。
        # 这正是单侧 DAS + 对侧锤击几何区别于二维 MASW 的关键。
        sd = np.linalg.norm(self.shot_xyz - point[None, :], axis=-1)[:, None]
        dg = np.linalg.norm(self.channel_xyz - point[None, :], axis=-1)[None, :]
        return t0 + (sd + dg) / velocity
