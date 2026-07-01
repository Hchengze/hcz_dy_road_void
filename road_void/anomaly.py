"""浅层道路空洞合成实验中的异常体模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Cavity:
    """表示空洞、松散回填、管沟或破碎区的等效异常体。

    这里的参数是工程化有效参数，而不是严格弹性参数。``x0`` 和 ``y0``
    描述平面位置，``h`` 是三维走时模型使用的顶部/主控埋深，``radius``
    控制散射响应宽度和衰减影响范围。

    ``shape`` 不是完整真实边界散射解，而是把不同形状离散成少量等效
    散射点：sphere 近似圆形空洞，box 可表示井室/箱涵，cylinder 可表示
    管线或管沟，ellipsoid 表示椭球状脱空，line 表示长条状松散带。这样
    可以在当前运动学原型里研究形状、尺度和位置对绕射走时与能量的影响。
    """

    x0: float
    y0: float
    h: float
    # radius 不直接生成真实空洞边界，而是控制散射响应的空间影响范围。
    radius: float = 1.5
    scattering_strength: float = 0.45
    attenuation_strength: float = 0.25
    tail_strength: float = 1.0
    label: str = "cavity"
    shape: str = "sphere"
    size_x: float | None = None
    size_y: float | None = None
    size_z: float | None = None
    azimuth: float = 0.0

    @property
    def xyz(self) -> tuple[float, float, float]:
        """返回三维散射中心坐标。"""

        return (self.x0, self.y0, self.h)

    def scatter_points(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """返回等效散射点坐标和归一化权重。

        当前正演不是弹性波边界积分或 FDTD，因此这里不追求真实边界散射。
        每个点贡献一条 ``S-D-G`` 绕射路径，所有点叠加后形成更宽或更复杂
        的散射事件。权重和为 1，保证 ``scattering_strength`` 仍是异常体
        总体散射强度的直观控制量。
        """

        shape = self.shape.lower()
        center = np.asarray(self.xyz, dtype=float)
        if shape == "sphere":
            r = max(float(self.radius), 0.1)
            offsets = np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.45 * r, 0.0, 0.0],
                    [-0.45 * r, 0.0, 0.0],
                    [0.0, 0.45 * r, 0.0],
                    [0.0, -0.45 * r, 0.0],
                    [0.0, 0.0, 0.35 * r],
                    [0.0, 0.0, -0.25 * r],
                ]
            )
        elif shape == "box":
            sx = self.size_x or 2.0 * self.radius
            sy = self.size_y or 2.0 * self.radius
            sz = self.size_z or max(self.radius, 0.5)
            offsets = _box_offsets(sx, sy, sz)
        elif shape == "cylinder":
            r = self.size_x or self.radius
            height = self.size_z or max(self.size_y or self.radius, 0.5)
            offsets = _cylinder_offsets(r, height)
        elif shape == "ellipsoid":
            sx = self.size_x or 2.0 * self.radius
            sy = self.size_y or 1.4 * self.radius
            sz = self.size_z or self.radius
            offsets = _ellipsoid_offsets(sx, sy, sz)
        elif shape in {"line", "zone"}:
            length = self.size_x or max(3.0 * self.radius, 1.0)
            offsets = _line_offsets(length, self.azimuth)
        else:
            raise ValueError(f"未知异常体形状: {self.shape}")
        points = center[None, :] + offsets
        points[:, 2] = np.maximum(points[:, 2], 0.0)
        weights = np.ones(points.shape[0], dtype=float)
        weights /= np.sum(weights)
        return points, weights

    @property
    def effective_radius(self) -> float:
        """返回用于阴影和显示的近似影响半径。"""

        if self.shape.lower() in {"box", "ellipsoid"}:
            return 0.5 * max(self.size_x or self.radius, self.size_y or self.radius, self.size_z or self.radius)
        if self.shape.lower() in {"line", "zone"}:
            return 0.5 * (self.size_x or self.radius)
        return self.radius


def _box_offsets(size_x: float, size_y: float, size_z: float) -> NDArray[np.float64]:
    hx, hy, hz = 0.5 * size_x, 0.5 * size_y, 0.5 * size_z
    corners = [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return np.asarray([[0.0, 0.0, 0.0], *corners], dtype=float)


def _cylinder_offsets(radius: float, height: float) -> NDArray[np.float64]:
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    side = [[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles]
    top_bottom = [[radius * np.cos(a), radius * np.sin(a), dz] for dz in (-0.5 * height, 0.5 * height) for a in angles[::2]]
    return np.asarray([[0.0, 0.0, 0.0], *side, *top_bottom], dtype=float)


def _ellipsoid_offsets(size_x: float, size_y: float, size_z: float) -> NDArray[np.float64]:
    ax, ay, az = 0.5 * size_x, 0.5 * size_y, 0.5 * size_z
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [ax, 0.0, 0.0],
            [-ax, 0.0, 0.0],
            [0.0, ay, 0.0],
            [0.0, -ay, 0.0],
            [0.0, 0.0, az],
            [0.0, 0.0, -az],
        ],
        dtype=float,
    )


def _line_offsets(length: float, azimuth: float) -> NDArray[np.float64]:
    theta = np.deg2rad(azimuth)
    direction = np.asarray([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    samples = np.linspace(-0.5 * length, 0.5 * length, 7)
    return samples[:, None] * direction[None, :]
