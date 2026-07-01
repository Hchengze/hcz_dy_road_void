"""简化等效瑞雷波速度模型。

本模块仍然不是完整频散反演或弹性波正演。``layered-effective`` 模式
会根据主频估计面波敏感深度，并把层状模型折算成一个等效速度 ``VR_eff``，
用于当前运动学走时计算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VelocityLayer:
    """一个水平层状等效瑞雷波速度单元。"""

    top: float
    bottom: float
    velocity: float
    name: str = "layer"


@dataclass(frozen=True)
class LayeredRayleighVelocityModel:
    """水平层状等效瑞雷波速度模型。"""

    layers: tuple[VelocityLayer, ...]

    @classmethod
    def simple_road_model(cls) -> "LayeredRayleighVelocityModel":
        """返回一个便于教学展示的浅层道路分层速度模型。"""

        return cls(
            layers=(
                VelocityLayer(0.0, 0.5, 320.0, "路面结构层"),
                VelocityLayer(0.5, 2.0, 260.0, "基层/回填层"),
                VelocityLayer(2.0, 6.0, 220.0, "浅层土体"),
            )
        )

    @property
    def max_depth(self) -> float:
        """返回模型最大深度。"""

        return max(layer.bottom for layer in self.layers)

    def velocity_at_depth(self, z: FloatArray) -> FloatArray:
        """按深度返回等效瑞雷波速度。"""

        z = np.asarray(z, dtype=float)
        velocity = np.full_like(z, self.layers[-1].velocity, dtype=float)
        for layer in self.layers:
            mask = (z >= layer.top) & (z < layer.bottom)
            velocity[mask] = layer.velocity
        return velocity

    def section(self, x: FloatArray, z: FloatArray) -> FloatArray:
        """生成 x-z 剖面速度矩阵。"""

        depth_velocity = self.velocity_at_depth(z)
        return np.tile(depth_velocity[:, None], (1, len(x)))

    def effective_velocity(
        self,
        reference_velocity: float,
        source_frequency: float,
        sensitivity_depth_factor: float = 0.5,
    ) -> float:
        """计算层状模型对应的频带等效瑞雷波速度。

        近似关系为 ``lambda = VR / f``，敏感深度取
        ``z_sensitive = alpha * lambda``。随后用指数权重
        ``w(z)=exp(-z/z_sensitive)`` 对层速度做调和平均。调和平均对低速层
        更敏感，适合作为浅层面波走时的轻量近似。

        这一步的目的只是让层状模型影响直达波和绕射波走时；它不代表完整
        Rayleigh 频散曲线计算，也不代表弹性波全波形模拟。
        """

        if reference_velocity <= 0 or source_frequency <= 0:
            raise ValueError("reference_velocity 和 source_frequency 必须为正数。")
        alpha = max(float(sensitivity_depth_factor), 0.05)
        wavelength = reference_velocity / source_frequency
        z_sensitive = max(alpha * wavelength, 0.05)
        weights: list[float] = []
        velocities: list[float] = []
        for layer in self.layers:
            top = max(layer.top, 0.0)
            bottom = max(layer.bottom, top + 1e-6)
            # 对 exp(-z/z_sensitive) 在层厚内积分，避免用单点代表厚层。
            weight = z_sensitive * (np.exp(-top / z_sensitive) - np.exp(-bottom / z_sensitive))
            weights.append(float(max(weight, 1e-9)))
            velocities.append(float(layer.velocity))
        w = np.asarray(weights, dtype=float)
        v = np.asarray(velocities, dtype=float)
        return float(np.sum(w) / np.sum(w / v))
