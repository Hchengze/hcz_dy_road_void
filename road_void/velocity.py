"""简化等效瑞雷波速度模型。

当前正演只使用单一等效速度 ``VR``。这里的速度模型主要用于教学展示：
说明“背景模型长什么样”、速度参数如何进入正演，以及未来如何扩展到
分层或频散模型。
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
    """用于展示的水平层状等效瑞雷波速度模型。"""

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
