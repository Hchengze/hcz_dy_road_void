"""浅层道路空洞合成实验中的异常体模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cavity:
    """表示空洞、松散回填、管沟或破碎区的紧凑散射体。

    这里的参数是工程化有效参数，而不是严格弹性参数。``x0`` 和 ``y0``
    描述平面位置，``h`` 是三维走时模型使用的顶部/主控埋深，``radius``
    控制散射响应宽度和衰减影响范围。
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

    @property
    def xyz(self) -> tuple[float, float, float]:
        """返回三维散射中心坐标。"""

        return (self.x0, self.y0, self.h)
