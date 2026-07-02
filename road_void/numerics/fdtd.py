"""FDTD 路线说明与轻量对比辅助。

当前项目中真正的 FDTD 原型仍在 ``road_void.elastic3d``。本文件不复制
elastic3d，只提供文字说明和小型摘要函数，便于把 FDTD 与 FEM/BEM/SEM
放在同一 numerics 目录下理解。
"""

from __future__ import annotations


def describe_fdtd_route() -> str:
    """返回 FDTD 路线说明。"""

    return (
        "FDTD/finite-difference 路线使用规则网格和差分模板更新波场。"
        "本项目的 road_void.elastic3d 属于小尺度三维 velocity-stress FDTD 原型，"
        "适合教学和规则网格波场 sanity check；复杂曲面边界会有 stair-step 近似，"
        "后续可继续发展严格 staggered-grid、完整 CPML 和 DAS 应变响应。"
    )


def compare_fdtd_to_kinematic() -> dict[str, str]:
    """返回 FDTD 与运动学正演的概念对比。"""

    return {
        "kinematic": "直接用 t_direct/t_diff 构造事件，速度快，适合扫描定位和参数敏感性。",
        "fdtd": "显式更新波场变量，更接近全波场，但计算量更大，默认不替代 workflow。",
        "relationship": "两者是不同层级工具：运动学模型做快速验证，FDTD 做小模型物理 sanity check。",
    }
