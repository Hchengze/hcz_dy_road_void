"""高级数值方法教学/研究原型集合。

这些模块与默认道路空洞 workflow 解耦：它们用于学习 FEM、BEM、SEM、
FDTD 等路线的基本数值结构，不替代当前已经稳定工作的运动学正演、
绕射扫描和 elastic3d 小模型。
"""

from .bem import BEM2DResult, run_bem2d_scatter_demo
from .fdtd import FDTD1DResult, compare_fdtd_to_kinematic, describe_fdtd_route, run_fdtd1d_wave_demo
from .fem import FEM1DResult, run_fem1d_wave_demo
from .sem import SEM1DResult, run_sem1d_wave_demo
from .validation import check_array_finite, check_energy_not_exploding, compare_traces_l2, estimate_arrival_time

__all__ = [
    "BEM2DResult",
    "FEM1DResult",
    "FDTD1DResult",
    "SEM1DResult",
    "check_array_finite",
    "check_energy_not_exploding",
    "compare_fdtd_to_kinematic",
    "compare_traces_l2",
    "describe_fdtd_route",
    "estimate_arrival_time",
    "run_bem2d_scatter_demo",
    "run_fdtd1d_wave_demo",
    "run_fem1d_wave_demo",
    "run_sem1d_wave_demo",
]
