"""2D 标量 BEM/边界积分思想教学原型。

真实三维弹性半空间 BEM 很复杂，需要弹性 Green 张量、奇异积分处理、
自由表面和大规模稠密矩阵。本模块只做一个二维标量边界散射演示：

1. 离散圆形障碍边界；
2. 使用简化频域 Green 函数连接边界点；
3. 求一个软边界近似的等效边界源；
4. 计算接收线上的散射响应。

这不是完整声学 BEM，更不是三维弹性 BEM；它的价值是帮助理解“只离散边界”
和“边界点到接收点 Green 函数叠加”的思想。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .validation import check_array_finite, compare_traces_l2


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass
class BEM2DResult:
    """2D 标量 BEM 教学演示输出。"""

    boundary: FloatArray
    receivers: FloatArray
    incident_on_boundary: ComplexArray
    boundary_source: ComplexArray
    scattered_response: ComplexArray
    frequency: float
    velocity: float


def run_bem2d_scatter_demo(
    n_boundary: int = 64,
    n_receivers: int = 80,
    radius: float = 2.0,
    center: tuple[float, float] = (0.0, 0.0),
    source: tuple[float, float] = (-8.0, 0.0),
    receiver_x: tuple[float, float] = (-6.0, 8.0),
    receiver_y: float = 5.0,
    frequency: float = 35.0,
    velocity: float = 300.0,
    save: bool = False,
    show: bool = False,
    outdir: str | Path = "outputs/numerics",
    dpi: int = 180,
) -> BEM2DResult:
    """运行二维标量边界积分散射教学 demo。

    这里使用 ``G(r)=exp(i*k*r)/sqrt(r+eps)`` 作为简化 2D Green 函数代理。
    真正 2D Helmholtz Green 函数包含 Hankel 函数；本实现故意避免复杂依赖，
    只保留边界点矩阵和散射叠加的结构。
    """

    if n_boundary < 12 or n_receivers < 4:
        raise ValueError("BEM demo 需要足够的边界点和接收点。")
    if radius <= 0 or frequency <= 0 or velocity <= 0:
        raise ValueError("radius、frequency、velocity 必须为正。")
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    boundary = np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)))
    receivers = np.column_stack((np.linspace(receiver_x[0], receiver_x[1], n_receivers), np.full(n_receivers, receiver_y)))
    k = 2.0 * np.pi * frequency / velocity
    ds = 2.0 * np.pi * radius / n_boundary
    source_point = np.asarray(source, dtype=float)

    incident = _green(np.linalg.norm(boundary - source_point[None, :], axis=1), k)
    rbb = np.linalg.norm(boundary[:, None, :] - boundary[None, :, :], axis=2)
    # 对角项是 Green 函数奇异性的粗略正则化；教学原型不做奇异积分。
    gbb = _green(np.maximum(rbb, 0.5 * ds), k) * ds
    boundary_source = np.linalg.solve(gbb + 1.0e-3 * np.eye(n_boundary), -incident)
    grb = _green(np.linalg.norm(receivers[:, None, :] - boundary[None, :, :], axis=2), k) * ds
    scattered = grb @ boundary_source

    result = BEM2DResult(boundary, receivers, incident, boundary_source, scattered, frequency, velocity)
    check_array_finite(np.real(result.scattered_response), "bem scattered real")
    check_array_finite(np.imag(result.scattered_response), "bem scattered imag")
    _plot_bem_result(result, source_point, Path(outdir), save, show, dpi)
    return result


def _green(r: FloatArray, k: float) -> ComplexArray:
    return np.exp(1j * k * r) / np.sqrt(np.maximum(r, 1.0e-6))


def _plot_bem_result(result: BEM2DResult, source: FloatArray, outdir: Path, save: bool, show: bool, dpi: int) -> None:
    if not save and not show:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.8, 5.4))
    plt.plot(result.boundary[:, 0], result.boundary[:, 1], "o", ms=4, label="边界离散点")
    plt.plot(result.receivers[:, 0], result.receivers[:, 1], ".", ms=4, label="接收点")
    plt.plot(source[0], source[1], "*", ms=12, label="点源")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BEM 2D 标量边界积分思想演示：圆形边界离散")
    plt.legend()
    _finish(outdir / "bem2d_boundary_points.png", save, show, dpi)

    plt.figure(figsize=(8.5, 4.2))
    response = np.real(result.scattered_response)
    plt.plot(result.receivers[:, 0], response, label="Re(scattered)")
    plt.xlabel("接收点 x")
    plt.ylabel("散射响应")
    plt.title("BEM 2D 标量边界积分思想演示：接收线散射响应")
    plt.legend(title=f"L2={compare_traces_l2(response, np.zeros_like(response)):.2e}")
    _finish(outdir / "bem2d_scattered_response.png", save, show, dpi)


def _finish(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
