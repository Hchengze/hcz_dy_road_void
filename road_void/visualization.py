"""合成记录、道路几何、速度模型与定位结果的可视化工具。

本模块的图件主要服务于“理解和展示”。其中波场动画是与当前
三维运动学正演一致的等效传播示意，不代表严格弹性波场快照。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import animation, font_manager
import numpy as np
from numpy.typing import NDArray

from .anomaly import Cavity
from .geometry import RoadGeometry
from .scan import CavityScanResult
from .velocity import LayeredRayleighVelocityModel


FloatArray = NDArray[np.float64]


def _shape_marker(cavity: Cavity) -> str:
    return {
        "sphere": "o",
        "box": "s",
        "cylinder": "^",
        "ellipsoid": "D",
        "line": "x",
        "zone": "x",
    }.get(cavity.shape.lower(), "o")


def _rotated_rectangle(cx: float, cy: float, length: float, width: float, azimuth: float) -> NDArray[np.float64]:
    theta = np.deg2rad(azimuth)
    along = np.asarray([np.cos(theta), np.sin(theta)])
    cross = np.asarray([-np.sin(theta), np.cos(theta)])
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        p = np.asarray([cx, cy]) + sx * 0.5 * length * along + sy * 0.5 * width * cross
        corners.append(p)
    return np.asarray(corners, dtype=float)


def _draw_anomaly_plan(ax: plt.Axes, cavity: Cavity, label: str | None = None) -> None:
    """在 x-y 平面画出不同异常体的简化几何轮廓。"""

    shape = cavity.shape.lower()
    color = "darkorange"
    if shape == "sphere":
        ax.add_patch(patches.Circle((cavity.x0, cavity.y0), cavity.radius, fill=False, ec=color, lw=2, label=label))
    elif shape == "box":
        sx = cavity.size_x or 2 * cavity.radius
        sy = cavity.size_y or 2 * cavity.radius
        ax.add_patch(patches.Rectangle((cavity.x0 - sx / 2, cavity.y0 - sy / 2), sx, sy, fill=False, ec=color, lw=2, label=label))
    elif shape == "cylinder":
        r = cavity.size_x or cavity.radius
        ax.add_patch(patches.Circle((cavity.x0, cavity.y0), r, fill=False, ec=color, lw=2, label=label))
        ax.plot([cavity.x0], [cavity.y0], marker="+", color=color, ms=10)
    elif shape == "ellipsoid":
        sx = cavity.size_x or 2 * cavity.radius
        sy = cavity.size_y or 1.4 * cavity.radius
        ax.add_patch(patches.Ellipse((cavity.x0, cavity.y0), sx, sy, fill=False, ec=color, lw=2, label=label))
    elif shape == "line":
        length = cavity.size_x or 3 * cavity.radius
        pts = _rotated_rectangle(cavity.x0, cavity.y0, length, 0.05 * max(length, 1.0), cavity.azimuth)
        ax.plot(pts[[0, 1], 0], pts[[0, 1], 1], color=color, lw=3, label=label)
    elif shape == "zone":
        length = cavity.size_x or 3 * cavity.radius
        width = cavity.size_y or 1.5 * cavity.radius
        pts = _rotated_rectangle(cavity.x0, cavity.y0, length, width, cavity.azimuth)
        ax.add_patch(patches.Polygon(pts, fill=False, ec=color, lw=2, label=label))
    points, _ = cavity.scatter_points()
    ax.scatter(points[:, 0], points[:, 1], s=12, c=color, alpha=0.55)


def _draw_anomaly_xz(ax: plt.Axes, cavity: Cavity) -> None:
    shape = cavity.shape.lower()
    color = "darkorange"
    if shape in {"sphere", "cylinder"}:
        width = 2 * (cavity.size_x or cavity.radius)
        height = cavity.size_z or (2 * cavity.radius if shape == "sphere" else 2 * cavity.radius)
        ax.add_patch(patches.Ellipse((cavity.x0, cavity.h), width, height, fill=False, ec=color, lw=2))
    elif shape == "box":
        sx = cavity.size_x or 2 * cavity.radius
        sz = cavity.size_z or cavity.radius
        ax.add_patch(patches.Rectangle((cavity.x0 - sx / 2, cavity.h - sz / 2), sx, sz, fill=False, ec=color, lw=2))
    elif shape == "ellipsoid":
        sx = cavity.size_x or 2 * cavity.radius
        sz = cavity.size_z or cavity.radius
        ax.add_patch(patches.Ellipse((cavity.x0, cavity.h), sx, sz, fill=False, ec=color, lw=2))
    elif shape in {"line", "zone"}:
        length = abs(np.cos(np.deg2rad(cavity.azimuth))) * (cavity.size_x or 3 * cavity.radius)
        height = cavity.size_z or 0.25
        ax.add_patch(patches.Rectangle((cavity.x0 - length / 2, cavity.h - height / 2), max(length, 0.2), height, fill=False, ec=color, lw=2))
    points, _ = cavity.scatter_points()
    ax.scatter(points[:, 0], points[:, 2], s=12, c=color, alpha=0.55)


def _draw_anomaly_yz(ax: plt.Axes, cavity: Cavity) -> None:
    shape = cavity.shape.lower()
    color = "darkorange"
    if shape in {"sphere", "cylinder"}:
        width = 2 * (cavity.size_x or cavity.radius)
        height = cavity.size_z or (2 * cavity.radius if shape == "sphere" else 2 * cavity.radius)
        ax.add_patch(patches.Ellipse((cavity.y0, cavity.h), width, height, fill=False, ec=color, lw=2))
    elif shape == "box":
        sy = cavity.size_y or 2 * cavity.radius
        sz = cavity.size_z or cavity.radius
        ax.add_patch(patches.Rectangle((cavity.y0 - sy / 2, cavity.h - sz / 2), sy, sz, fill=False, ec=color, lw=2))
    elif shape == "ellipsoid":
        sy = cavity.size_y or 1.4 * cavity.radius
        sz = cavity.size_z or cavity.radius
        ax.add_patch(patches.Ellipse((cavity.y0, cavity.h), sy, sz, fill=False, ec=color, lw=2))
    elif shape in {"line", "zone"}:
        length = abs(np.sin(np.deg2rad(cavity.azimuth))) * (cavity.size_x or 3 * cavity.radius)
        width = cavity.size_y if shape == "zone" else 0.2
        ax.add_patch(patches.Rectangle((cavity.y0 - max(length, width or 0.2) / 2, cavity.h - 0.12), max(length, width or 0.2), 0.24, fill=False, ec=color, lw=2))
    points, _ = cavity.scatter_points()
    ax.scatter(points[:, 1], points[:, 2], s=12, c=color, alpha=0.55)


def _configure_chinese_font() -> None:
    """为 matplotlib 配置常见中文字体，避免中文图题显示为方框。"""

    font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    font_names: list[str] = []
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_names.append(font_manager.FontProperties(fname=str(font_path)).get_name())
    if font_names:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = font_names + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


def _finish_figure(
    output: str | Path | None,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """保存和/或交互显示当前图件。

    ``save`` 和 ``show`` 可以同时为 True：先保存文件，再弹出 matplotlib
    交互窗口。若二者都为 False，则只关闭图件，适合自动测试或快速跑流程。
    """

    plt.tight_layout()
    if save and output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def plot_shot_gather(
    data: FloatArray,
    geometry: RoadGeometry,
    shot_index: int,
    direct_times: FloatArray | None = None,
    diffraction_times: FloatArray | None = None,
    title: str | None = None,
    output: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制单炮记录，并可叠加三维直达波和绕射波理论走时曲线。"""

    gather = data[shot_index]
    clip = np.percentile(np.abs(gather), 98.5)
    extent = [
        float(geometry.channel_x[0]),
        float(geometry.channel_x[-1]),
        float(geometry.time_axis[-1]),
        float(geometry.time_axis[0]),
    ]
    plt.figure(figsize=(10, 5.5))
    plt.imshow(gather, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip, extent=extent)
    plt.colorbar(label="振幅")
    if direct_times is not None:
        plt.plot(geometry.channel_x, direct_times[shot_index], "k-", lw=1.4, label="三维直达瑞雷波")
    if diffraction_times is not None:
        plt.plot(geometry.channel_x, diffraction_times[shot_index], "y--", lw=1.4, label="三维空洞绕射")
    plt.xlabel("DAS 通道 x (m)")
    plt.ylabel("时间 (s)")
    plt.title(title or f"第 {shot_index} 炮记录")
    if direct_times is not None or diffraction_times is not None:
        plt.legend(loc="upper right")
    _finish_figure(output, save=save, show=show, dpi=dpi)


def plot_score_slices(
    result: CavityScanResult,
    true_x: float | None = None,
    true_y: float | None = None,
    true_h: float | None = None,
    output: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制 x-y、x-h、y-h 三个最大评分切片，并标出 top-k 候选点。"""

    scores = result.scores
    xy = np.max(scores, axis=(2, 3))
    xh = np.max(scores, axis=(1, 3))
    yh = np.max(scores, axis=(0, 3))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    im0 = axes[0].imshow(
        xy.T,
        origin="lower",
        aspect="auto",
        extent=[result.grid.x[0], result.grid.x[-1], result.grid.y[0], result.grid.y[-1]],
        cmap="viridis",
    )
    axes[0].set_xlabel("x0 (m)")
    axes[0].set_ylabel("y0 (m)")
    axes[0].set_title("x-y 最大评分")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        xh.T,
        origin="lower",
        aspect="auto",
        extent=[result.grid.x[0], result.grid.x[-1], result.grid.h[0], result.grid.h[-1]],
        cmap="magma",
    )
    axes[1].set_xlabel("x0 (m)")
    axes[1].set_ylabel("h/顶部埋深 (m)")
    axes[1].set_title("x-h 最大评分")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        yh.T,
        origin="lower",
        aspect="auto",
        extent=[result.grid.y[0], result.grid.y[-1], result.grid.h[0], result.grid.h[-1]],
        cmap="cividis",
    )
    axes[2].set_xlabel("y0 (m)")
    axes[2].set_ylabel("h/顶部埋深 (m)")
    axes[2].set_title("y-h 最大评分")
    fig.colorbar(im2, ax=axes[2])

    for candidate in result.top_candidates:
        axes[0].plot(candidate.x0, candidate.y0, "w.", ms=5, alpha=0.75)
        axes[1].plot(candidate.x0, candidate.h, "w.", ms=5, alpha=0.75)
        axes[2].plot(candidate.y0, candidate.h, "w.", ms=5, alpha=0.75)
    axes[0].plot(result.best.x0, result.best.y0, "r+", ms=12, mew=2, label="最佳")
    axes[1].plot(result.best.x0, result.best.h, "c+", ms=12, mew=2, label="最佳")
    axes[2].plot(result.best.y0, result.best.h, "r+", ms=12, mew=2, label="最佳")
    if true_x is not None and true_y is not None:
        axes[0].plot(true_x, true_y, "wo", mec="k", label="真值")
    if true_x is not None and true_h is not None:
        axes[1].plot(true_x, true_h, "wo", mec="k", label="真值")
    if true_y is not None and true_h is not None:
        axes[2].plot(true_y, true_h, "wo", mec="k", label="真值")
    for ax in axes:
        ax.legend(loc="upper right")
    _finish_figure(output, save=save, show=show, dpi=dpi)


def plot_multishot_scan_diagnostics(
    result: CavityScanResult,
    output_dir: str | Path,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制单炮与多炮联合扫描的简化诊断图。

    这些图不改变定位算法，只帮助观察：哪些炮对联合评分贡献较大、单炮
    最佳 x 是否离散，以及联合结果是否比单炮更稳定。
    """

    if result.per_shot_best is None or result.per_shot_score_contribution is None:
        return
    output_dir = Path(output_dir)
    shot_id = np.arange(len(result.per_shot_best))
    best_x = np.asarray([c.x0 for c in result.per_shot_best], dtype=float)
    best_y = np.asarray([c.y0 for c in result.per_shot_best], dtype=float)
    best_h = np.asarray([c.h for c in result.per_shot_best], dtype=float)
    contribution = np.asarray(result.per_shot_score_contribution, dtype=float)

    plt.figure(figsize=(8.5, 4.6))
    plt.plot(shot_id, best_x, "o-", label="单炮最佳 x")
    plt.axhline(result.best.x0, color="r", lw=2, label="多炮联合最佳 x")
    plt.xlabel("炮号")
    plt.ylabel("x0 (m)")
    plt.title("单炮最佳 x 与多炮联合结果对比")
    plt.legend()
    _finish_figure(output_dir / "per_shot_best_x.png", save=save, show=show, dpi=dpi)

    plt.figure(figsize=(8.5, 4.6))
    plt.bar(shot_id, contribution, color="steelblue", alpha=0.85)
    plt.xlabel("炮号")
    plt.ylabel("最佳候选处单炮评分贡献")
    plt.title("多炮联合评分的单炮贡献")
    _finish_figure(output_dir / "per_shot_score_contribution.png", save=save, show=show, dpi=dpi)

    plt.figure(figsize=(7.2, 5.4))
    sc = plt.scatter(best_y, best_h, c=shot_id, cmap="viridis", s=45, label="单炮最佳")
    plt.scatter(result.best.y0, result.best.h, c="r", marker="+", s=180, linewidths=2.5, label="多炮联合最佳")
    plt.gca().invert_yaxis()
    plt.xlabel("y0 (m)")
    plt.ylabel("h/顶部埋深 (m)")
    plt.title("single-shot vs joint：y-h 离散性")
    plt.colorbar(sc, label="炮号")
    plt.legend()
    _finish_figure(output_dir / "single_shot_vs_joint.png", save=save, show=show, dpi=dpi)


def plot_road_geometry_3d(
    geometry: RoadGeometry,
    cavities: list[Cavity] | None = None,
    output: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制道路、DAS 光纤、锤击炮线和空洞位置的三维示意图。"""

    cavities = cavities or []
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    x_min, x_max = float(geometry.channel_x[0]), float(geometry.channel_x[-1])
    xs = [x_min, x_max, x_max, x_min, x_min]
    ys = [geometry.fiber_y, geometry.fiber_y, float(geometry.shot_y), float(geometry.shot_y), geometry.fiber_y]
    zs = [0, 0, 0, 0, 0]
    ax.plot(xs, ys, zs, color="0.45", lw=2, label="道路边界")
    ax.plot(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), 0, "b-", lw=2.5, label="DAS 光纤")
    ax.scatter(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), 0, s=10, c="b", alpha=0.45, label="DAS 通道")
    ax.plot(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), 0, "r--", lw=2, label="锤击炮线")
    ax.scatter(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), 0, s=28, c="r", label="锤击点")
    for cavity in cavities:
        ax.scatter(cavity.x0, cavity.y0, cavity.h, s=130, c="orange", marker=_shape_marker(cavity), edgecolors="k", label=f"{cavity.shape}:{cavity.label}")
        ax.plot([cavity.x0, cavity.x0], [cavity.y0, cavity.y0], [0, cavity.h], "k:", lw=1)
        points, _ = cavity.scatter_points()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=14, c="darkorange", alpha=0.6)
    ax.text(x_min, geometry.road_width * 0.5, 0.4, f"W={geometry.road_width:.1f} m", color="k")
    ax.set_xlabel("x 沿道路/光纤方向 (m)")
    ax.set_ylabel("y 横穿道路方向 (m)")
    ax.set_zlabel("z 深度 (m)")
    ax.set_title("三维道路 DAS + 锤击几何示意")
    ax.set_zlim(max(6.0, max([c.h for c in cavities], default=2.0) + 1.0), -0.5)
    ax.view_init(elev=23, azim=-58)
    ax.legend(loc="upper left")
    _finish_figure(output, save=save, show=show, dpi=dpi)


def geometry_yz_section_metadata(geometry: RoadGeometry, cavities: list[Cavity] | None = None) -> dict[str, object]:
    """返回 y-z 横穿道路剖面的坐标定义，供测试和人工核查使用。

    本项目统一约定：x 沿道路，y 横穿道路，z 为深度且向下为正。第三个
    geometry 子图必须以 y 为横轴、z/深度为纵轴，并把异常体画在
    ``(cavity_y, cavity_depth)``，不能误画成 z-y。
    """

    cavities = cavities or []
    return {
        "title": "横穿道路剖面（y-z）",
        "x_axis": "y",
        "y_axis": "z_depth_positive_down",
        "fiber_y": float(geometry.fiber_y),
        "shot_y": float(geometry.shot_y),
        "y_limits": (float(geometry.fiber_y) - 2.0, float(geometry.shot_y) + 2.0),
        "z_limits": (6.0, -0.5),
        "cavity_points": [(float(cav.y0), float(cav.h)) for cav in cavities],
    }


def plot_geometry_plan_and_sections(
    geometry: RoadGeometry,
    cavities: list[Cavity] | None = None,
    output: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制 x-y 平面布设图、x-z 剖面图和 y-z 剖面图。"""

    cavities = cavities or []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x_min, x_max = float(geometry.channel_x[0]), float(geometry.channel_x[-1])
    axes[0].fill_between([x_min, x_max], geometry.fiber_y, float(geometry.shot_y), color="0.92", label="道路范围")
    axes[0].plot(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), "b-", lw=2, label="DAS 光纤")
    axes[0].scatter(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), s=10, c="b", alpha=0.5)
    axes[0].scatter(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), s=26, c="r", label="锤击点")
    for idx, cavity in enumerate(cavities):
        _draw_anomaly_plan(axes[0], cavity, label=f"{cavity.shape} 异常体" if idx == 0 else None)
        axes[0].scatter(cavity.x0, cavity.y0, s=80, c="orange", marker=_shape_marker(cavity), edgecolors="k")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("x-y 平面布设")
    axes[0].legend(loc="upper right")
    axes[0].axis("equal")

    axes[1].axhline(0, color="0.25", lw=1)
    for cavity in cavities:
        _draw_anomaly_xz(axes[1], cavity)
        axes[1].scatter(cavity.x0, cavity.h, s=80, c="orange", marker=_shape_marker(cavity), edgecolors="k")
        axes[1].vlines(cavity.x0, 0, cavity.h, color="0.35", linestyles=":")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(6, -0.5)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z/深度 (m)")
    axes[1].set_title("x-z 沿道路剖面")

    axes[2].axhline(0, color="0.25", lw=1)
    axes[2].axvline(geometry.fiber_y, color="b", lw=2, label="光纤侧")
    axes[2].axvline(float(geometry.shot_y), color="r", lw=2, linestyle="--", label="锤击侧")
    for idx, cavity in enumerate(cavities):
        _draw_anomaly_yz(axes[2], cavity)
        axes[2].scatter(cavity.y0, cavity.h, s=80, c="orange", marker=_shape_marker(cavity), edgecolors="k", label=f"{cavity.shape} 异常体" if idx == 0 else None)
    axes[2].set_xlim(geometry.fiber_y - 2, float(geometry.shot_y) + 2)
    axes[2].set_ylim(6, -0.5)
    axes[2].set_xlabel("y 横穿道路方向 (m)")
    axes[2].set_ylabel("z/深度 (m，向下为正)")
    axes[2].set_title("横穿道路剖面（y-z）")
    axes[2].text(
        0.02,
        0.04,
        "光纤 y=0；锤击线 y=W；0 在上、深部在下",
        transform=axes[2].transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    axes[2].legend(loc="upper right")
    _finish_figure(output, save=save, show=show, dpi=dpi)


def plot_velocity_model(
    model: LayeredRayleighVelocityModel,
    x_range: tuple[float, float],
    cavities: list[Cavity] | None = None,
    output: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
    effective_velocity: float | None = None,
    velocity_info: dict[str, object] | None = None,
) -> None:
    """绘制当前运动学正演/扫描实际使用的等效速度模型。

    ``uniform`` 模式显示单一 VR；``layered-effective`` 模式显示层状速度，
    并在图注中标出折算后的 ``VR_eff``。注意这里仍是等效瑞雷波走时模型，
    不是 elastic3d 中的 Vp/Vs/rho 全波场模型。
    """

    cavities = cavities or []
    x = np.linspace(x_range[0], x_range[1], 180)
    z = np.linspace(0, model.max_depth, 160)
    section = model.section(x, z)
    plt.figure(figsize=(10, 5.2))
    im = plt.imshow(
        section,
        origin="upper",
        aspect="auto",
        extent=[x[0], x[-1], z[-1], z[0]],
        cmap="viridis",
    )
    plt.colorbar(im, label="等效瑞雷波速度 (m/s)")
    for layer in model.layers:
        plt.axhline(layer.bottom, color="w", lw=0.9, alpha=0.85)
        z_mid = 0.5 * (layer.top + layer.bottom)
        plt.text(
            x[0] + 0.02 * (x[-1] - x[0]),
            z_mid,
            f"{layer.name}: {layer.velocity:.0f} m/s",
            color="white",
            fontsize=9,
            va="center",
            bbox={"facecolor": "black", "alpha": 0.25, "edgecolor": "none"},
        )
    for cavity in cavities:
        plt.scatter(cavity.x0, cavity.h, s=130, c="orange", marker=_shape_marker(cavity), edgecolors="k", label=f"{cavity.shape} 异常体")
    plt.xlabel("x 沿道路方向 (m)")
    plt.ylabel("z/深度 (m)")
    if velocity_info:
        mode = velocity_info.get("velocity_mode", "unknown")
        vr = float(velocity_info.get("rayleigh_velocity", effective_velocity or 0.0))
        vr_eff = float(velocity_info.get("effective_velocity", effective_velocity or vr))
        f0 = float(velocity_info.get("source_frequency", 0.0))
        wavelength = float(velocity_info.get("wavelength", 0.0))
        factor = float(velocity_info.get("sensitivity_depth_factor", 0.0))
        if mode == "uniform":
            title = f"uniform：正演/扫描使用单一 VR={vr:.1f} m/s"
            note = "当前 velocity_mode=uniform；layer_depths/layer_velocities 不参与运动学走时。"
        else:
            title = f"layered-effective：层状速度折算 VR_eff={vr_eff:.1f} m/s"
            note = (
                f"VR={vr:.1f} m/s, f={f0:.1f} Hz, lambda=VR/f={wavelength:.2f} m, "
                f"alpha={factor:.2f}\n"
                f"layer_depths={velocity_info.get('layer_depths')}, "
                f"layer_velocities={velocity_info.get('layer_velocities')}"
            )
        plt.title(title)
        plt.text(
            0.5,
            -0.18,
            note,
            transform=plt.gca().transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    else:
        suffix = "" if effective_velocity is None else f"；当前 VR_eff={effective_velocity:.1f} m/s"
        plt.title(f"简化分层等效瑞雷波速度模型{suffix}")
    if cavities:
        plt.legend(loc="lower right")
    _finish_figure(output, save=save, show=show, dpi=dpi)


def plot_diffraction_path_demo(
    geometry: RoadGeometry,
    cavity: Cavity,
    shot_index: int,
    channel_index: int,
    output: str | Path | None = None,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """绘制直达路径与源-空洞-接收绕射路径对比图。"""

    source = geometry.shot_xyz[shot_index]
    receiver = geometry.channel_xyz[channel_index]
    d = np.asarray(cavity.xyz)
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([source[0], receiver[0]], [source[1], receiver[1]], [source[2], receiver[2]], "k-", lw=2, label="直达路径 S-G")
    ax.plot([source[0], d[0], receiver[0]], [source[1], d[1], receiver[1]], [source[2], d[2], receiver[2]], "r--", lw=2.5, label="绕射路径 S-D-G")
    ax.scatter(*source, s=90, c="r", label="震源 S")
    ax.scatter(*receiver, s=90, c="b", label="接收点 G")
    ax.scatter(*d, s=130, c="orange", edgecolors="k", label="异常体 D")
    ax.text2D(
        0.03,
        0.94,
        "直达波: t = t0 + |S-G| / VR\n绕射波: t = t0 + (|S-D| + |D-G|) / VR",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z/深度 (m)")
    ax.set_title("直达路径与三维绕射路径示意")
    ax.set_zlim(max(5.0, cavity.h + 1.0), -0.5)
    ax.view_init(elev=23, azim=-58)
    ax.legend(loc="upper right")
    _finish_figure(output, save=save, show=show, dpi=dpi)


def compute_direct_wavefield_snapshot(
    geometry: RoadGeometry,
    source_index: int,
    velocity: float,
    frame_time: float,
    t0: float,
    xx: FloatArray,
    yy: FloatArray,
    sigma: float,
) -> tuple[FloatArray, float]:
    """计算直达波等效运动学快照。

    这是平面 x-y 上的传播示意：用高斯等时圈表示直达瑞雷波前，不是弹性
    波方程数值快照。返回值包括示意振幅场和当前直达波前半径。
    """

    sx, sy, _ = geometry.shot_xyz[source_index]
    radius = max(0.0, velocity * (frame_time - t0))
    dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    spreading = 1.0 / np.sqrt(np.maximum(radius, 1.0))
    field = spreading * np.exp(-0.5 * ((dist - radius) / sigma) ** 2)
    return field, radius


def compute_scattered_wavefield_snapshot(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    source_index: int,
    velocity: float,
    frame_time: float,
    t0: float,
    xx: FloatArray,
    yy: FloatArray,
    sigma: float,
) -> tuple[FloatArray, list[tuple[Cavity, float, float]]]:
    """计算异常体散射波等效运动学快照。

    每个异常体先等待直达波从震源到达其三维散射中心，触发时间为
    ``t0 + |S-D|/VR``；之后才从异常体平面位置向外扩散。该函数只用于
    传播路径教学展示，不代表真实弹性散射场。
    """

    sx, sy, sz = geometry.shot_xyz[source_index]
    scatter = np.zeros_like(xx, dtype=float)
    fronts: list[tuple[Cavity, float, float]] = []
    for cav in cavities:
        sd = np.sqrt((sx - cav.x0) ** 2 + (sy - cav.y0) ** 2 + (sz - cav.h) ** 2)
        trigger = t0 + sd / velocity
        if frame_time < trigger:
            fronts.append((cav, trigger, -1.0))
            continue
        radius = velocity * (frame_time - trigger)
        dist = np.sqrt((xx - cav.x0) ** 2 + (yy - cav.y0) ** 2)
        spreading = 1.0 / np.sqrt(np.maximum(sd + radius, 1.0))
        scatter += 0.7 * cav.scattering_strength * spreading * np.exp(-0.5 * ((dist - radius) / sigma) ** 2)
        fronts.append((cav, trigger, radius))
    return scatter, fronts


def _wavefield_grid(geometry: RoadGeometry) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    x = np.linspace(float(geometry.channel_x[0]), float(geometry.channel_x[-1]), 220)
    y = np.linspace(geometry.fiber_y, float(geometry.shot_y), 120)
    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def _wavefield_frame_times(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    source_index: int,
    velocity: float,
    t0: float,
) -> dict[str, float]:
    sx, sy, sz = geometry.shot_xyz[source_index]
    hit_times = [
        t0 + np.sqrt((sx - cav.x0) ** 2 + (sy - cav.y0) ** 2 + (sz - cav.h) ** 2) / velocity
        for cav in cavities
    ]
    first_hit = min(hit_times)
    travel_to_hit = max(first_hit - t0, geometry.dt)
    return {
        "wavefield_frame_early.png": t0 + 0.35 * travel_to_hit,
        "wavefield_frame_hit_cavity.png": first_hit,
        "wavefield_frame_scattered.png": min(geometry.t_max * 0.8, first_hit + max(0.08, 0.55 * travel_to_hit)),
    }


def animate_kinematic_wavefield(
    geometry: RoadGeometry,
    cavity: Cavity | list[Cavity],
    source_index: int,
    velocity: float,
    output: str | Path,
    t0: float = 0.02,
    n_frames: int = 48,
    fps: int = 10,
    save: bool = True,
    show: bool = False,
    velocity_info: dict[str, object] | None = None,
) -> None:
    """生成等效运动学波场 GIF，展示直达波前与异常体散射波前。

    动画使用等时圈和简化高斯波包构造，只用于检查传播顺序、散射触发
    时刻和几何关系，不是严格弹性波场快照。
    """

    output = Path(output)
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
    cavities = cavity if isinstance(cavity, list) else [cavity]
    if not cavities:
        raise ValueError("至少需要一个异常体才能展示散射波场。")
    x, y, xx, yy = _wavefield_grid(geometry)
    sigma = max(0.8, 0.7 * velocity * geometry.dt * 8.0)
    frame_times = _wavefield_frame_times(geometry, cavities, source_index, velocity, t0)
    t_end = min(geometry.t_max * 0.8, max(frame_times.values()) + 0.12)
    times = np.linspace(t0, t_end, n_frames)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    title_note = _wavefield_velocity_note(velocity_info)

    def draw_frame(frame_time: float, global_time: float | None = None) -> list[object]:
        ax.clear()
        direct, direct_radius = compute_direct_wavefield_snapshot(geometry, source_index, velocity, frame_time, t0, xx, yy, sigma)
        scatter, fronts = compute_scattered_wavefield_snapshot(geometry, cavities, source_index, velocity, frame_time, t0, xx, yy, sigma)
        field = direct + scatter
        im = ax.imshow(
            field,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap="inferno",
            vmin=0.0,
            vmax=max(0.35, float(np.percentile(field, 99.8))),
        )
        sx, sy, _ = geometry.shot_xyz[source_index]
        ax.plot(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), "c-", lw=2, label="DAS 光纤")
        ax.plot(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), "w.", ms=3, alpha=0.5, label="锤击线")
        ax.scatter([sx], [sy], s=90, c="lime", edgecolors="k", label="当前震源")
        ax.add_patch(patches.Circle((sx, sy), direct_radius, fill=False, ec="white", lw=1.5, ls="--", alpha=0.9))
        for cav, trigger, radius in fronts:
            _draw_anomaly_plan(ax, cav, None)
            if radius >= 0:
                ax.add_patch(patches.Circle((cav.x0, cav.y0), radius, fill=False, ec="deepskyblue", lw=1.5, ls=":", alpha=0.9))
            ax.text(cav.x0, cav.y0, f"{cav.shape}\n触发 {trigger:.3f}s", color="white", fontsize=8, ha="center", va="bottom")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x 沿道路方向 (m)")
        ax.set_ylabel("y 横穿道路方向 (m)")
        time_text = f"t={frame_time:.3f} s" if global_time is None else f"global_time={global_time:.3f} s, local_t={frame_time:.3f} s"
        ax.set_title(f"x-y 平面运动学波场示意（非弹性波场）；{time_text}\n深度 z 只进入 S-D-G 走时，不显示完整三维波场；{title_note}", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        return [im]

    def update(frame: int) -> list[object]:
        return draw_frame(float(times[frame]))

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    if save:
        writer = animation.PillowWriter(fps=fps)
        anim.save(output, writer=writer)
    if show:
        draw_frame(float(times[min(n_frames // 2, n_frames - 1)]))
        plt.show()
    else:
        plt.close(fig)


def plot_kinematic_wavefield_frames(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    source_index: int,
    velocity: float,
    outdir: str | Path,
    t0: float = 0.02,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
    velocity_info: dict[str, object] | None = None,
    filename_prefix: str = "",
) -> None:
    """输出等效运动学波场的三个关键静态帧。

    三帧分别对应早期直达波、波前到达主异常体附近、散射波已展开。它们
    是与当前运动学正演一致的传播示意，不是严格弹性波场快照。
    """

    if not cavities:
        return
    outdir = Path(outdir)
    frame_times = _wavefield_frame_times(geometry, cavities, source_index, velocity, t0)
    for filename, frame_time in frame_times.items():
        output_name = filename
        if filename_prefix:
            output_name = filename.replace("wavefield_", f"{filename_prefix}_", 1)
        _plot_single_wavefield_frame(
            geometry,
            cavities,
            source_index,
            velocity,
            frame_time,
            outdir / output_name,
            t0=t0,
            save=save,
            show=show,
            dpi=dpi,
            velocity_info=velocity_info,
        )


def _draw_wire_sphere(
    ax: plt.Axes,
    center: tuple[float, float, float],
    radius: float,
    *,
    color: str,
    alpha: float = 0.45,
    half_down: bool = False,
    label: str | None = None,
) -> None:
    """在三维图中画等时球面/半球面线框。

    ``half_down=True`` 用于地表震源的直达波前：只画向地下扩展的半球，
    因为当前示意强调道路浅层瑞雷波/近地表传播几何，而不是空气中的波场。
    """

    if radius <= 0:
        return
    u = np.linspace(0, 2 * np.pi, 40)
    v_max = np.pi / 2 if half_down else np.pi
    v = np.linspace(0, v_max, 18)
    uu, vv = np.meshgrid(u, v)
    cx, cy, cz = center
    x = cx + radius * np.sin(vv) * np.cos(uu)
    y = cy + radius * np.sin(vv) * np.sin(uu)
    z = cz + radius * np.cos(vv)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.6, label=label)


def _plot_single_wavefield_frame_3d(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    source_index: int,
    velocity: float,
    frame_time: float,
    output: Path,
    t0: float,
    save: bool,
    show: bool,
    dpi: int,
    velocity_info: dict[str, object] | None = None,
) -> None:
    """绘制三维运动学等时面示意，不是完整弹性波场。"""

    sx, sy, sz = geometry.shot_xyz[source_index]
    direct_radius = max(0.0, velocity * max(0.0, frame_time - t0))
    z_max = max(6.0, max((cav.h + 2.5 * cav.radius for cav in cavities), default=4.0))
    fig = plt.figure(figsize=(9.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    x0, x1 = float(geometry.channel_x[0]), float(geometry.channel_x[-1])
    y0, y1 = float(geometry.fiber_y), float(geometry.shot_y)

    # 地表矩形只作为几何参照；真正传播仍由等时面示意，不是弹性波方程快照。
    ax.plot([x0, x1], [y0, y0], [0, 0], "c-", lw=2.2, label="DAS 光纤 z=0")
    ax.plot(geometry.shot_x, np.full_like(geometry.shot_x, y1), np.zeros_like(geometry.shot_x), "r.", ms=3, alpha=0.55, label="锤击线 z=0")
    ax.scatter([sx], [sy], [sz], s=85, c="lime", edgecolors="k", label="当前震源")
    _draw_wire_sphere(ax, (float(sx), float(sy), float(sz)), direct_radius, color="white", alpha=0.55, half_down=True, label="直达等时半球")

    for cav in cavities:
        trigger = t0 + np.sqrt((sx - cav.x0) ** 2 + (sy - cav.y0) ** 2 + (sz - cav.h) ** 2) / velocity
        scatter_radius = velocity * (frame_time - trigger)
        points, _ = cav.scatter_points()
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c="darkorange", alpha=0.55)
        ax.scatter([cav.x0], [cav.y0], [cav.h], s=70, c="orange", edgecolors="k", label=f"{cav.shape} 异常体")
        ax.plot([sx, cav.x0], [sy, cav.y0], [sz, cav.h], color="orange", ls=":", lw=1.1)
        if scatter_radius >= 0:
            _draw_wire_sphere(ax, (cav.x0, cav.y0, cav.h), float(scatter_radius), color="deepskyblue", alpha=0.42, half_down=False, label="散射等时球面")
        ax.text(cav.x0, cav.y0, cav.h, f"{cav.shape}\n触发 {trigger:.3f}s", fontsize=8)

    note = _wavefield_velocity_note(velocity_info)
    ax.set_title(
        "三维运动学波场示意，不是完整弹性波场\n"
        f"3D kinematic wavefield schematic; t={frame_time:.3f}s; {note}",
        fontsize=10,
    )
    ax.set_xlabel("x 沿道路方向 (m)")
    ax.set_ylabel("y 横穿道路方向 (m)")
    ax.set_zlabel("z 深度 (m，向下为正)")
    ax.set_xlim(x0, x1)
    ax.set_ylim(min(y0, y1) - 1.0, max(y0, y1) + 1.0)
    ax.set_zlim(z_max, -0.5)
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    _finish_figure(output, save=save, show=show, dpi=dpi)


def plot_kinematic_wavefield_frames_3d(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    source_index: int,
    velocity: float,
    outdir: str | Path,
    t0: float = 0.02,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
    velocity_info: dict[str, object] | None = None,
    filename_prefix: str = "06_wavefield_3d",
) -> list[Path]:
    """输出三维运动学等时面关键帧。

    深度 z 进入 S-D-G 走时和三维几何显示，但这里仍然只是等时面示意；
    真正的 x-y-z 弹性波场应使用 ``elastic3d``。
    """

    if not cavities:
        return []
    outdir = Path(outdir)
    written: list[Path] = []
    frame_times = _wavefield_frame_times(geometry, cavities, source_index, velocity, t0)
    for filename, frame_time in frame_times.items():
        output_name = filename.replace("wavefield_", f"{filename_prefix}_", 1)
        output = outdir / output_name
        _plot_single_wavefield_frame_3d(
            geometry,
            cavities,
            source_index,
            velocity,
            frame_time,
            output,
            t0=t0,
            save=save,
            show=show,
            dpi=dpi,
            velocity_info=velocity_info,
        )
        written.append(output)
    return written


def animate_kinematic_wavefield_3d(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    source_index: int,
    velocity: float,
    output: str | Path,
    t0: float = 0.02,
    n_frames: int = 32,
    fps: int = 8,
    save: bool = True,
    show: bool = False,
    velocity_info: dict[str, object] | None = None,
) -> None:
    """生成三维运动学等时面 GIF，不是完整弹性波场。"""

    if not cavities:
        return
    output = Path(output)
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
    frame_times = _wavefield_frame_times(geometry, cavities, source_index, velocity, t0)
    t_end = min(geometry.t_max * 0.8, max(frame_times.values()) + 0.10)
    times = np.linspace(t0, t_end, n_frames)
    x0, x1 = float(geometry.channel_x[0]), float(geometry.channel_x[-1])
    y0, y1 = float(geometry.fiber_y), float(geometry.shot_y)
    z_max = max(6.0, max((cav.h + 2.5 * cav.radius for cav in cavities), default=4.0))
    note = _wavefield_velocity_note(velocity_info)

    fig = plt.figure(figsize=(9.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    def draw(frame_time: float) -> list[object]:
        ax.clear()
        sx, sy, sz = geometry.shot_xyz[source_index]
        direct_radius = max(0.0, velocity * max(0.0, frame_time - t0))
        ax.plot([x0, x1], [y0, y0], [0, 0], "c-", lw=2.2, label="DAS 光纤 z=0")
        ax.plot(geometry.shot_x, np.full_like(geometry.shot_x, y1), np.zeros_like(geometry.shot_x), "r.", ms=3, alpha=0.55, label="锤击线 z=0")
        ax.scatter([sx], [sy], [sz], s=85, c="lime", edgecolors="k", label="当前震源")
        _draw_wire_sphere(ax, (float(sx), float(sy), float(sz)), direct_radius, color="white", alpha=0.55, half_down=True, label="直达等时半球")
        for cav in cavities:
            trigger = t0 + np.sqrt((sx - cav.x0) ** 2 + (sy - cav.y0) ** 2 + (sz - cav.h) ** 2) / velocity
            scatter_radius = velocity * (frame_time - trigger)
            ax.scatter([cav.x0], [cav.y0], [cav.h], s=70, c="orange", edgecolors="k", label=f"{cav.shape} 异常体")
            if scatter_radius >= 0:
                _draw_wire_sphere(ax, (cav.x0, cav.y0, cav.h), float(scatter_radius), color="deepskyblue", alpha=0.42, half_down=False, label="散射等时球面")
        ax.set_title("三维运动学波场示意，不是完整弹性波场\n" f"t={frame_time:.3f}s; {note}", fontsize=10)
        ax.set_xlabel("x 沿道路方向 (m)")
        ax.set_ylabel("y 横穿道路方向 (m)")
        ax.set_zlabel("z 深度 (m，向下为正)")
        ax.set_xlim(x0, x1)
        ax.set_ylim(min(y0, y1) - 1.0, max(y0, y1) + 1.0)
        ax.set_zlim(z_max, -0.5)
        ax.view_init(elev=24, azim=-58)
        ax.legend(loc="upper left", fontsize=8)
        return []

    anim = animation.FuncAnimation(fig, lambda i: draw(float(times[i])), frames=len(times), interval=1000 / fps, blit=False)
    if save:
        anim.save(output, writer=animation.PillowWriter(fps=fps))
    if show:
        draw(float(times[min(len(times) // 2, len(times) - 1)]))
        plt.show()
    else:
        plt.close(fig)


def _plot_single_wavefield_frame(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    source_index: int,
    velocity: float,
    frame_time: float,
    output: Path,
    t0: float,
    save: bool,
    show: bool,
    dpi: int,
    velocity_info: dict[str, object] | None = None,
) -> None:
    sx, sy, _ = geometry.shot_xyz[source_index]
    x, y, xx, yy = _wavefield_grid(geometry)
    sigma = max(0.8, 0.7 * velocity * geometry.dt * 8.0)
    direct, direct_radius = compute_direct_wavefield_snapshot(geometry, source_index, velocity, frame_time, t0, xx, yy, sigma)
    scatter, fronts = compute_scattered_wavefield_snapshot(geometry, cavities, source_index, velocity, frame_time, t0, xx, yy, sigma)
    field = direct + scatter
    plt.figure(figsize=(8.5, 5.2))
    plt.imshow(field, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="auto", cmap="inferno", vmin=0, vmax=max(0.35, float(np.percentile(field, 99.8))))
    plt.colorbar(label="归一化示意振幅")
    plt.plot(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), "c-", lw=2, label="DAS 光纤")
    plt.plot(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), "w.", ms=3, alpha=0.5, label="锤击线")
    plt.scatter([sx], [sy], s=90, c="lime", edgecolors="k", label="当前震源")
    plt.gca().add_patch(patches.Circle((sx, sy), direct_radius, fill=False, ec="white", lw=1.5, ls="--", alpha=0.9, label="直达波前"))
    for cav, trigger, radius in fronts:
        _draw_anomaly_plan(plt.gca(), cav, None)
        if radius >= 0:
            plt.gca().add_patch(patches.Circle((cav.x0, cav.y0), radius, fill=False, ec="deepskyblue", lw=1.5, ls=":", alpha=0.9))
        plt.text(cav.x0, cav.y0, f"{cav.shape}\n触发 {trigger:.3f}s", color="white", fontsize=8, ha="center", va="bottom")
    plt.xlabel("x 沿道路方向 (m)")
    plt.ylabel("y 横穿道路方向 (m)")
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.gca().set_aspect("equal", adjustable="box")
    note = _wavefield_velocity_note(velocity_info)
    plt.title(f"x-y 平面运动学波场示意（非弹性波场）；t={frame_time:.3f} s\n深度 z 只进入 S-D-G 走时；{note}", fontsize=9)
    plt.legend(loc="upper right")
    _finish_figure(output, save=save, show=show, dpi=dpi)


def plot_multishot_wavefield_frames(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    shot_indices: list[int],
    velocity: float,
    outdir: str | Path,
    t0: float = 0.02,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
    velocity_info: dict[str, object] | None = None,
    filename_prefix: str = "multishot_frame",
) -> list[Path]:
    """输出少量多炮波场关键帧。

    每个炮点只输出一张“散射已展开”的代表性图，避免 multi-shot 模式默认
    生成大量单帧。它只是多炮覆盖的传播示意，不是多炮联合反演。
    """

    if not cavities:
        return []
    outdir = Path(outdir)
    written: list[Path] = []
    for shot_index in shot_indices:
        frame_time = _wavefield_frame_times(geometry, cavities, shot_index, velocity, t0)["wavefield_frame_scattered.png"]
        output = outdir / f"{filename_prefix}_shot{shot_index:03d}.png"
        _plot_single_wavefield_frame(
            geometry,
            cavities,
            shot_index,
            velocity,
            frame_time,
            output,
            t0=t0,
            save=save,
            show=show,
            dpi=dpi,
            velocity_info=velocity_info,
        )
        written.append(output)
    return written


def animate_multishot_kinematic_wavefield(
    geometry: RoadGeometry,
    cavities: list[Cavity],
    shot_indices: list[int],
    velocity: float,
    output: str | Path,
    t0: float = 0.02,
    n_frames: int = 48,
    fps: int = 10,
    shot_interval: float = 0.25,
    save: bool = True,
    show: bool = False,
    velocity_info: dict[str, object] | None = None,
) -> None:
    """生成多炮顺序激发的等效运动学 GIF。

    每一炮使用局部时间从激发到散射展开；GIF 的 global_time 只是教学展示
    用的顺序时间轴。该动画帮助理解多炮覆盖，不代表多炮联合反演，也不
    代表严格分层介质弹性波场。
    """

    if not cavities or not shot_indices:
        return
    output = Path(output)
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
    x, y, xx, yy = _wavefield_grid(geometry)
    sigma = max(0.8, 0.7 * velocity * geometry.dt * 8.0)
    title_note = _wavefield_velocity_note(velocity_info)
    frames_per_shot = max(4, int(np.ceil(n_frames / max(len(shot_indices), 1))))
    frame_items: list[tuple[int, float, float]] = []
    for order, shot_index in enumerate(shot_indices):
        frame_times = _wavefield_frame_times(geometry, cavities, shot_index, velocity, t0)
        t_end = min(geometry.t_max * 0.8, max(frame_times.values()) + 0.08)
        for local_time in np.linspace(t0, t_end, frames_per_shot):
            frame_items.append((shot_index, float(local_time), order * shot_interval + float(local_time - t0)))
    frame_items = frame_items[:n_frames]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    def draw(item: tuple[int, float, float]) -> list[object]:
        shot_index, local_time, global_time = item
        ax.clear()
        direct, direct_radius = compute_direct_wavefield_snapshot(geometry, shot_index, velocity, local_time, t0, xx, yy, sigma)
        scatter, fronts = compute_scattered_wavefield_snapshot(geometry, cavities, shot_index, velocity, local_time, t0, xx, yy, sigma)
        field = direct + scatter
        im = ax.imshow(field, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], aspect="auto", cmap="inferno", vmin=0.0, vmax=max(0.35, float(np.percentile(field, 99.8))))
        sx, sy, _ = geometry.shot_xyz[shot_index]
        ax.plot(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), "c-", lw=2, label="DAS 光纤")
        ax.plot(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), "w.", ms=3, alpha=0.45, label="锤击线")
        ax.scatter([sx], [sy], s=95, c="lime", edgecolors="k", label=f"当前炮 {shot_index}")
        ax.add_patch(patches.Circle((sx, sy), direct_radius, fill=False, ec="white", lw=1.4, ls="--", alpha=0.9))
        for cav, trigger, radius in fronts:
            _draw_anomaly_plan(ax, cav, None)
            if radius >= 0:
                ax.add_patch(patches.Circle((cav.x0, cav.y0), radius, fill=False, ec="deepskyblue", lw=1.4, ls=":", alpha=0.9))
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x 沿道路方向 (m)")
        ax.set_ylabel("y 横穿道路方向 (m)")
        ax.set_title(
            f"multi-shot x-y 平面运动学示意；shot={shot_index}, source_x={sx:.1f} m, global={global_time:.3f}s\n"
            f"local_t={local_time:.3f}s；z 只进走时；{title_note}",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8)
        return [im]

    anim = animation.FuncAnimation(fig, lambda i: draw(frame_items[i]), frames=len(frame_items), interval=1000 / fps, blit=False)
    if save:
        anim.save(output, writer=animation.PillowWriter(fps=fps))
    if show:
        draw(frame_items[min(len(frame_items) // 2, len(frame_items) - 1)])
        plt.show()
    else:
        plt.close(fig)


def _wavefield_velocity_note(velocity_info: dict[str, object] | None) -> str:
    if not velocity_info:
        return "velocity_mode=unknown"
    mode = velocity_info.get("velocity_mode", "unknown")
    vr = float(velocity_info.get("rayleigh_velocity", 0.0))
    vr_eff = float(velocity_info.get("effective_velocity", vr))
    if mode == "layered-effective":
        return (
            f"layered-effective: VR={vr:.1f}, VR_eff={vr_eff:.1f} m/s；"
            "非严格分层波场"
        )
    return f"uniform: VR={vr:.1f} m/s"
