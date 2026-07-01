"""合成记录、道路几何、速度模型与定位结果的可视化工具。

本模块的图件主要服务于“理解和展示”。其中波场动画是与当前
三维运动学正演一致的等效传播示意，不代表严格弹性波场快照。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, font_manager
import numpy as np
from numpy.typing import NDArray

from .anomaly import Cavity
from .geometry import RoadGeometry
from .scan import CavityScanResult
from .velocity import LayeredRayleighVelocityModel


FloatArray = NDArray[np.float64]


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


def _finish_figure(output: str | Path | None) -> None:
    """保存或显示当前图件。"""

    plt.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_shot_gather(
    data: FloatArray,
    geometry: RoadGeometry,
    shot_index: int,
    direct_times: FloatArray | None = None,
    diffraction_times: FloatArray | None = None,
    title: str | None = None,
    output: str | Path | None = None,
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
    _finish_figure(output)


def plot_score_slices(
    result: CavityScanResult,
    true_x: float | None = None,
    true_y: float | None = None,
    true_h: float | None = None,
    output: str | Path | None = None,
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
    _finish_figure(output)


def plot_road_geometry_3d(
    geometry: RoadGeometry,
    cavities: list[Cavity] | None = None,
    output: str | Path | None = None,
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
        ax.scatter(cavity.x0, cavity.y0, cavity.h, s=130, c="orange", edgecolors="k", label=f"异常体 {cavity.label}")
        ax.plot([cavity.x0, cavity.x0], [cavity.y0, cavity.y0], [0, cavity.h], "k:", lw=1)
    ax.text(x_min, geometry.road_width * 0.5, 0.4, f"W={geometry.road_width:.1f} m", color="k")
    ax.set_xlabel("x 沿道路/光纤方向 (m)")
    ax.set_ylabel("y 横穿道路方向 (m)")
    ax.set_zlabel("z 深度 (m)")
    ax.set_title("三维道路 DAS + 锤击几何示意")
    ax.set_zlim(max(6.0, max([c.h for c in cavities], default=2.0) + 1.0), -0.5)
    ax.view_init(elev=23, azim=-58)
    ax.legend(loc="upper left")
    _finish_figure(output)


def plot_geometry_plan_and_sections(
    geometry: RoadGeometry,
    cavities: list[Cavity] | None = None,
    output: str | Path | None = None,
) -> None:
    """绘制 x-y 平面布设图、x-z 剖面图和 y-z 剖面图。"""

    cavities = cavities or []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x_min, x_max = float(geometry.channel_x[0]), float(geometry.channel_x[-1])
    axes[0].fill_between([x_min, x_max], geometry.fiber_y, float(geometry.shot_y), color="0.92", label="道路范围")
    axes[0].plot(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), "b-", lw=2, label="DAS 光纤")
    axes[0].scatter(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), s=10, c="b", alpha=0.5)
    axes[0].scatter(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), s=26, c="r", label="锤击点")
    for cavity in cavities:
        axes[0].scatter(cavity.x0, cavity.y0, s=110, c="orange", edgecolors="k", label="异常体")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("x-y 平面布设")
    axes[0].legend(loc="upper right")
    axes[0].axis("equal")

    axes[1].axhline(0, color="0.25", lw=1)
    for cavity in cavities:
        axes[1].scatter(cavity.x0, cavity.h, s=110, c="orange", edgecolors="k")
        axes[1].vlines(cavity.x0, 0, cavity.h, color="0.35", linestyles=":")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(6, -0.5)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z/深度 (m)")
    axes[1].set_title("x-z 沿道路剖面")

    axes[2].axhline(0, color="0.25", lw=1)
    axes[2].axvline(geometry.fiber_y, color="b", lw=2, label="光纤侧")
    axes[2].axvline(float(geometry.shot_y), color="r", lw=2, linestyle="--", label="锤击侧")
    for cavity in cavities:
        axes[2].scatter(cavity.y0, cavity.h, s=110, c="orange", edgecolors="k", label="异常体")
    axes[2].set_xlim(geometry.fiber_y - 2, float(geometry.shot_y) + 2)
    axes[2].set_ylim(6, -0.5)
    axes[2].set_xlabel("y (m)")
    axes[2].set_ylabel("z/深度 (m)")
    axes[2].set_title("y-z 横向剖面")
    axes[2].legend(loc="upper right")
    _finish_figure(output)


def plot_velocity_model(
    model: LayeredRayleighVelocityModel,
    x_range: tuple[float, float],
    cavities: list[Cavity] | None = None,
    output: str | Path | None = None,
) -> None:
    """绘制简化分层等效瑞雷波速度模型，并叠加异常体位置。"""

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
        plt.axhline(layer.bottom, color="w", lw=0.8, alpha=0.8)
    for cavity in cavities:
        plt.scatter(cavity.x0, cavity.h, s=130, c="orange", edgecolors="k", label="异常体")
    plt.xlabel("x 沿道路方向 (m)")
    plt.ylabel("z/深度 (m)")
    plt.title("简化分层等效瑞雷波速度模型")
    if cavities:
        plt.legend(loc="lower right")
    _finish_figure(output)


def plot_diffraction_path_demo(
    geometry: RoadGeometry,
    cavity: Cavity,
    shot_index: int,
    channel_index: int,
    output: str | Path | None = None,
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
    _finish_figure(output)


def animate_kinematic_wavefield(
    geometry: RoadGeometry,
    cavity: Cavity,
    source_index: int,
    velocity: float,
    output: str | Path,
    t0: float = 0.02,
    n_frames: int = 48,
    fps: int = 10,
) -> None:
    """生成等效运动学波场 GIF，展示直达波前与空洞散射波前。

    动画中的“波场”由走时圆环和简化振幅构成，只用于解释传播路径、
    绕射出现时刻和 DAS 接收线位置，不应解释为严格弹性波场快照。
    """

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sx, sy, _ = geometry.shot_xyz[source_index]
    x = np.linspace(float(geometry.channel_x[0]), float(geometry.channel_x[-1]), 180)
    y = np.linspace(geometry.fiber_y, float(geometry.shot_y), 90)
    xx, yy = np.meshgrid(x, y)
    dist_source = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    dist_cavity = np.sqrt((xx - cavity.x0) ** 2 + (yy - cavity.y0) ** 2)
    source_to_cavity = np.sqrt((sx - cavity.x0) ** 2 + (sy - cavity.y0) ** 2 + cavity.h**2)
    t_scatter_start = t0 + source_to_cavity / velocity
    times = np.linspace(t0, geometry.t_max * 0.65, n_frames)
    sigma = max(0.8, velocity * geometry.dt * 3.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    image = ax.imshow(
        np.zeros_like(xx),
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        cmap="inferno",
        vmin=0,
        vmax=1.1,
    )
    ax.plot(geometry.channel_x, np.full_like(geometry.channel_x, geometry.fiber_y), "c-", lw=2, label="DAS 光纤")
    ax.scatter(geometry.shot_x, np.full_like(geometry.shot_x, float(geometry.shot_y)), s=12, c="w", alpha=0.5, label="锤击点")
    ax.scatter([sx], [sy], s=90, c="lime", edgecolors="k", label="当前震源")
    ax.scatter([cavity.x0], [cavity.y0], s=110, c="orange", edgecolors="k", label="异常体")
    ax.set_xlabel("x 沿道路方向 (m)")
    ax.set_ylabel("y 横穿道路方向 (m)")
    ax.set_title("等效运动学波场示意")
    ax.legend(loc="upper right")
    time_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, color="w", fontsize=11)

    def update(frame: int) -> list[object]:
        t = times[frame]
        direct_radius = max(0.0, velocity * (t - t0))
        direct = np.exp(-0.5 * ((dist_source - direct_radius) / sigma) ** 2)
        scatter = np.zeros_like(direct)
        if t > t_scatter_start:
            scatter_radius = velocity * (t - t_scatter_start)
            scatter = cavity.scattering_strength * np.exp(-0.5 * ((dist_cavity - scatter_radius) / sigma) ** 2)
        field = direct + scatter
        field /= max(float(np.max(field)), 1e-9)
        image.set_data(field)
        time_text.set_text(f"t = {t:.3f} s")
        return [image, time_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output, writer=writer)
    plt.close(fig)
