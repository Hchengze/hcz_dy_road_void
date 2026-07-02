"""道路地下场景模型：层状结构、异常体属性和科研级合成模型展示。

本模块服务于“科研级合成数据”阶段。它比单纯的等效散射点更接近道路
地质描述：包含沥青层、基层、路基层，以及空洞、松散区、管线/管沟、
条带弱异常等。注意它仍是合成研究模型，不是真实工程地质解释。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .anomaly import Cavity
from .config import RoadVoidConfig


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RoadLayer:
    """道路浅层一维层状介质。

    thickness 单位 m；vp/vs 单位 m/s；rho 单位 kg/m3；q_or_attenuation 是
    简化衰减参数。这里的参数用于合成模型说明和局部 elastic 验证参考，
    不等同于真实道路取芯或原位测试结果。
    """

    thickness: float
    vp: float
    vs: float
    rho: float
    q_or_attenuation: float
    label: str


@dataclass(frozen=True)
class ScenarioAnomaly:
    """道路地下异常体的属性级描述。

    shape 与 Cavity 的等效散射几何一致；vp/vs/rho_scale 用于表示空洞、
    松散区或管线等相对背景速度/密度的简化变化。当前 workflow 的扫描
    仍使用运动学散射模型，这些属性主要用于模型展示、DAS-like 数据集
    metadata 和 elastic3d 局部验证。
    """

    shape: str
    x: float
    y: float
    depth: float
    radius: float
    size_x: float | None
    size_y: float | None
    size_z: float | None
    azimuth: float
    vp_scale: float
    vs_scale: float
    rho_scale: float
    scattering_strength: float
    label: str


@dataclass(frozen=True)
class RoadSubsurfaceScenario:
    """科研级道路地下合成场景。

    layers 描述背景道路结构；anomalies 描述局部异常体。该对象可以生成
    x-z、y-z 剖面和简化三维属性体，用于说明“合成数据来自什么地下模型”。
    """

    layers: tuple[RoadLayer, ...]
    anomalies: tuple[ScenarioAnomaly, ...]
    road_width: float
    road_length: float
    max_depth: float = 8.0

    def to_dict(self) -> dict[str, object]:
        return {
            "road_width": self.road_width,
            "road_length": self.road_length,
            "max_depth": self.max_depth,
            "layers": [asdict(layer) for layer in self.layers],
            "anomalies": [asdict(anomaly) for anomaly in self.anomalies],
            "note": "科研级合成道路地下模型，不是真实工程地质模型。",
        }

    def layer_boundaries(self) -> FloatArray:
        return np.cumsum([layer.thickness for layer in self.layers], dtype=float)

    def property_at_depth(self, z: FloatArray, name: str = "vs") -> FloatArray:
        """按深度返回背景层状属性。"""

        values = np.zeros_like(z, dtype=float)
        top = 0.0
        for layer in self.layers:
            bottom = top + layer.thickness
            mask = (z >= top) & (z < bottom)
            values[mask] = float(getattr(layer, name))
            top = bottom
        values[z >= top] = float(getattr(self.layers[-1], name))
        return values


def build_default_subsurface_scenario(config: RoadVoidConfig) -> RoadSubsurfaceScenario:
    """从 RoadVoidConfig 构建默认道路三层地下模型。

    三层结构为沥青/路面层、基层、路基/土体层。参数是教学和合成数据用的
    合理数量级，不应直接当作真实道路材料参数。
    """

    layers = (
        RoadLayer(0.25, 1800.0, 850.0, 2300.0, 60.0, "asphalt layer 沥青/路面层"),
        RoadLayer(0.85, 950.0, 420.0, 2050.0, 35.0, "base layer 基层"),
        RoadLayer(7.0, 650.0, 260.0, 1850.0, 25.0, "subgrade layer 路基/土体层"),
    )
    anomalies = tuple(_scenario_anomaly_from_cavity(cavity) for cavity in config.to_cavities())
    max_depth = max(8.0, max((a.depth + 2.5 * a.radius for a in anomalies), default=6.0))
    return RoadSubsurfaceScenario(
        layers=layers,
        anomalies=anomalies,
        road_width=config.geometry.road_width,
        road_length=config.geometry.road_length,
        max_depth=max_depth,
    )


def _scenario_anomaly_from_cavity(cavity: Cavity) -> ScenarioAnomaly:
    shape = cavity.shape.lower()
    if shape == "sphere":
        vp_scale, vs_scale, rho_scale, label = 0.20, 0.15, 0.30, "void cavity 空洞"
    elif shape in {"line", "zone"}:
        vp_scale, vs_scale, rho_scale, label = 0.70, 0.55, 0.85, "fracture/weak zone 条带弱异常"
    elif shape == "cylinder":
        vp_scale, vs_scale, rho_scale, label = 1.25, 1.10, 1.05, "pipe-like anomaly 管线/圆柱异常"
    else:
        vp_scale, vs_scale, rho_scale, label = 0.65, 0.50, 0.80, "loose zone 松散区"
    return ScenarioAnomaly(
        shape=shape,
        x=cavity.x0,
        y=cavity.y0,
        depth=cavity.h,
        radius=cavity.radius,
        size_x=cavity.size_x,
        size_y=cavity.size_y,
        size_z=cavity.size_z,
        azimuth=cavity.azimuth,
        vp_scale=vp_scale,
        vs_scale=vs_scale,
        rho_scale=rho_scale,
        scattering_strength=cavity.scattering_strength,
        label=label,
    )


def make_property_section_xz(
    scenario: RoadSubsurfaceScenario,
    *,
    nx: int = 220,
    nz: int = 120,
    property_name: str = "vs",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """生成沿道路 x-z 剖面的属性图。异常体按投影方式叠加。"""

    x = np.linspace(0.0, scenario.road_length, nx)
    z = np.linspace(0.0, scenario.max_depth, nz)
    xx, zz = np.meshgrid(x, z)
    section = np.tile(scenario.property_at_depth(z, property_name)[:, None], (1, nx))
    for anomaly in scenario.anomalies:
        mask = _anomaly_mask_xz(anomaly, xx, zz)
        section[mask] *= float(getattr(anomaly, f"{property_name}_scale", 1.0))
    return x, z, section


def make_property_section_yz(
    scenario: RoadSubsurfaceScenario,
    *,
    ny: int = 100,
    nz: int = 120,
    property_name: str = "vs",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """生成横穿道路 y-z 剖面，横轴为 y，纵轴为深度 z。"""

    y = np.linspace(0.0, scenario.road_width, ny)
    z = np.linspace(0.0, scenario.max_depth, nz)
    yy, zz = np.meshgrid(y, z)
    section = np.tile(scenario.property_at_depth(z, property_name)[:, None], (1, ny))
    for anomaly in scenario.anomalies:
        mask = _anomaly_mask_yz(anomaly, yy, zz)
        section[mask] *= float(getattr(anomaly, f"{property_name}_scale", 1.0))
    return y, z, section


def make_property_volume(
    scenario: RoadSubsurfaceScenario,
    *,
    nx: int = 80,
    ny: int = 36,
    nz: int = 36,
    property_name: str = "vs",
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """生成简化三维属性体，默认用于小图展示和 sanity check。"""

    x = np.linspace(0.0, scenario.road_length, nx)
    y = np.linspace(0.0, scenario.road_width, ny)
    z = np.linspace(0.0, scenario.max_depth, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    volume = np.broadcast_to(scenario.property_at_depth(z, property_name)[None, None, :], (nx, ny, nz)).copy()
    for anomaly in scenario.anomalies:
        mask = _anomaly_mask_3d(anomaly, xx, yy, zz)
        volume[mask] *= float(getattr(anomaly, f"{property_name}_scale", 1.0))
    return x, y, z, volume


def plot_subsurface_sections(
    scenario: RoadSubsurfaceScenario,
    output_xz: str | Path,
    output_yz: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """输出科研级地下模型 x-z 与 y-z 剖面图。"""

    _plot_section(*make_property_section_xz(scenario), scenario, output_xz, "x 沿道路方向 (m)", "02b 地下模型 x-z 剖面：Vs", save, show, dpi)
    _plot_section(*make_property_section_yz(scenario), scenario, output_yz, "y 横穿道路方向 (m)", "02c 地下模型 y-z 剖面：Vs", save, show, dpi)


def plot_subsurface_3d(
    scenario: RoadSubsurfaceScenario,
    output: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> None:
    """输出简化三维地下场景图，只表达层状结构和异常体位置。"""

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    x0, x1 = 0.0, scenario.road_length
    y0, y1 = 0.0, scenario.road_width
    for boundary in scenario.layer_boundaries()[:-1]:
        xx, yy = np.meshgrid(np.linspace(x0, x1, 3), np.linspace(y0, y1, 3))
        zz = np.full_like(xx, boundary)
        ax.plot_surface(xx, yy, zz, alpha=0.12, color="gray", edgecolor="none")
        ax.text(x1, y1, boundary, f"z={boundary:.2f} m", fontsize=8)
    for anomaly in scenario.anomalies:
        ax.scatter([anomaly.x], [anomaly.y], [anomaly.depth], s=70, label=f"{anomaly.shape}: {anomaly.label}")
        ax.text(anomaly.x, anomaly.y, anomaly.depth, anomaly.shape, fontsize=8)
    ax.plot([x0, x1], [0, 0], [0, 0], "c-", lw=2, label="DAS 光纤")
    ax.plot([x0, x1], [scenario.road_width, scenario.road_width], [0, 0], "r--", lw=1.5, label="锤击线")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(scenario.max_depth, 0.0)
    ax.set_xlabel("x 沿道路方向 (m)")
    ax.set_ylabel("y 横穿道路方向 (m)")
    ax.set_zlabel("z 深度 (m，向下为正)")
    ax.set_title("02d 简化三维地下模型：层状道路结构与异常体位置\n科研级合成模型，不是真实工程地质模型")
    ax.legend(loc="upper left", fontsize=8)
    _finish(output, save, show, dpi)


def _plot_section(
    coord: FloatArray,
    z: FloatArray,
    section: FloatArray,
    scenario: RoadSubsurfaceScenario,
    output: str | Path,
    xlabel: str,
    title: str,
    save: bool,
    show: bool,
    dpi: int,
) -> None:
    plt.figure(figsize=(8.8, 4.8))
    plt.imshow(section, origin="upper", aspect="auto", extent=[coord[0], coord[-1], z[-1], z[0]], cmap="viridis")
    plt.colorbar(label="Vs (m/s)")
    for boundary in scenario.layer_boundaries()[:-1]:
        plt.axhline(boundary, color="w", lw=1.0, ls="--")
    for anomaly in scenario.anomalies:
        x = anomaly.x if xlabel.startswith("x") else anomaly.y
        plt.scatter([x], [anomaly.depth], c="orange", edgecolors="k", s=60)
        plt.text(x, anomaly.depth, f" {anomaly.shape}", color="white", fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel("z/深度 (m，向下为正)")
    plt.title(title)
    _finish(output, save, show, dpi)


def _finish(output: str | Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()


def _anomaly_mask_xz(anomaly: ScenarioAnomaly, xx: FloatArray, zz: FloatArray) -> FloatArray:
    sx = anomaly.size_x or 2.0 * anomaly.radius
    sz = anomaly.size_z or 2.0 * anomaly.radius
    if anomaly.shape in {"box", "zone", "line"}:
        return (np.abs(xx - anomaly.x) <= sx / 2) & (np.abs(zz - anomaly.depth) <= max(sz, 0.3) / 2)
    return ((xx - anomaly.x) / (sx / 2)) ** 2 + ((zz - anomaly.depth) / (sz / 2)) ** 2 <= 1.0


def _anomaly_mask_yz(anomaly: ScenarioAnomaly, yy: FloatArray, zz: FloatArray) -> FloatArray:
    sy = anomaly.size_y or 2.0 * anomaly.radius
    sz = anomaly.size_z or 2.0 * anomaly.radius
    if anomaly.shape in {"box", "zone", "line"}:
        return (np.abs(yy - anomaly.y) <= sy / 2) & (np.abs(zz - anomaly.depth) <= max(sz, 0.3) / 2)
    return ((yy - anomaly.y) / (sy / 2)) ** 2 + ((zz - anomaly.depth) / (sz / 2)) ** 2 <= 1.0


def _anomaly_mask_3d(anomaly: ScenarioAnomaly, xx: FloatArray, yy: FloatArray, zz: FloatArray) -> FloatArray:
    if anomaly.shape == "box":
        sx = anomaly.size_x or 2 * anomaly.radius
        sy = anomaly.size_y or 2 * anomaly.radius
        sz = anomaly.size_z or 2 * anomaly.radius
        return (np.abs(xx - anomaly.x) <= sx / 2) & (np.abs(yy - anomaly.y) <= sy / 2) & (np.abs(zz - anomaly.depth) <= sz / 2)
    sx = anomaly.size_x or 2 * anomaly.radius
    sy = anomaly.size_y or 2 * anomaly.radius
    sz = anomaly.size_z or 2 * anomaly.radius
    return ((xx - anomaly.x) / (sx / 2)) ** 2 + ((yy - anomaly.y) / (sy / 2)) ** 2 + ((zz - anomaly.depth) / (sz / 2)) ** 2 <= 1.0

