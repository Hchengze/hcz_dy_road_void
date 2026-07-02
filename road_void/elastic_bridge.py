"""workflow 与小尺度 elastic3d 的局部验证桥接。

默认道路 workflow 的坐标范围通常是几十到上百米，而 elastic3d 原型是
几十米以内的小网格。这里通过“围绕主异常体裁剪局部区域并平移坐标”
生成 elastic validation case，避免把道路全尺度直接塞进小模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .anomaly import Cavity
from .config import RoadVoidConfig
from .elastic3d import Elastic3DConfig, Elastic3DResult, plot_elastic3d_outputs, run_elastic3d


@dataclass(frozen=True)
class ElasticValidationCase:
    """局部 elastic3d 验证算例。"""

    road_config: RoadVoidConfig
    elastic_config: Elastic3DConfig
    origin_x: float
    origin_y: float
    mapped_cavities: list[Cavity]
    in_bounds: bool
    note: str


def build_local_elastic_validation_case(
    config: RoadVoidConfig,
    *,
    elastic_config: Elastic3DConfig | None = None,
) -> ElasticValidationCase:
    """从 RoadVoidConfig 生成局部 elastic3d 验证模型。

    裁剪规则：以第一个异常体为中心，取 elastic3d 小网格尺寸覆盖的局部
    x-y 区域，并把异常体坐标平移到局部网格中。若没有异常体，则使用
    elastic3d 默认异常体参数。
    """

    ecfg = elastic_config or Elastic3DConfig(nt=220)
    cavities = config.to_cavities()
    if cavities:
        primary = cavities[0]
        origin_x = primary.x0 - 0.5 * ecfg.nx * ecfg.dx
        origin_y = primary.y0 - 0.5 * ecfg.ny * ecfg.dy
        mapped = [_map_cavity_to_local(cav, origin_x, origin_y) for cav in cavities]
    else:
        origin_x = 0.0
        origin_y = 0.0
        mapped = [Cavity(ecfg.anomaly_x, ecfg.anomaly_y, ecfg.anomaly_z, radius=ecfg.anomaly_radius)]
    in_bounds = all(_cavity_in_elastic_bounds(cav, ecfg) for cav in mapped)
    return ElasticValidationCase(
        road_config=config,
        elastic_config=ecfg,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        mapped_cavities=mapped,
        in_bounds=in_bounds,
        note="局部 elastic validation：道路坐标已平移到小尺度 elastic3d 网格，不代表全道路全波场模拟。",
    )


def run_elastic_validation_case(case: ElasticValidationCase) -> Elastic3DResult:
    """运行局部 elastic3d 验证。"""

    if not case.in_bounds:
        raise ValueError("局部异常体没有完全落入 elastic3d 小网格，请调整 nx/ny/dx/dy 或异常体位置。")
    return run_elastic3d(case.elastic_config, case.mapped_cavities)


def plot_elastic_validation_outputs(
    case: ElasticValidationCase,
    result: Elastic3DResult,
    outdir: str | Path,
    *,
    save: bool = True,
    show: bool = False,
    dpi: int = 180,
) -> tuple[Path, Path, Path]:
    """输出 local_model_slices、elastic_gather、elastic_wavefield_snapshot。"""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slice_path = outdir / "local_model_slices.png"
    gather_path = outdir / "elastic_gather.png"
    snapshot_path = outdir / "elastic_wavefield_snapshot.png"
    _plot_local_model(case, result, slice_path, save, show, dpi)
    _plot_gather(result, gather_path, save, show, dpi)
    _plot_snapshot(result, snapshot_path, save, show, dpi)
    # 同时保留 elastic3d 原有输出风格，便于和独立 elastic3d 子命令对照。
    plot_elastic3d_outputs(result, outdir, save=save, show=False, dpi=dpi)
    return slice_path, gather_path, snapshot_path


def _map_cavity_to_local(cavity: Cavity, origin_x: float, origin_y: float) -> Cavity:
    return Cavity(
        x0=cavity.x0 - origin_x,
        y0=cavity.y0 - origin_y,
        h=cavity.h,
        radius=cavity.radius,
        scattering_strength=cavity.scattering_strength,
        attenuation_strength=cavity.attenuation_strength,
        tail_strength=cavity.tail_strength,
        label=cavity.label,
        shape=cavity.shape,
        size_x=cavity.size_x,
        size_y=cavity.size_y,
        size_z=cavity.size_z,
        azimuth=cavity.azimuth,
    )


def _cavity_in_elastic_bounds(cavity: Cavity, config: Elastic3DConfig) -> bool:
    margin = max(cavity.radius, 0.5)
    return (
        margin <= cavity.x0 <= config.nx * config.dx - margin
        and margin <= cavity.y0 <= config.ny * config.dy - margin
        and 0.0 < cavity.h <= config.nz * config.dz - margin
    )


def _plot_local_model(case: ElasticValidationCase, result: Elastic3DResult, output: Path, save: bool, show: bool, dpi: int) -> None:
    cfg = case.elastic_config
    y_idx = cfg.ny // 2
    plt.figure(figsize=(8, 4.8))
    plt.imshow(result.model.vp[:, y_idx, :].T, origin="upper", aspect="auto", extent=[0, cfg.nx * cfg.dx, cfg.nz * cfg.dz, 0], cmap="viridis")
    plt.colorbar(label="Vp (m/s)")
    for cav in case.mapped_cavities:
        plt.scatter([cav.x0], [cav.h], c="orange", edgecolors="k", s=70)
        plt.text(cav.x0, cav.h, f" {cav.shape}", color="white", fontsize=8)
    plt.xlabel("local x (m)")
    plt.ylabel("z 深度 (m)")
    plt.title("local elastic3d validation model：道路局部裁剪/平移，不是全道路模型")
    _finish(output, save, show, dpi)


def _plot_gather(result: Elastic3DResult, output: Path, save: bool, show: bool, dpi: int) -> None:
    clip = np.percentile(np.abs(result.gather), 99)
    plt.figure(figsize=(8.5, 4.8))
    plt.imshow(result.gather, aspect="auto", cmap="seismic", vmin=-clip, vmax=clip)
    plt.colorbar(label="elastic record")
    plt.xlabel("receiver index")
    plt.ylabel("time sample")
    plt.title("elastic validation gather：局部小尺度全波场原型")
    _finish(output, save, show, dpi)


def _plot_snapshot(result: Elastic3DResult, output: Path, save: bool, show: bool, dpi: int) -> None:
    if not result.snapshots:
        return
    snapshot = result.snapshots.get("wavefield_snapshot_mid")
    if snapshot is None:
        snapshot = next(iter(result.snapshots.values()))
    plt.figure(figsize=(8, 4.8))
    plt.imshow(snapshot.T, origin="upper", aspect="auto", cmap="seismic")
    plt.colorbar(label="vz snapshot")
    plt.xlabel("local x index")
    plt.ylabel("z index")
    plt.title("elastic validation wavefield snapshot：小尺度 FDTD 原型")
    _finish(output, save, show, dpi)


def _finish(output: Path, save: bool, show: bool, dpi: int) -> None:
    plt.tight_layout()
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()
