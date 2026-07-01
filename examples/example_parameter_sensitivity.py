"""参数敏感性分析教学样例。

该脚本由 ``main.py sensitivity`` 调用，也可以单独运行。它只输出 CSV 和
少量趋势图，避免像 tutorial/all 那样重复生成几何和正演图件。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void import visualization as _visualization  # 触发中文字体配置
from road_void.config import RoadVoidConfig
from road_void.workflow import run_location_workflow
from main import add_geometry_args, add_scan_args, add_wave_args, config_from_args


def quick_config(config: RoadVoidConfig) -> RoadVoidConfig:
    """把扫描网格调粗，保证敏感性分析适合本地快速试验。"""

    p = config.processing
    c = config.cavity
    return replace(
        config,
        processing=replace(
            p,
            scan_x_min=max(0.0, c.cavity_x - 8.0),
            scan_x_max=min(config.geometry.road_length, c.cavity_x + 8.0),
            scan_x_step=max(2.0, p.scan_x_step),
            scan_y_min=max(1.0, c.cavity_y - 5.0),
            scan_y_max=min(config.geometry.road_width - 1.0, c.cavity_y + 5.0),
            scan_y_step=max(2.0, p.scan_y_step),
            scan_h_min=max(0.4, c.cavity_h - 1.5),
            scan_h_max=max(3.0, c.cavity_h + 1.5),
            scan_h_step=max(0.6, p.scan_h_step),
            scan_vr_min=config.velocity.rayleigh_velocity - 20.0,
            scan_vr_max=config.velocity.rayleigh_velocity + 20.0,
            scan_vr_step=max(20.0, p.scan_vr_step),
            top_k=5,
        ),
    )


def run_case(name: str, parameter: str, value: float, config: RoadVoidConfig) -> dict[str, object]:
    result = run_location_workflow(config).scan_result
    best = result.best
    true = config.cavity
    y_width = result.uncertainty["y0"][1] - result.uncertainty["y0"][0]
    h_width = result.uncertainty["h"][1] - result.uncertainty["h"][0]
    return {
        "case": name,
        "parameter": parameter,
        "value": value,
        "best_x": best.x0,
        "best_y": best.y0,
        "best_h": best.h,
        "best_vr": best.velocity,
        "x_error": best.x0 - true.cavity_x,
        "y_error": best.y0 - true.cavity_y,
        "h_error": best.h - true.cavity_h,
        "confidence": result.confidence,
        "score": best.score,
        "yh_tradeoff_warning": y_width >= 0.5 * config.geometry.road_width or h_width >= 1.5,
    }


def build_cases(config: RoadVoidConfig) -> list[tuple[str, str, float, RoadVoidConfig]]:
    cfg = quick_config(config)
    return [
        ("道路宽度影响", "road_width", 12.0, replace(cfg, geometry=replace(cfg.geometry, road_width=12.0, source_y=12.0))),
        ("道路宽度影响", "road_width", cfg.geometry.road_width, cfg),
        ("道路宽度影响", "road_width", 28.0, replace(cfg, geometry=replace(cfg.geometry, road_width=28.0, source_y=28.0))),
        ("空洞深度影响", "cavity_depth", 1.0, replace(cfg, cavity=replace(cfg.cavity, cavity_h=1.0))),
        ("空洞深度影响", "cavity_depth", cfg.cavity.cavity_h, cfg),
        ("空洞深度影响", "cavity_depth", 4.0, replace(cfg, cavity=replace(cfg.cavity, cavity_h=4.0))),
        ("速度误差影响", "rayleigh_velocity", cfg.velocity.rayleigh_velocity - 20.0, replace(cfg, velocity=replace(cfg.velocity, rayleigh_velocity=cfg.velocity.rayleigh_velocity - 20.0))),
        ("速度误差影响", "rayleigh_velocity", cfg.velocity.rayleigh_velocity, cfg),
        ("速度误差影响", "rayleigh_velocity", cfg.velocity.rayleigh_velocity + 20.0, replace(cfg, velocity=replace(cfg.velocity, rayleigh_velocity=cfg.velocity.rayleigh_velocity + 20.0))),
        ("噪声影响", "noise_level", 0.0, replace(cfg, noise=replace(cfg.noise, noise_level=0.0))),
        ("噪声影响", "noise_level", cfg.noise.noise_level, cfg),
        ("噪声影响", "noise_level", 0.12, replace(cfg, noise=replace(cfg.noise, noise_level=0.12))),
    ]


def save_plots(rows: list[dict[str, object]], outdir: Path, save: bool) -> None:
    if not save:
        return
    for parameter, y_name, title, filename in [
        ("noise_level", "confidence", "噪声强度对定位置信度的影响", "noise_vs_confidence.png"),
        ("rayleigh_velocity", "h_error", "等效瑞雷波速度误差对深度误差的影响", "vr_vs_h_error.png"),
        ("road_width", "confidence", "道路宽度对定位置信度的影响", "road_width_vs_confidence.png"),
    ]:
        subset = [row for row in rows if row["parameter"] == parameter]
        plt.figure(figsize=(7, 4.5))
        plt.plot([float(row["value"]) for row in subset], [float(row[y_name]) for row in subset], "o-", lw=2)
        plt.xlabel(parameter)
        plt.ylabel(y_name)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()


def main(output_dir: str | Path, config: RoadVoidConfig, save: bool = True) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [run_case(name, parameter, value, cfg) for name, parameter, value, cfg in build_cases(config)]
    csv_path = outdir / "parameter_sensitivity_results.csv"
    if save:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        save_plots(rows, outdir, save=True)
    print(f"敏感性分析完成，共 {len(rows)} 组；CSV: {csv_path if save else '未保存'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="参数敏感性分析教学样例")
    add_geometry_args(parser)
    add_wave_args(parser)
    add_scan_args(parser)
    parser.add_argument("--outdir", default="outputs/sensitivity")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    main(args.outdir, quick_config(config_from_args(args)), save=not args.no_save)
