"""示例 09：参数敏感性分析。

该脚本批量改变道路宽度、空洞深度、速度、噪声、采样间距和散射强度，
观察最佳定位结果、误差、置信度和 y-h 耦合风险如何变化。默认使用较粗
扫描网格，目的是快速看趋势，而不是替代精细反演。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void.config import RoadVoidConfig, load_config
from road_void import visualization as _visualization  # 触发中文字体配置
from road_void.workflow import run_location_workflow


def make_quick_config(config: RoadVoidConfig) -> RoadVoidConfig:
    """把扫描网格调粗，用于敏感性分析快速运行。"""

    c = config.cavity
    p = config.processing
    quick_processing = replace(
        p,
        scan_x_min=max(config.geometry.channel_x_min, c.cavity_x - 8.0),
        scan_x_max=min(config.geometry.channel_x_max, c.cavity_x + 8.0),
        scan_x_step=max(2.0, p.scan_x_step),
        scan_y_min=max(1.0, c.cavity_y - 5.0),
        scan_y_max=min(config.geometry.road_width - 1.0, c.cavity_y + 5.0),
        scan_y_step=max(2.0, p.scan_y_step),
        scan_h_min=max(0.4, c.cavity_h - 1.5),
        scan_h_max=max(c.cavity_h + 1.5, 3.0),
        scan_h_step=max(0.6, p.scan_h_step),
        scan_vr_min=config.velocity.rayleigh_velocity - 20.0,
        scan_vr_max=config.velocity.rayleigh_velocity + 20.0,
        scan_vr_step=max(20.0, p.scan_vr_step),
        top_k=5,
    )
    return replace(config, processing=quick_processing)


def run_one(case_name: str, parameter: str, value: float, config: RoadVoidConfig) -> dict[str, float | str | bool]:
    """运行一组参数，并返回定位误差和置信度指标。"""

    result = run_location_workflow(config).scan_result
    true = config.cavity
    best = result.best
    uncertainty = result.uncertainty
    y_width = uncertainty["y0"][1] - uncertainty["y0"][0]
    h_width = uncertainty["h"][1] - uncertainty["h"][0]
    warning = y_width >= 0.5 * config.geometry.road_width or h_width >= 1.5
    return {
        "case": case_name,
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
        "yh_tradeoff_warning": warning,
        "uncertainty_y_width": y_width,
        "uncertainty_h_width": h_width,
    }


def build_cases(config: RoadVoidConfig) -> list[tuple[str, str, float, RoadVoidConfig]]:
    """生成参数敏感性实验组合。"""

    cfg = make_quick_config(config)
    cases: list[tuple[str, str, float, RoadVoidConfig]] = []
    for value in [12.0, cfg.geometry.road_width, 28.0]:
        cases.append(("道路宽度影响", "road_width", value, replace(cfg, geometry=replace(cfg.geometry, road_width=value, source_y=value))))
    for value in [1.0, cfg.cavity.cavity_h, 4.0]:
        cases.append(("空洞深度影响", "cavity_depth", value, replace(cfg, cavity=replace(cfg.cavity, cavity_h=value))))
    for value in [cfg.velocity.rayleigh_velocity - 20.0, cfg.velocity.rayleigh_velocity, cfg.velocity.rayleigh_velocity + 20.0]:
        cases.append(("速度误差影响", "rayleigh_velocity", value, replace(cfg, velocity=replace(cfg.velocity, rayleigh_velocity=value))))
    for value in [0.0, cfg.noise.noise_level, 0.12]:
        cases.append(("噪声影响", "noise_level", value, replace(cfg, noise=replace(cfg.noise, noise_level=value))))
    for value in [1.0, cfg.geometry.channel_spacing, 2.0]:
        cases.append(("通道间距影响", "channel_spacing", value, replace(cfg, geometry=replace(cfg.geometry, channel_spacing=value))))
    for value in [0.5, cfg.cavity.scattering_strength, 1.5]:
        cases.append(("散射强度影响", "scattering_strength", value, replace(cfg, cavity=replace(cfg.cavity, scattering_strength=value))))
    return cases


def save_plots(rows: list[dict[str, float | str | bool]], output_dir: Path) -> None:
    """保存关键参数变化曲线。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    for parameter, y_name, title, filename in [
        ("noise_level", "confidence", "噪声强度对定位置信度的影响", "noise_vs_confidence.png"),
        ("rayleigh_velocity", "h_error", "等效瑞雷波速度误差对深度误差的影响", "vr_vs_h_error.png"),
        ("road_width", "confidence", "道路宽度对定位置信度的影响", "road_width_vs_confidence.png"),
    ]:
        subset = [row for row in rows if row["parameter"] == parameter]
        if not subset:
            continue
        x = [float(row["value"]) for row in subset]
        y = [float(row[y_name]) for row in subset]
        plt.figure(figsize=(7, 4.5))
        plt.plot(x, y, "o-", lw=2)
        plt.xlabel(parameter)
        plt.ylabel(y_name)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def main(output_dir: str | Path = "outputs/sensitivity", config: RoadVoidConfig | None = None) -> None:
    """执行参数敏感性分析，并保存 CSV 与曲线图。"""

    config = config or load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_one(case_name, parameter, value, cfg) for case_name, parameter, value, cfg in build_cases(config)]
    csv_path = output_dir / "parameter_sensitivity_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_plots(rows, output_dir)
    print(f"参数敏感性分析完成：{csv_path}")
    print(f"曲线图已保存到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="参数敏感性分析")
    parser.add_argument("--config", default="configs/default_road_void.yaml")
    parser.add_argument("--output-dir", default="outputs/sensitivity")
    args = parser.parse_args()
    main(args.output_dir, load_config(args.config))
