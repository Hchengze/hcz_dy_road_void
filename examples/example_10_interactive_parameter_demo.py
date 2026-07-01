"""示例 10：半交互式参数实验。

该脚本不做 GUI，而是通过 argparse 快速覆盖几个关键参数，并重新生成
定位图件。适合学习“改一个参数，观察结果怎么变”。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void.config import RoadVoidConfig, apply_overrides, load_config
from road_void.visualization import plot_score_slices, plot_shot_gather
from road_void.workflow import run_location_workflow


def main(output_dir: str | Path = "outputs/interactive", config: RoadVoidConfig | None = None) -> None:
    """运行一次参数实验，并保存原始记录、残差记录和评分图。"""

    config = config or load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow = run_location_workflow(config)
    dataset = workflow.dataset
    geom = dataset.geometry
    cavities = config.to_cavities()
    shot_index = geom.n_shots // 2
    if cavities:
        shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index=shot_index,
        direct_times=dataset.direct_times,
        diffraction_times=dataset.diffraction_times[0] if dataset.diffraction_times else None,
        title="参数实验：原始合成记录",
        output=output_dir / "interactive_raw_gather.png",
    )
    best = workflow.scan_result.best
    best_times = geom.diffraction_times((best.x0, best.y0, best.h), best.velocity, workflow.velocity_fit.t0)
    plot_shot_gather(
        workflow.residual,
        geom,
        shot_index=shot_index,
        diffraction_times=best_times,
        title="参数实验：残差记录与最佳绕射曲线",
        output=output_dir / "interactive_residual_best_curve.png",
    )
    true_x = cavities[0].x0 if cavities else None
    true_y = cavities[0].y0 if cavities else None
    true_h = cavities[0].h if cavities else None
    plot_score_slices(workflow.scan_result, true_x=true_x, true_y=true_y, true_h=true_h, output=output_dir / "interactive_scan_scores.png")
    print(
        "参数实验完成："
        f"最佳疑似异常 x={best.x0:.1f} m, y={best.y0:.1f} m, h={best.h:.1f} m, "
        f"confidence={workflow.scan_result.confidence:.3f}"
    )
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="半交互式参数实验")
    parser.add_argument("--config", default="configs/default_road_void.yaml")
    parser.add_argument("--output-dir", default="outputs/interactive")
    parser.add_argument("--road-width", type=float, default=None)
    parser.add_argument("--cavity-depth", type=float, default=None)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--rayleigh-velocity", type=float, default=None)
    args = parser.parse_args()
    cfg = apply_overrides(
        load_config(args.config),
        road_width=args.road_width,
        cavity_depth=args.cavity_depth,
        noise_level=args.noise_level,
        rayleigh_velocity=args.rayleigh_velocity,
    )
    main(args.output_dir, cfg)
