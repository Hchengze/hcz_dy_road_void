"""城市道路空洞三维 DAS 正演、定位与展示示例统一入口。

配置驱动示例：

```
python main.py scan --config configs/default_road_void.yaml
python main.py tutorial --config configs/deep_cavity_demo.yaml --noise-level 0.08
```
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from road_void.config import RoadVoidConfig, apply_overrides, load_config
from road_void.visualization import plot_score_slices, plot_shot_gather
from road_void.workflow import run_location_workflow, simulate_from_config


CASES = (
    "no-cavity",
    "with-cavity",
    "scan",
    "geometry",
    "velocity",
    "wavefield",
    "path",
    "tutorial",
    "sensitivity",
    "interactive",
    "all",
)


def run_no_cavity(config: RoadVoidConfig, output_dir: Path) -> None:
    """运行无空洞三维正演示例。"""

    config = replace(config, cavity=replace(config.cavity, enable_cavity=False))
    dataset = simulate_from_config(config)
    out = output_dir / "main_01_no_cavity_gather.png"
    plot_shot_gather(
        dataset.data,
        dataset.geometry,
        shot_index=dataset.geometry.n_shots // 2,
        direct_times=dataset.direct_times,
        title="无空洞三维道路几何：直达瑞雷波",
        output=out,
    )
    print(f"[无空洞正演] 已保存 {out}")
    print(f"[无空洞正演] 数据形状: {dataset.data.shape}")


def run_with_cavity(config: RoadVoidConfig, output_dir: Path) -> None:
    """运行含空洞三维正演示例。"""

    dataset = simulate_from_config(config)
    geom = dataset.geometry
    cavities = config.to_cavities()
    shot_index = geom.n_shots // 2
    if cavities:
        shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    out = output_dir / "main_02_with_cavity_gather.png"
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index=shot_index,
        direct_times=dataset.direct_times,
        diffraction_times=dataset.diffraction_times[0] if dataset.diffraction_times else None,
        title="含空洞场景：直达波与三维绕射/散射响应",
        output=out,
    )
    print(f"[含空洞正演] 已保存 {out}")
    if cavities:
        c = cavities[0]
        print(f"[含空洞正演] 配置空洞: x={c.x0:.1f} m, y={c.y0:.1f} m, h={c.h:.1f} m")


def run_scan(config: RoadVoidConfig, output_dir: Path) -> None:
    """运行配置驱动的三维空洞定位扫描。"""

    result = run_location_workflow(config)
    dataset = result.dataset
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
        title="原始合成记录：直达波与空洞绕射",
        output=output_dir / "main_03_raw_gather.png",
    )
    best = result.scan_result.best
    best_times = geom.diffraction_times((best.x0, best.y0, best.h), best.velocity, result.velocity_fit.t0)
    plot_shot_gather(
        result.residual,
        geom,
        shot_index=shot_index,
        diffraction_times=best_times,
        title="直达波压制残差与最佳三维绕射曲线",
        output=output_dir / "main_03_residual_best_curve.png",
    )
    true_x = cavities[0].x0 if cavities else None
    true_y = cavities[0].y0 if cavities else None
    true_h = cavities[0].h if cavities else None
    plot_score_slices(
        result.scan_result,
        true_x=true_x,
        true_y=true_y,
        true_h=true_h,
        output=output_dir / "main_03_scan_scores.png",
    )
    fit = result.velocity_fit
    print(f"[定位扫描] 直达波估计 VR={fit.velocity:.1f} m/s, t0={fit.t0:.4f} s, RMS={fit.residual_rms:.4f} s")
    print(
        "[定位扫描] 最佳疑似异常体: "
        f"x={best.x0:.1f} m, y={best.y0:.1f} m, h={best.h:.1f} m, "
        f"VR={best.velocity:.1f} m/s, score={best.score:.4f}"
    )
    print(f"[定位扫描] 不确定性范围: {result.scan_result.uncertainty}")
    print(f"[定位扫描] 图片已保存到 {output_dir}")


def run_showcase_case(case: str, config: RoadVoidConfig, output_dir: Path) -> None:
    """调用展示型示例脚本。"""

    if case == "geometry":
        from examples.example_04_plot_3d_geometry import main as run
    elif case == "velocity":
        from examples.example_05_plot_velocity_model import main as run
    elif case == "wavefield":
        from examples.example_06_wavefield_animation import main as run
    elif case == "path":
        from examples.example_07_diffraction_path_demo import main as run
    elif case == "tutorial":
        from examples.example_08_full_workflow_tutorial import main as run
    elif case == "sensitivity":
        from examples.example_09_parameter_sensitivity import main as run
    elif case == "interactive":
        from examples.example_10_interactive_parameter_demo import main as run
    else:
        raise ValueError(f"未知展示示例: {case}")
    run(output_dir=output_dir, config=config)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="城市道路空洞三维 DAS 正演、定位与展示入口")
    parser.add_argument("case", nargs="?", choices=CASES, help="要运行的示例。")
    parser.add_argument("--case", dest="case_option", choices=CASES, help="要运行的示例，兼容旧入口风格。")
    parser.add_argument("--config", default=None, help="YAML 配置文件路径，例如 configs/default_road_void.yaml。")
    parser.add_argument("--output-dir", default="outputs", help="图片、动画和表格输出目录，默认 outputs。")
    parser.add_argument("--road-width", type=float, default=None, help="覆盖道路宽度 W，单位 m。")
    parser.add_argument("--cavity-depth", type=float, default=None, help="覆盖空洞顶部埋深 h，单位 m。")
    parser.add_argument("--noise-level", type=float, default=None, help="覆盖随机噪声强度。")
    parser.add_argument("--rayleigh-velocity", type=float, default=None, help="覆盖等效瑞雷波速度 VR，单位 m/s。")
    return parser.parse_args()


def main() -> None:
    """执行命令行入口。"""

    args = parse_args()
    case = args.case_option or args.case or "all"
    config = load_config(args.config)
    config = apply_overrides(
        config,
        road_width=args.road_width,
        cavity_depth=args.cavity_depth,
        noise_level=args.noise_level,
        rayleigh_velocity=args.rayleigh_velocity,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if case in ("no-cavity", "all"):
        run_no_cavity(config, output_dir)
    if case in ("with-cavity", "all"):
        run_with_cavity(config, output_dir)
    if case in ("scan", "all"):
        run_scan(config, output_dir)
    for showcase in ("geometry", "velocity", "wavefield", "path", "tutorial", "sensitivity", "interactive"):
        if case in (showcase, "all"):
            target_dir = output_dir / "tutorial" if showcase == "tutorial" and case == "all" else output_dir
            run_showcase_case(showcase, config, target_dir)


if __name__ == "__main__":
    main()
