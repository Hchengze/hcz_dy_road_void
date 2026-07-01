"""示例 08：完整教学型流程，从建模、正演到定位解释。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void.config import RoadVoidConfig, load_config
from road_void.visualization import (
    animate_kinematic_wavefield,
    plot_geometry_plan_and_sections,
    plot_road_geometry_3d,
    plot_score_slices,
    plot_shot_gather,
    plot_velocity_model,
)
from road_void.workflow import run_location_workflow


def main(output_dir: str | Path = "outputs/tutorial", config: RoadVoidConfig | None = None) -> None:
    """按教学步骤执行完整原型流程，并保存关键图件。"""

    config = config or load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("步骤1：构建道路三维几何。")
    geom = config.to_geometry()

    print("步骤2：设定空洞/异常体参数。")
    cavities = config.to_cavities()

    print("步骤3：生成场景图和速度模型图。")
    plot_road_geometry_3d(geom, cavities, output_dir / "01_3d_geometry.png")
    plot_geometry_plan_and_sections(geom, cavities, output_dir / "02_plan_sections.png")
    velocity_model = config.to_velocity_model()
    plot_velocity_model(velocity_model, x_range=(float(geom.channel_x[0]), float(geom.channel_x[-1])), cavities=cavities, output=output_dir / "03_velocity_model.png")

    print("步骤4：进行三维等效瑞雷波正演。")
    workflow = run_location_workflow(config)
    dataset = workflow.dataset
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
        output=output_dir / "04_raw_gather.png",
    )

    print("步骤5：拟合三维直达波速度。")
    fit = workflow.velocity_fit
    print(f"直达波速度拟合结果：VR={fit.velocity:.1f} m/s, t0={fit.t0:.4f} s, RMS={fit.residual_rms:.4f} s")

    print("步骤6：模板减去直达波并归一化残差记录。")
    plot_shot_gather(
        workflow.residual,
        geom,
        shot_index=shot_index,
        title="直达波模板减去后的残差记录",
        output=output_dir / "05_residual_gather.png",
    )

    print("步骤7：扫描三维绕射候选参数。")
    result = workflow.scan_result
    best_times = geom.diffraction_times((result.best.x0, result.best.y0, result.best.h), result.best.velocity, fit.t0)
    plot_shot_gather(
        workflow.residual,
        geom,
        shot_index=shot_index,
        diffraction_times=best_times,
        title="残差记录与最佳三维绕射曲线",
        output=output_dir / "06_residual_best_curve.png",
    )
    true_x = cavities[0].x0 if cavities else None
    true_y = cavities[0].y0 if cavities else None
    true_h = cavities[0].h if cavities else None
    plot_score_slices(result, true_x=true_x, true_y=true_y, true_h=true_h, output=output_dir / "07_scan_score_slices.png")

    print("步骤8：生成运动学传播动画。")
    if cavities:
        animate_kinematic_wavefield(geom, cavities[0], source_index=shot_index, velocity=fit.velocity, output=output_dir / "08_kinematic_wavefield.gif")
    else:
        print("当前配置未启用空洞，跳过空洞散射动画。")

    print("步骤9：输出疑似异常解释。")
    print(
        f"最佳疑似异常体：x={result.best.x0:.1f} m, y={result.best.y0:.1f} m, "
        f"h={result.best.h:.1f} m, VR={result.best.velocity:.1f} m/s"
    )
    print(f"不确定性范围：{result.uncertainty}")
    print("解释原则：该结果表示疑似异常范围，需要结合现场资料和其他检测手段验证。")
    print(f"教学流程图件已保存到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整教学型配置驱动流程")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs/tutorial")
    args = parser.parse_args()
    main(args.output_dir, load_config(args.config))
