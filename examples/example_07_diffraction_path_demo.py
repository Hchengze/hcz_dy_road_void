"""示例 07：解释直达路径、绕射路径和 shot gather 理论曲线。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void.config import RoadVoidConfig, load_config
from road_void.visualization import plot_diffraction_path_demo, plot_shot_gather
from road_void.workflow import simulate_from_config


def main(output_dir: str | Path = "outputs", config: RoadVoidConfig | None = None) -> None:
    """生成路径示意图和叠加理论绕射曲线的单炮记录。"""

    config = config or load_config()
    output_dir = Path(output_dir)
    dataset = simulate_from_config(config)
    geom = dataset.geometry
    cavities = config.to_cavities()
    if not cavities:
        print("示例07跳过：当前配置未启用空洞，无法绘制绕射路径。")
        return
    cavity = cavities[0]
    shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavity.x0))
    channel_index = min(range(geom.n_channels), key=lambda i: abs(geom.channel_x[i] - cavity.x0))
    plot_diffraction_path_demo(
        geom,
        cavity,
        shot_index=shot_index,
        channel_index=channel_index,
        output=output_dir / "example_07_diffraction_path.png",
    )
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index=shot_index,
        direct_times=dataset.direct_times,
        diffraction_times=dataset.diffraction_times[0],
        title="直达波与绕射波理论曲线叠加",
        output=output_dir / "example_07_gather_with_curves.png",
    )
    print(f"示例07完成：路径图与走时曲线图已保存到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制直达路径、绕射路径和理论曲线")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.output_dir, load_config(args.config))
