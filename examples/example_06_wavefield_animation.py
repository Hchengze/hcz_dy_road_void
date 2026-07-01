"""示例 06：生成等效运动学波场传播动画。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void.config import RoadVoidConfig, load_config
from road_void.visualization import animate_kinematic_wavefield


def main(output_dir: str | Path = "outputs", config: RoadVoidConfig | None = None) -> None:
    """输出 GIF，解释直达波前与空洞散射波前的几何关系。"""

    config = config or load_config()
    output_dir = Path(output_dir)
    geom = config.to_geometry()
    cavities = config.to_cavities()
    if not cavities:
        print("示例06跳过：当前配置未启用空洞，无法展示空洞散射波前。")
        return
    cavity = cavities[0]
    source_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavity.x0))
    out = output_dir / "example_06_kinematic_wavefield.gif"
    animate_kinematic_wavefield(geom, cavity, source_index=source_index, velocity=config.velocity.rayleigh_velocity, output=out)
    print(f"示例06完成：等效运动学波场动画已保存为 {out}")
    print("说明：该动画是运动学传播示意，不是严格弹性波场快照。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成等效运动学波场 GIF")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.output_dir, load_config(args.config))
