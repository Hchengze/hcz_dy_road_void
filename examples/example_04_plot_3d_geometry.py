"""示例 04：绘制三维道路几何、平面布设图和剖面图。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void.config import RoadVoidConfig, load_config
from road_void.visualization import plot_geometry_plan_and_sections, plot_road_geometry_3d


def main(output_dir: str | Path = "outputs", config: RoadVoidConfig | None = None) -> None:
    """生成适合汇报展示的道路三维几何图件。"""

    config = config or load_config()
    output_dir = Path(output_dir)
    geom = config.to_geometry()
    cavities = config.to_cavities()
    plot_road_geometry_3d(geom, cavities, output_dir / "example_04_3d_geometry.png")
    plot_geometry_plan_and_sections(geom, cavities, output_dir / "example_04_plan_sections.png")
    print(f"示例04完成：图件已保存到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制三维道路几何和平面/剖面图")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.output_dir, load_config(args.config))
