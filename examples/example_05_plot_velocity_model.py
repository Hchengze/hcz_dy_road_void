"""示例 05：绘制简化等效瑞雷波速度模型。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void.config import RoadVoidConfig, load_config
from road_void.visualization import plot_velocity_model


def main(output_dir: str | Path = "outputs", config: RoadVoidConfig | None = None) -> None:
    """展示当前原型可使用的简化分层速度模型。"""

    config = config or load_config()
    output_dir = Path(output_dir)
    geom = config.to_geometry()
    model = config.to_velocity_model()
    plot_velocity_model(
        model,
        x_range=(float(geom.channel_x[0]), float(geom.channel_x[-1])),
        cavities=config.to_cavities(),
        output=output_dir / "example_05_velocity_model.png",
    )
    print(f"示例05完成：速度模型图已保存到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制简化等效瑞雷波速度模型")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    main(args.output_dir, load_config(args.config))
