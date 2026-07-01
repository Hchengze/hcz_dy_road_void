"""无空洞道路 DAS 单炮记录正演示例。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void import ForwardModelConfig, RayleighKinematicForwardModel, RoadGeometry
from road_void.visualization import plot_shot_gather


def main() -> None:
    geom = RoadGeometry.typical_four_lane(x_min=0, x_max=80, channel_spacing=1.0, shot_spacing=4.0)
    cfg = ForwardModelConfig(rayleigh_velocity=240.0, noise_std=0.02, random_seed=11)
    dataset = RayleighKinematicForwardModel(geom, cfg).simulate()
    out = Path("outputs/example_01_no_cavity_gather.png")
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index=geom.n_shots // 2,
        direct_times=dataset.direct_times,
        title="无空洞三维道路几何：直达瑞雷波",
        output=out,
    )
    print(f"已保存 {out}")
    print(f"数据形状: {dataset.data.shape}")


if __name__ == "__main__":
    main()
