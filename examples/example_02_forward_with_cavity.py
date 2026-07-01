"""含三维空洞散射体的道路 DAS 单炮记录正演示例。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void import Cavity, ForwardModelConfig, RayleighKinematicForwardModel, RoadGeometry
from road_void.visualization import plot_shot_gather


def main() -> None:
    geom = RoadGeometry.typical_four_lane(x_min=0, x_max=80, channel_spacing=1.0, shot_spacing=4.0)
    cavity = Cavity(x0=42.0, y0=8.5, h=2.2, radius=2.0, scattering_strength=0.9, attenuation_strength=0.3)
    cfg = ForwardModelConfig(rayleigh_velocity=240.0, noise_std=0.02, random_seed=22)
    dataset = RayleighKinematicForwardModel(geom, cfg).simulate([cavity])
    shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavity.x0))
    out = Path("outputs/example_02_with_cavity_gather.png")
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index=shot_index,
        direct_times=dataset.direct_times,
        diffraction_times=dataset.diffraction_times[0],
        title="含空洞场景：直达波与三维绕射/散射响应",
        output=out,
    )
    print(f"已保存 {out}")
    print(f"真实空洞: x={cavity.x0:.1f} m, y={cavity.y0:.1f} m, h={cavity.h:.1f} m")


if __name__ == "__main__":
    main()
