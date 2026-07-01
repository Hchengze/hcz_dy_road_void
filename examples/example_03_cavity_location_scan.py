"""通过三维绕射走时扫描定位合成道路空洞。"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_void import (
    Cavity,
    CavityScanGrid,
    ForwardModelConfig,
    RayleighKinematicForwardModel,
    RoadGeometry,
    scan_cavity_diffraction,
)
from road_void.processing import (
    fit_direct_velocity,
    pick_direct_arrivals,
    subtract_direct_wave_template,
    trace_normalize,
)
from road_void.visualization import plot_score_slices, plot_shot_gather


def main() -> None:
    geom = RoadGeometry.typical_four_lane(x_min=0, x_max=80, channel_spacing=1.0, shot_spacing=4.0)
    true_cavity = Cavity(x0=42.0, y0=8.5, h=2.2, radius=2.0, scattering_strength=1.0, attenuation_strength=0.25)
    cfg = ForwardModelConfig(rayleigh_velocity=240.0, noise_std=0.015, coda_amplitude=0.05, random_seed=33)
    dataset = RayleighKinematicForwardModel(geom, cfg).simulate([true_cavity])
    out_dir = Path("outputs")
    shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - true_cavity.x0))
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index=shot_index,
        direct_times=dataset.direct_times,
        diffraction_times=dataset.diffraction_times[0],
        title="原始合成记录：直达波与空洞绕射",
        output=out_dir / "example_03_raw_gather.png",
    )

    picks = pick_direct_arrivals(dataset.data, geom, velocity_hint=230.0, t0_hint=0.02)
    fit = fit_direct_velocity(picks, geom)
    residual = subtract_direct_wave_template(
        dataset.data,
        geom,
        fit.velocity,
        fit.t0,
        frequency=cfg.source_frequency,
        wavelet=cfg.wavelet,
        fit_half_width=0.035,
    )
    residual = trace_normalize(residual)
    plot_shot_gather(
        residual,
        geom,
        shot_index=shot_index,
        title="直达波模板减去后的残差记录",
        output=out_dir / "example_03_residual_gather.png",
    )

    grid = CavityScanGrid(
        x=np.arange(32.0, 52.1, 1.0),
        y=np.arange(3.0, 14.1, 1.0),
        h=np.arange(0.8, 4.1, 0.4),
        velocity=np.arange(220.0, 261.0, 10.0),
    )
    result = scan_cavity_diffraction(residual, geom, grid, t0=fit.t0, half_window=0.012, top_k=8)

    best_times = geom.diffraction_times((result.best.x0, result.best.y0, result.best.h), result.best.velocity, fit.t0)
    plot_shot_gather(
        residual,
        geom,
        shot_index=shot_index,
        diffraction_times=best_times,
        title="直达波压制残差与最佳三维绕射曲线",
        output=out_dir / "example_03_residual_best_curve.png",
    )
    plot_score_slices(
        result,
        true_x=true_cavity.x0,
        true_y=true_cavity.y0,
        true_h=true_cavity.h,
        output=out_dir / "example_03_scan_scores.png",
    )

    print(f"直达波估计 VR={fit.velocity:.1f} m/s, t0={fit.t0:.4f} s, RMS={fit.residual_rms:.4f} s")
    print(
        "最佳疑似异常体: "
        f"x={result.best.x0:.1f} m, y={result.best.y0:.1f} m, "
        f"h={result.best.h:.1f} m, VR={result.best.velocity:.1f} m/s, "
        f"score={result.best.score:.4f}"
    )
    print(f"不确定性范围: {result.uncertainty}")
    print(f"图片已保存到 {out_dir}")


if __name__ == "__main__":
    main()
