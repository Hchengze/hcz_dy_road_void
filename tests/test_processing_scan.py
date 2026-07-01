import numpy as np

from road_void import (
    Cavity,
    CavityScanGrid,
    ForwardModelConfig,
    RayleighKinematicForwardModel,
    RoadGeometry,
    scan_cavity_diffraction,
)
from road_void.processing import fit_direct_velocity, mute_direct_wave, pick_direct_arrivals, trace_normalize


def test_direct_velocity_fit_recovers_effective_rayleigh_speed():
    geom = RoadGeometry.from_ranges(road_width=15, x_min=0, x_max=30, channel_spacing=2, shot_spacing=10, dt=0.001, t_max=0.7)
    cfg = ForwardModelConfig(rayleigh_velocity=250.0, t0=0.018, noise_std=0.0, traffic_noise_std=0.0, random_seed=2)
    ds = RayleighKinematicForwardModel(geom, cfg).simulate()
    picks = pick_direct_arrivals(ds.data, geom, velocity_hint=245.0, t0_hint=0.02, search_half_width=0.035)
    fit = fit_direct_velocity(picks, geom)
    assert abs(fit.velocity - 250.0) < 8.0
    assert abs(fit.t0 - 0.018) < 0.012


def test_cavity_scan_recovers_approximate_x_location():
    geom = RoadGeometry.from_ranges(road_width=15, x_min=0, x_max=50, channel_spacing=2, shot_spacing=5, dt=0.002, t_max=0.75)
    cavity = Cavity(x0=26.0, y0=8.0, h=2.0, radius=2.0, scattering_strength=1.4, attenuation_strength=0.1)
    cfg = ForwardModelConfig(rayleigh_velocity=240.0, t0=0.02, noise_std=0.004, traffic_noise_std=0.0, random_seed=3)
    ds = RayleighKinematicForwardModel(geom, cfg).simulate([cavity])
    residual = mute_direct_wave(ds.data, geom, 240.0, 0.02, half_width=0.035)
    residual = trace_normalize(residual)
    grid = CavityScanGrid(
        x=np.arange(18.0, 34.1, 2.0),
        y=np.arange(5.0, 12.1, 2.0),
        h=np.arange(1.0, 3.1, 1.0),
        velocity=np.array([230.0, 240.0, 250.0]),
    )
    result = scan_cavity_diffraction(residual, geom, grid, t0=0.02, half_window=0.014, top_k=5)
    assert abs(result.best.x0 - cavity.x0) <= 4.0
    assert result.best.score > 0
    assert result.uncertainty["x0"][0] <= result.best.x0 <= result.uncertainty["x0"][1]


def test_scan_modes_return_joint_and_single_shot_structures():
    geom = RoadGeometry.from_ranges(road_width=15, x_min=0, x_max=30, channel_spacing=3, shot_spacing=10, dt=0.002, t_max=0.65)
    cavity = Cavity(x0=16.0, y0=8.0, h=2.0, scattering_strength=1.2)
    cfg = ForwardModelConfig(rayleigh_velocity=240.0, t0=0.02, noise_std=0.002, traffic_noise_std=0.0, random_seed=6)
    ds = RayleighKinematicForwardModel(geom, cfg).simulate([cavity])
    residual = trace_normalize(mute_direct_wave(ds.data, geom, 240.0, 0.02, half_width=0.035))
    grid = CavityScanGrid(
        x=np.arange(10.0, 22.1, 4.0),
        y=np.arange(5.0, 11.1, 3.0),
        h=np.arange(1.0, 3.1, 1.0),
        velocity=np.array([230.0, 240.0]),
    )
    joint = scan_cavity_diffraction(residual, geom, grid, t0=0.02, scan_mode="joint", top_k=3)
    single = scan_cavity_diffraction(residual, geom, grid, t0=0.02, scan_mode="single-shot", shot_index=1, top_k=3)
    compare = scan_cavity_diffraction(residual, geom, grid, t0=0.02, scan_mode="compare", top_k=3)
    assert joint.scan_mode == "joint"
    assert single.scan_mode == "single-shot"
    assert compare.per_shot_best is not None
    assert len(compare.per_shot_best) == geom.n_shots
    assert compare.consistency is not None
