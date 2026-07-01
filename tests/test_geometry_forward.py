import numpy as np

from main import build_parser, config_from_args
from road_void import Cavity, ForwardModelConfig, RayleighKinematicForwardModel, RoadGeometry


def test_geometry_distances_and_direct_times_use_lateral_offset():
    geom = RoadGeometry(road_width=15.0, channel_x=np.array([0.0]), shot_x=np.array([0.0]), dt=0.001, t_max=0.5)
    distances = geom.source_receiver_distances()
    assert distances.shape == (1, 1)
    assert np.isclose(distances[0, 0], 15.0)
    assert np.isclose(geom.direct_times(300.0, t0=0.01)[0, 0], 0.06)


def test_diffraction_times_are_source_to_cavity_to_receiver_3d():
    geom = RoadGeometry(road_width=10.0, channel_x=np.array([0.0]), shot_x=np.array([0.0]), dt=0.001, t_max=0.5)
    t = geom.diffraction_times((0.0, 5.0, 3.0), velocity=200.0, t0=0.0)
    expected = 2.0 * np.sqrt(5.0**2 + 3.0**2) / 200.0
    assert np.isclose(t[0, 0], expected)


def test_forward_shapes_no_cavity_and_with_cavity():
    geom = RoadGeometry.from_ranges(road_width=15.0, x_min=0, x_max=20, channel_spacing=2, shot_spacing=5, dt=0.002, t_max=0.6)
    cfg = ForwardModelConfig(noise_std=0.0, traffic_noise_std=0.0, random_seed=1)
    no_cavity = RayleighKinematicForwardModel(geom, cfg).simulate()
    with_cavity = RayleighKinematicForwardModel(geom, cfg).simulate([Cavity(x0=10, y0=7, h=2)])
    assert no_cavity.data.shape == (geom.n_shots, geom.n_times, geom.n_channels)
    assert with_cavity.data.shape == no_cavity.data.shape
    assert len(with_cavity.diffraction_times) == 1


def test_layered_effective_velocity_changes_theoretical_times():
    parser = build_parser()
    uniform_args = parser.parse_args(["forward", "--velocity-mode", "uniform", "--no-save"])
    layered_args = parser.parse_args(["forward", "--velocity-mode", "layered-effective", "--layer-velocities", "180,240,320", "--no-save"])
    uniform_cfg = config_from_args(uniform_args)
    layered_cfg = config_from_args(layered_args)
    geom = uniform_cfg.to_geometry()
    t_uniform = geom.direct_times(uniform_cfg.effective_rayleigh_velocity(), uniform_cfg.record.t0)
    t_layered = geom.direct_times(layered_cfg.effective_rayleigh_velocity(), layered_cfg.record.t0)
    assert not np.allclose(t_uniform, t_layered)


def test_layer_velocity_changes_effective_travel_times():
    parser = build_parser()
    slow_args = parser.parse_args(["forward", "--velocity-mode", "layered-effective", "--layer-velocities", "160,210,260", "--no-save"])
    fast_args = parser.parse_args(["forward", "--velocity-mode", "layered-effective", "--layer-velocities", "260,320,420", "--no-save"])
    slow_cfg = config_from_args(slow_args)
    fast_cfg = config_from_args(fast_args)
    assert slow_cfg.effective_rayleigh_velocity() != fast_cfg.effective_rayleigh_velocity()


def test_anomalies_string_parses_multiple_shapes():
    parser = build_parser()
    args = parser.parse_args([
        "forward",
        "--anomalies",
        "sphere:42,8.5,2.2,2.0,1.0;box:58,6,1.5,4,3,1,0.8",
        "--no-save",
    ])
    cavities = config_from_args(args).to_cavities()
    assert len(cavities) == 2
    assert {c.shape for c in cavities} == {"sphere", "box"}


def test_anomaly_shapes_generate_scatter_points():
    for shape in ["sphere", "box", "cylinder", "ellipsoid", "line"]:
        cavity = Cavity(x0=10, y0=5, h=2, radius=2, shape=shape, size_x=4, size_y=3, size_z=1)
        points, weights = cavity.scatter_points()
        assert points.shape[0] >= 2
        assert points.shape[1] == 3
        assert np.isclose(np.sum(weights), 1.0)


def test_multi_anomaly_forward_differs_from_single_anomaly():
    geom = RoadGeometry.from_ranges(road_width=15.0, x_min=0, x_max=30, channel_spacing=3, shot_spacing=10, dt=0.002, t_max=0.6)
    cfg = ForwardModelConfig(noise_std=0.0, traffic_noise_std=0.0, random_seed=5)
    one = [Cavity(x0=12, y0=7, h=2, scattering_strength=1.0)]
    two = [*one, Cavity(x0=22, y0=5, h=1.5, shape="box", size_x=4, size_y=3, size_z=1, scattering_strength=0.8)]
    ds_one = RayleighKinematicForwardModel(geom, cfg).simulate(one)
    ds_two = RayleighKinematicForwardModel(geom, cfg).simulate(two)
    assert not np.allclose(ds_one.data, ds_two.data)
