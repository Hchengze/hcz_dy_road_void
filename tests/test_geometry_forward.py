import numpy as np

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
