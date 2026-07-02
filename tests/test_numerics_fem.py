import numpy as np

from road_void.numerics.fem import assemble_fem1d_matrices, run_fem1d_wave_demo
from road_void.numerics.validation import check_array_finite, check_energy_not_exploding


def test_fem1d_matrix_assembly_shapes_and_symmetry():
    x, mass, stiffness = assemble_fem1d_matrices(n_nodes=11, length=10.0, velocity=2.0)
    assert x.shape == (11,)
    assert mass.shape == (11, 11)
    assert stiffness.shape == (11, 11)
    assert np.allclose(mass, mass.T)
    assert np.allclose(stiffness, stiffness.T)


def test_fem1d_wave_demo_runs():
    result = run_fem1d_wave_demo(n_nodes=41, length=40.0, velocity=200.0, duration=0.08)
    assert result.snapshots.shape[0] == 5
    assert result.receiver_trace.ndim == 1
    check_array_finite(result.receiver_trace, "fem receiver")
    check_energy_not_exploding(result.receiver_trace, growth_limit=1.0e6)
