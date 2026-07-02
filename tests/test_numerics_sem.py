import numpy as np

from road_void.numerics.sem import gll_nodes_weights, lagrange_derivative_matrix, run_sem1d_wave_demo
from road_void.numerics.validation import check_array_finite, check_energy_not_exploding


def test_gll_nodes_weights_basic_properties():
    nodes, weights = gll_nodes_weights(order=4)
    assert np.isclose(nodes[0], -1.0)
    assert np.isclose(nodes[-1], 1.0)
    assert np.all(weights > 0)
    assert np.isclose(np.sum(weights), 2.0)
    dmat = lagrange_derivative_matrix(nodes)
    assert dmat.shape == (5, 5)


def test_sem1d_wave_demo_runs():
    result = run_sem1d_wave_demo(n_elements=4, order=3, length=40.0, velocity=200.0, duration=0.06)
    assert result.snapshots.shape[0] == 5
    assert result.receiver_trace.ndim == 1
    check_array_finite(result.receiver_trace, "sem receiver")
    check_energy_not_exploding(result.receiver_trace, growth_limit=1.0e6)
