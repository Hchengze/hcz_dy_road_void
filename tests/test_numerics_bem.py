import numpy as np

from road_void.numerics.bem import run_bem2d_scatter_demo
from road_void.numerics.validation import check_array_finite


def test_bem2d_scatter_demo_runs():
    result = run_bem2d_scatter_demo(n_boundary=24, n_receivers=30)
    assert result.boundary.shape == (24, 2)
    assert result.receivers.shape == (30, 2)
    assert result.scattered_response.shape == (30,)
    check_array_finite(np.real(result.scattered_response), "bem real")
    assert np.max(np.abs(result.scattered_response)) > 0
