import numpy as np
import pytest

from road_void.numerics.validation import (
    check_array_finite,
    check_energy_not_exploding,
    compare_traces_l2,
    estimate_arrival_time,
)


def test_validation_tools_work():
    trace = np.array([0.0, 0.1, 1.0, 0.2])
    check_array_finite(trace, "trace")
    check_energy_not_exploding(trace, growth_limit=10.0)
    assert estimate_arrival_time(trace, dt=0.01, threshold_fraction=0.5) == 0.02
    assert compare_traces_l2(trace, trace) == 0.0


def test_validation_rejects_nan_and_exploding_energy():
    with pytest.raises(ValueError):
        check_array_finite(np.array([np.nan]))
    with pytest.raises(ValueError):
        check_energy_not_exploding(np.array([1.0e6]), growth_limit=1.0)
