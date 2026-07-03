from __future__ import annotations

import warnings

import pytest

from road_void.test_scenarios import SCENARIO_MATRIX, run_lightweight_workflow_case


SLOW_SCENARIOS = [scenario.name for scenario in SCENARIO_MATRIX if scenario.slow]


@pytest.mark.slow
@pytest.mark.parametrize("scenario_name", SLOW_SCENARIOS)
def test_extended_parameter_matrix_scenarios(scenario_name: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_lightweight_workflow_case(scenario_name)
    assert result["localization"].best_estimate
    assert not [w for w in caught if issubclass(w.category, (RuntimeWarning, DeprecationWarning, FutureWarning))]
