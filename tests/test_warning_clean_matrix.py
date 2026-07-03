from __future__ import annotations

import warnings

import pytest

from road_void.test_scenarios import QUICK_SCENARIOS, run_lightweight_workflow_case


PROJECT_WARNING_TYPES = (RuntimeWarning, DeprecationWarning, FutureWarning, UserWarning)


@pytest.mark.parametrize("scenario_name", QUICK_SCENARIOS)
def test_quick_matrix_has_no_project_python_warnings(scenario_name: str):
    """参数矩阵不应产生代码层 Python warning。

    scan 范围未覆盖、低置信度、DAS-like 近似等属于业务诊断，应进入报告或控制台摘要；
    这里捕获的是 RuntimeWarning/DeprecationWarning/FutureWarning/UserWarning 这类代码运行问题。
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_lightweight_workflow_case(scenario_name)

    unexpected = [
        f"{item.category.__name__}: {item.message}"
        for item in caught
        if issubclass(item.category, PROJECT_WARNING_TYPES)
    ]
    assert unexpected == []
