"""运行道路空洞 workflow 参数组合回归矩阵。

用法：
    python tools/check_workflow_matrix.py --quick
    python tools/check_workflow_matrix.py --extended

脚本只做本地回归检查：运行轻量场景、捕获 Python warning、写出 Markdown 报告。
它不替代 pytest，也不改变默认 workflow 输出。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from road_void.test_scenarios import QUICK_SCENARIOS, EXTENDED_SCENARIOS, scenario_by_name, run_lightweight_workflow_case


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 workflow 参数组合回归矩阵并生成 Markdown 报告。")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="只运行默认 quick matrix。")
    group.add_argument("--extended", action="store_true", help="运行包含 slow 场景的 extended matrix。")
    parser.add_argument("--output", default="outputs/workflow_matrix_report.md", help="Markdown 报告输出路径。")
    args = parser.parse_args()

    names = EXTENDED_SCENARIOS if args.extended else QUICK_SCENARIOS
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for name in names:
        scenario = scenario_by_name(name)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = run_lightweight_workflow_case(name)
                status = "PASS"
                message = f"confidence={result['localization'].confidence_score:.4g}"
            except Exception as exc:  # noqa: BLE001 - 本地检查脚本需要把失败写入报告。
                status = "FAIL"
                message = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        rows.append(
            {
                "name": name,
                "expected": scenario.expected,
                "status": status,
                "warnings": len(caught),
                "message": message,
                "seconds": time.perf_counter() - t0,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_report(rows, time.perf_counter() - started), encoding="utf-8")
    print(f"workflow matrix report: {output}")
    failed = [row for row in rows if row["status"] != "PASS"]
    return 1 if failed else 0


def _render_report(rows: list[dict[str, object]], seconds: float) -> str:
    lines = [
        "# Workflow Parameter Matrix Report",
        "",
        f"- total_scenarios: {len(rows)}",
        f"- elapsed_seconds: {seconds:.2f}",
        "",
        "| scenario | expected behavior | status | warning count | note | seconds |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {expected} | {status} | {warnings} | {message} | {seconds:.2f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "说明：warning count 捕获的是 Python warning。scan range miss、低置信度、DAS-like 近似等属于业务诊断，应进入 report/控制台说明，不应作为 Python warning 刷屏。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
