# 参数组合回归测试策略

## 为什么默认参数测试不够

道路空洞 workflow 由几何、速度、异常体、噪声、扫描、DAS-like 响应、绕射属性和定位评估共同组成。默认参数通过只能说明一条路径可运行，不能证明以下组合也安全：

- `velocity_mode=uniform / layered-effective`；
- 不同 `layer_velocities` 和 `source_frequency`；
- `sphere / box / cylinder / ellipsoid / line / zone`；
- 多异常体 `--anomalies` 字符串；
- 扫描范围覆盖、贴边或不覆盖异常体；
- 高噪声、短记录、低散射强度；
- `wavefield_view=plan / 3d`；
- `save / no-save / clean-output`。

因此项目新增参数组合回归矩阵，用来捕获隐藏的 RuntimeWarning、NaN/inf、空数组、参数不同步和输出路径分裂。

## quick matrix

默认 pytest 运行 quick matrix，场景定义在 `road_void/test_scenarios.py`：

- `default_layered_sphere`
- `uniform_sphere`
- `layered_cylinder`
- `multi_anomaly`
- `scan_range_miss`
- `high_noise`

这些场景使用小几何和粗扫描网格，目标是快速检查主链条：

```text
RoadVoidConfig -> scenario -> synthetic dataset -> DAS-like -> diffraction -> joint localization
```

## extended matrix

扩展矩阵用 `@pytest.mark.slow` 标记，默认不运行。它覆盖更多 shape、短记录、宽道路等组合：

```bash
python -m pytest -q -m slow
```

## warning 检查

代码层 warning 不应污染正常运行。常用命令：

```bash
python -m pytest -q -W default
python -W default main.py workflow --no-save
```

以下属于代码问题，应修复根因：

- `RuntimeWarning`
- `DeprecationWarning`
- `FutureWarning`
- NumPy `divide by zero / invalid value / mean of empty slice`
- Matplotlib figure/animation warning

以下属于业务诊断，应进入控制台摘要或 `research_report.md`，不应通过 Python warning 刷屏：

- scan 范围没有覆盖异常体；
- 定位置信度低；
- `layered-effective` 不是完整分层弹性波场；
- DAS-like 不是完整仪器响应。

## 本地矩阵脚本

手动回归检查：

```bash
python tools/check_workflow_matrix.py --quick
python tools/check_workflow_matrix.py --extended
```

输出：

```text
outputs/workflow_matrix_report.md
```

报告包含每个场景的 pass/fail、warning 数量、运行耗时和简要说明。

## 每次 Codex 修改后的最低检查

涉及 workflow、参数、输出、正演、扫描、波场、dataset、diffraction 或 inversion 时，至少运行：

```bash
python -m compileall -q road_void examples main.py
python -m pytest -q
python -m pytest -q -W default
python tools/check_workflow_matrix.py --quick
```

如果修改影响 shape、记录长度、扫描边界或更复杂速度组合，再运行：

```bash
python -m pytest -q -m slow
```
