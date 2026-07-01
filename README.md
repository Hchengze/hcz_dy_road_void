# 城市道路空洞三维 DAS 正演与定位研究原型

这是一个本地算法研究原型，用于探索“道路一侧 DAS 光纤接收 + 道路另一侧锤击激发”的城市道路空洞/脱空/松散异常体探测方法。项目目标是便于直接运行、修改参数、查看图件、理解公式和验证算法闭环；不追求发布、安装或生产级工程化。

当前核心仍是三维等效瑞雷波运动学/属性正演，不是完整三维弹性波全波形模拟。

## 推荐入口

日常使用优先运行 `main.py`：

```bash
python main.py geometry --show --no-save
python main.py forward --show --save
python main.py scan --road-width 30 --cavity-depth 2.5 --noise-level 0.1 --save
python main.py tutorial --save
```

主要子命令：

- `geometry`：只画道路、光纤、锤击炮线、空洞位置、平面图和剖面图。
- `forward`：只做三维等效瑞雷波正演并画 shot gather。
- `velocity`：展示等效速度/分层速度模型说明。
- `wavefield`：生成等效运动学波场示意，可选 GIF。
- `path`：展示直达路径、绕射路径、公式和理论曲线。
- `scan`：做绕射扫描定位，输出最佳候选和评分图。
- `sensitivity`：做参数敏感性分析，输出少量趋势图和 CSV。
- `tutorial`：生成一套不重复的教学流程图。
- `all`：快速总览，只输出代表性的几何、正演和扫描图，避免输出爆炸。

## 图件保存与交互显示

所有绘图入口支持：

```bash
--save       保存图件
--no-save    不保存图件
--show       弹出 matplotlib 交互窗口
--no-show    不显示窗口
--outdir     指定输出目录
--dpi        指定保存分辨率
```

默认不主动保存大量图件。建议本地调参时使用：

```bash
python main.py scan --show --no-save
```

批量生成汇报图时使用：

```bash
python main.py tutorial --save --outdir outputs/tutorial
```

默认输出目录按功能分组：

- `outputs/geometry/`
- `outputs/forward/`
- `outputs/velocity/`
- `outputs/wavefield/`
- `outputs/path/`
- `outputs/scan/`
- `outputs/tutorial/`
- `outputs/sensitivity/`
- `outputs/overview/`

## 常用参数

几何参数：

```bash
python main.py geometry --road-width 24 --road-length 100 --channel-spacing 1 --source-spacing 5 --show --no-save
```

正演参数：

```bash
python main.py forward --rayleigh-velocity 240 --source-frequency 35 --noise-level 0.03 --save
```

扫描参数：

```bash
python main.py scan --scan-x-min 30 --scan-x-max 55 --scan-h-min 0.5 --scan-h-max 5 --scan-h-step 0.3 --save
```

参数含义详见：

- [docs/parameter_guide_zh.md](docs/parameter_guide_zh.md)
- [docs/method_overview_zh.md](docs/method_overview_zh.md)

## 关键公式

直达瑞雷波：

```text
t_direct = t0 + distance(S, G) / VR
```

空洞绕射/散射波：

```text
t_diff = t0 + [distance(S, D) + distance(D, G)] / VR
```

锤击主频和波长：

```text
lambda = VR / f
```

其中 `VR` 是某一频带内的等效瑞雷波速度，不是完整随深度变化的速度模型。`VR` 偏大可能导致深度解释偏深，`VR` 偏小可能导致深度解释偏浅。

## examples 说明

日常使用请优先用 `main.py`。`examples/` 只保留少量教学样例：

- `example_minimal_forward.py`：最小正演示例，薄包装 `main.py forward`。
- `example_minimal_scan.py`：最小定位示例，薄包装 `main.py scan`。
- `example_parameter_sensitivity.py`：参数敏感性分析逻辑，被 `main.py sensitivity` 调用。

不再维护大量重复的 `example_04/example_08` 等入口，避免一个参数要改两套代码。

## 参数敏感性分析

```bash
python main.py sensitivity --save
```

输出：

```text
outputs/sensitivity/parameter_sensitivity_results.csv
outputs/sensitivity/noise_vs_confidence.png
outputs/sensitivity/vr_vs_h_error.png
outputs/sensitivity/road_width_vs_confidence.png
```

用于观察道路宽度、空洞深度、速度误差、噪声等参数对定位结果和置信度的影响。

## 运行测试

```bash
python -m compileall -q road_void examples main.py
python -m pytest -q
```

## 关于 YAML 配置

早期版本保留了 `configs/` 和 `road_void/config.py` 中的 YAML 读取能力，主要作为历史兼容和参数记录参考。当前推荐方式是直接通过 `main.py` 的子命令参数修改实验，不再把 YAML 作为主入口。

## 结果解释原则

当前输出只能解释为：

- 疑似异常范围；
- 相对评分和置信度；
- 不确定性范围；
- 建议复核区域。

不能直接确诊空洞。真实道路 DAS 数据接入前，还需要完成触发校正、光纤路径标定、通道耦合 QC、gauge length 影响评估、浅层速度估计，以及管线、井盖、交通和施工干扰核查。

## 当前限制与下一步

- 波场 GIF 是等效运动学传播示意，不是高保真弹性波场快照。
- `velocity.py` 的层状速度模型目前主要用于教学展示；核心正演/扫描仍使用等效 `VR`。
- 后续建议扩展：多频带速度约束、FK/扇形滤波、预测滤波绕射增强、多炮联合反演、真实 DAS 触发校正、光纤路径标定和通道耦合 QC。
