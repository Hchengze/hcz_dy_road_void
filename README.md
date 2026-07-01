# 城市道路空洞三维 DAS 正演与定位原型

本项目是一个面向城市道路空洞探测的研究原型，观测方式为“道路一侧既有 DAS 光纤接收 + 道路另一侧锤击逐点激发”。当前阶段重点是：

1. 建立三维道路场景和单侧 DAS 几何；
2. 生成可控的三维等效瑞雷面波合成数据；
3. 识别空洞/脱空/松散异常体引起的绕射与散射响应；
4. 通过三维源-空洞-接收走时扫描输出疑似异常范围；
5. 增强中文文档、注释、图件和教学展示能力。

它**不是**普通二维 MASW，也**不是**完整三维弹性波 FDTD。当前正演属于工程化的“三维等效瑞雷波运动学/属性原型”，适合先验证几何、走时、绕射扫描和结果解释逻辑。

## 方法总览

更完整的中文方法说明见：

- [docs/method_overview_zh.md](docs/method_overview_zh.md)
- [docs/parameter_guide_zh.md](docs/parameter_guide_zh.md)

核心流程如下：

```text
道路三维几何
  -> 锤击点 / DAS 通道 / 空洞参数
  -> 三维等效瑞雷波正演
  -> 合成 shot gather
  -> 直达波到时拾取
  -> 三维几何速度拟合
  -> 直达波 mute 或模板减去
  -> 三维绕射走时扫描 (x0, y0, h, VR)
  -> 评分体 / top-k 候选 / 不确定性范围
  -> 疑似异常解释与现场复核建议
```

## 坐标系统

- `x`：沿道路方向，也是沿 DAS 光纤方向；
- `y`：横穿道路方向；
- `z`：深度方向，向下为正；
- 震源点：`S_j = (x_j, W, 0)`；
- DAS 通道：`G_i = (x_i, 0, 0)`；
- 空洞或主要散射中心：`D = (x0, y0, h)`。

直达波和绕射波走时均使用三维坐标计算，保留道路横向孔径，不把观测系统简化为二维共线测线。

## 主要模块

- `road_void/geometry.py`：`RoadGeometry`，道路宽度、DAS 通道、锤击点和三维走时几何。
- `road_void/anomaly.py`：`Cavity`，空洞/脱空/松散异常体的有效散射模型。
- `road_void/forward.py`：`RayleighKinematicForwardModel`，三维等效瑞雷波正演。
- `road_void/processing.py`：带通、归一化、包络、直达波拾取、速度拟合、直达波压制。
- `road_void/scan.py`：`scan_cavity_diffraction`，三维绕射走时扫描和不确定性估计。
- `road_void/io.py`：真实 DAS 数据接入、通道坐标映射和预处理预留接口。
- `road_void/velocity.py`：教学展示用的简化分层等效瑞雷波速度模型。
- `road_void/visualization.py`：场景图、速度模型图、shot gather、评分图和运动学动画。

## 运行入口

统一入口支持以下命令：

```bash
python main.py no-cavity
python main.py with-cavity
python main.py scan
python main.py geometry
python main.py velocity
python main.py wavefield
python main.py path
python main.py tutorial
python main.py all
```

也兼容：

```bash
python main.py --case scan
```

默认输出目录为 `outputs/`，可通过 `--output-dir` 指定：

```bash
python main.py tutorial --config configs/default_road_void.yaml --output-dir outputs/tutorial_demo
```

## 如何修改参数

推荐从 `configs/default_road_void.yaml` 开始复制和修改。配置文件使用 YAML，可写中文注释，主要分为：

- `geometry`：道路宽度、道路长度、DAS 通道间距、锤击点间距；
- `cavity`：是否启用空洞、空洞位置、深度、尺度、散射强度；
- `velocity`：等效瑞雷波速度、速度模型类型、锤击主频、子波类型；
- `record`：采样率、记录时长、触发时延、随机种子；
- `noise`：随机噪声、交通噪声、坏道、弱耦合和通道增益变化；
- `processing`：直达波压制、扫描范围、扫描步长、top-k 和不确定性阈值。

常用运行方式：

```bash
python main.py geometry --config configs/four_lane_demo.yaml
python main.py velocity --config configs/default_road_void.yaml
python main.py wavefield --config configs/high_noise_demo.yaml
python main.py path --config configs/default_road_void.yaml
python main.py scan --config configs/deep_cavity_demo.yaml
python main.py tutorial --config configs/default_road_void.yaml
python main.py all --config configs/default_road_void.yaml
```

常用命令行覆盖参数：

```bash
python main.py scan --config configs/default_road_void.yaml --road-width 30 --cavity-depth 2.5 --noise-level 0.08
python main.py scan --config configs/default_road_void.yaml --rayleigh-velocity 260
```

这些覆盖参数主要用于快速实验；更完整的参数修改建议直接编辑 YAML。

## 配置模板

- `configs/default_road_void.yaml`：默认四车道空洞定位闭环；
- `configs/four_lane_demo.yaml`：四车道道路示例；
- `configs/six_lane_demo.yaml`：六车道道路示例；
- `configs/shallow_cavity_demo.yaml`：浅埋异常体示例；
- `configs/deep_cavity_demo.yaml`：深埋异常体示例；
- `configs/high_noise_demo.yaml`：高噪声和弱耦合示例；
- `configs/no_cavity_demo.yaml`：无空洞误报风险检查示例。

真实数据接入前，以下参数必须实测或标定：`road_width`、光纤通道坐标、锤击点坐标、触发时间 `t0`、等效瑞雷波速度 `rayleigh_velocity`、通道耦合质量、坏道和 gauge length。

## 参数敏感性分析

运行：

```bash
python examples/example_09_parameter_sensitivity.py --config configs/default_road_void.yaml
```

输出：

```text
outputs/sensitivity/parameter_sensitivity_results.csv
outputs/sensitivity/noise_vs_confidence.png
outputs/sensitivity/vr_vs_h_error.png
outputs/sensitivity/road_width_vs_confidence.png
```

读图方式：

- `noise_level vs confidence`：噪声越大，置信度通常越低；若高噪声仍高置信，应警惕误报；
- `rayleigh_velocity vs h_error`：速度偏差会系统影响深度解释；
- `road_width vs confidence`：道路越宽，路径越长，受限孔径不确定性通常更明显。

半交互式参数实验：

```bash
python examples/example_10_interactive_parameter_demo.py --config configs/default_road_void.yaml --road-width 30 --cavity-depth 2.5 --noise-level 0.1
```

## 示例脚本说明

- `examples/example_01_forward_no_cavity.py`：无空洞正演，只展示直达瑞雷波。
- `examples/example_02_forward_with_cavity.py`：含空洞正演，展示直达波与三维绕射/散射响应。
- `examples/example_03_cavity_location_scan.py`：完整定位扫描，输出原始记录、残差记录、最佳曲线和评分图。
- `examples/example_04_plot_3d_geometry.py`：三维道路几何图、x-y 平面布设图、x-z/y-z 剖面图。
- `examples/example_05_plot_velocity_model.py`：简化分层等效瑞雷波速度模型图。
- `examples/example_06_wavefield_animation.py`：等效运动学波场 GIF，展示直达波前与空洞散射波前。
- `examples/example_07_diffraction_path_demo.py`：直达路径、源-空洞-接收绕射路径和 shot gather 理论曲线。
- `examples/example_08_full_workflow_tutorial.py`：教学型完整流程，从建模、正演、速度拟合到定位解释。
- `examples/example_09_parameter_sensitivity.py`：参数敏感性分析，输出 CSV 和曲线图。
- `examples/example_10_interactive_parameter_demo.py`：半交互式参数实验，用命令行快速改参数并重跑。

## 如何读图

- 三维场景图：确认光纤、锤击线、道路宽度和空洞在正确三维位置；
- 平面布设图：理解单侧 DAS 与对侧锤击形成的横向孔径；
- 剖面图：理解空洞埋深和道路浅层结构；
- 速度模型图：理解当前使用的是简化等效速度，而不是完整弹性速度模型；
- shot gather：观察直达波同相轴和空洞绕射/散射事件；
- 残差 gather：观察直达波削弱后异常事件是否更清楚；
- x-y / x-h / y-h 评分图：观察疑似异常高分区、top-k 候选和真值/最佳点关系；
- 波场 GIF：用于理解传播路径和绕射现象，不代表严格弹性波场快照。

## 输出文件

常见输出包括：

- `outputs/example_04_3d_geometry.png`
- `outputs/example_04_plan_sections.png`
- `outputs/example_05_velocity_model.png`
- `outputs/example_06_kinematic_wavefield.gif`
- `outputs/example_07_diffraction_path.png`
- `outputs/example_07_gather_with_curves.png`
- `outputs/example_03_raw_gather.png`
- `outputs/example_03_residual_gather.png`
- `outputs/example_03_residual_best_curve.png`
- `outputs/example_03_scan_scores.png`
- `outputs/tutorial/01_3d_geometry.png` 等教学流程图件

## 运行测试

```bash
python -m pytest
```

## 中文字体说明

`road_void/visualization.py` 会优先尝试加载 Windows 常见中文字体：微软雅黑、黑体、宋体。如果在其他系统上图中文字显示为方框，可以安装或配置支持中文的 matplotlib 字体，例如 Noto Sans CJK 或 Source Han Sans。

## 结果解释原则

当前阶段输出应表述为：

- 疑似异常体位置范围；
- 评分或置信度；
- 不确定性范围；
- 建议复核区域。

不应直接表述为“确诊空洞”。真实道路数据应用前，需要完成光纤路径标定、锤击触发校正、浅层速度估计、通道耦合 QC、gauge length 影响评估，以及管线、井盖、交通和施工干扰核查。

## 当前局限与下一步

当前模型计算快、可解释、便于教学展示，但物理真实性有限。后续若要进一步逼近真实波场，建议扩展：

1. 2.5D/3D 弹性波正演；
2. 频散建模和多频带约束；
3. FK/扇形滤波、预测滤波等绕射增强；
4. 多炮联合反演和不确定性量化；
5. 真实 DAS 数据接入、触发校正和光纤路径标定；
6. 与 GPR、钻探、管线资料和道路结构资料联合解释。
