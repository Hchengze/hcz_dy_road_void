# 城市道路空洞三维 DAS 正演与定位研究原型

这是一个本地算法研究原型，用于探索“道路一侧 DAS 光纤接收 + 道路另一侧锤击激发”的城市道路空洞/脱空/松散异常体探测方法。项目目标是便于直接运行、修改参数、查看图件、理解公式和验证算法闭环；不追求发布、安装或生产级工程化。

当前有两条正演路线：

- 默认 workflow 使用快速三维等效瑞雷波运动学/属性正演，用于几何验证、合成数据和绕射扫描定位。
- `elastic3d` 子命令提供小尺度三维弹性波全波形有限差分原型，用于理解更接近真实波场的传播、散射和接收现象。

默认 `python main.py` 仍运行快速 workflow；`elastic3d` 是独立实验模块，当前不替代绕射扫描定位主线。

## 推荐入口

日常使用优先直接运行完整 workflow：

```bash
python main.py
```

这等价于：

```bash
python main.py workflow
```

`workflow` 会按“建模几何 -> 速度/频率说明 -> 正演 -> 路径公式 -> 绕射扫描 -> 结果解释”的算法顺序执行一遍，并且在同一上下文中只正演一次、扫描一次，避免重复生成大量图件。

常用 workflow 运行方式：

```bash
python main.py workflow --show --no-save
python main.py workflow --save --no-show
python main.py workflow --road-width 30 --cavity-depth 2.5 --noise-level 0.1 --save
python main.py workflow --animate --save
```

单独调试某一环节时，再运行对应子命令：

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
- `elastic3d`：运行小尺度三维弹性波全波形有限差分原型，输出速度切片、波场快照和接收记录。
- `workflow`：默认完整流程入口，控制台按算法步骤解释，输出一套代表性图件。
- `all`：当前作为 `workflow` 的别名，保留给习惯使用 `all` 的场景；日常推荐使用 `workflow` 或直接 `python main.py`。

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
- `outputs/workflow/`
- `outputs/sensitivity/`

`workflow` 默认保存文件名采用步骤编号，便于按算法顺序阅读：

```text
outputs/workflow/01_geometry_plan_sections.png
outputs/workflow/01_geometry_3d.png
outputs/workflow/02_velocity_model.png
outputs/workflow/03_forward_gather.png
outputs/workflow/04_diffraction_path.png
outputs/workflow/04_gather_with_curves.png
outputs/workflow/05_residual_best_curve.png
outputs/workflow/05_scan_score_slices.png
outputs/workflow/06_kinematic_wavefield.gif  # 仅 --animate --save 时生成
```

默认 `workflow` 不生成 GIF，避免每次运行输出过多。需要查看传播过程时：

```bash
python main.py workflow --animate --save
python main.py wavefield --animate --save
```

如果运行 `python main.py wavefield --save` 而不加 `--animate`，会输出三张关键帧：

```text
outputs/wavefield/wavefield_frame_early.png
outputs/wavefield/wavefield_frame_hit_cavity.png
outputs/wavefield/wavefield_frame_scattered.png
```

这些图和 GIF 都是“等效运动学传播示意”，用于理解直达波前、异常体散射触发时刻和 DAS 接收线位置，不是完整弹性波场快照。当前实现中散射波只有在直达波从震源传播到异常体之后才出现，三张关键帧含义为：

- `early`：直达波刚离开震源，异常体散射不应出现；
- `hit_cavity`：直达波前接近或到达第一个异常体；
- `scattered`：异常体散射波已经从异常体位置向外传播。

## 常用参数

几何参数：

```bash
python main.py geometry --road-width 24 --road-length 100 --channel-spacing 1 --source-spacing 5 --show --no-save
```

正演参数：

```bash
python main.py forward --rayleigh-velocity 240 --source-frequency 35 --noise-level 0.03 --save
python main.py forward --velocity-mode layered-effective --layer-depths 0.4,1.5,4.0 --layer-velocities 180,240,320 --save
```

速度模式：

- `--velocity-mode uniform`：保持原始单一等效速度，走时公式使用 `VR`。
- `--velocity-mode layered-effective`：根据 `lambda=VR/f` 和 `z_sensitive≈alpha*lambda` 对层状速度做指数权重调和平均，得到 `VR_eff`，并让正演和扫描都使用这个有效速度。

上一阶段的层状速度图主要用于展示，因此三层模型对 shot gather 影响不明显；当前 `layered-effective` 会让层速度通过 `VR_eff` 进入直达波和绕射波走时。但它仍是轻量近似，不是完整 Rayleigh 频散反演，也不是三维弹性波全波形正演。

扫描参数：

```bash
python main.py scan --scan-x-min 30 --scan-x-max 55 --scan-h-min 0.5 --scan-h-max 5 --scan-h-step 0.3 --save
python main.py scan --scan-mode joint --save
python main.py scan --scan-mode single-shot --shot-index 5 --save
python main.py scan --scan-mode compare --save
```

多炮联合扫描：

- `joint`：默认模式，多炮联合评分。
- `single-shot`：只用指定炮，便于观察单炮孔径下的不稳定性。
- `compare`：同时保留每炮最佳候选和多炮联合结果。

输出的 `per_shot_best_x.png`、`per_shot_score_contribution.png`、`single_shot_vs_joint.png` 可用于观察多炮联合是否让 `x0` 更稳定。注意：单侧 DAS + 对侧锤击下，`y0-h` 耦合仍然存在，不能因为多炮联合就把横向位置和深度解释成唯一精确值。

多异常体输入示例：

```bash
python main.py workflow --anomalies "sphere:42,8.5,2.2,2.0,1.0;box:58,6,1.5,4,3,1,0.8" --save
```

支持的 shape：

- `sphere:x,y,h,radius,strength`
- `box:x,y,h,size_x,size_y,size_z,strength`
- `cylinder:x,y,h,radius,height,strength`
- `ellipsoid:x,y,h,size_x,size_y,size_z,strength`
- `line:x,y,h,length,azimuth,strength`
- `zone:x,y,h,length,azimuth,strength`

当前 shape 只是等效散射几何模型：每个异常体会被离散成少量散射点，叠加多条 `S-D-G` 绕射路径。它用于研究形状、尺度和位置对合成响应的影响，不代表真实弹性边界散射。当前扫描仍默认寻找主异常体；多异常联合反演可后续扩展为“找一个、减去、再找下一个”的迭代流程。

各 shape 在运动学正演中的散射点含义：

- `sphere`：中心点加六个方向点；
- `box`：中心、八个角点和六个面中心；
- `cylinder`：柱面圆周点和上下端面点；
- `ellipsoid`：椭球赤道和一个竖向子午面采样点；
- `line`：沿给定方位角的一串散射点；
- `zone`：沿给定方位角的长条带状散射点集合。

## 小尺度 3D elastic FDTD 原型

`elastic3d` 用于和默认运动学模型形成对比：

```bash
python main.py elastic3d --no-save
python main.py elastic3d --save
python main.py elastic3d --animate --save
```

默认输出：

```text
outputs/elastic3d/velocity_model_slice.png
outputs/elastic3d/wavefield_snapshot_early.png
outputs/elastic3d/wavefield_snapshot_mid.png
outputs/elastic3d/wavefield_snapshot_late.png
outputs/elastic3d/elastic3d_gather.png
outputs/elastic3d/elastic3d_wavefield.gif  # 仅 --animate --save 时生成
```

它实现的是小尺度三维各向同性 velocity-stress 有限差分原型，变量包括 `vx/vy/vz` 和 `sxx/syy/szz/sxy/sxz/syz`，模型参数包括 `Vp/Vs/rho/lambda/mu`。它会做 CFL 稳定性检查，并使用简化自由表面和 sponge 吸收边界。

注意：`elastic3d` 默认模型尺寸很小，坐标范围与道路 workflow 不完全相同；如果显式传入 `--anomalies`，请确保异常体坐标落在小模型范围内。当前 elastic3d 还不作为扫描定位输入，只用于小模型波场现象验证。

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

- workflow 中的波场 GIF 是等效运动学传播示意，不是高保真弹性波场快照。
- `layered-effective` 已经让层状速度通过 `VR_eff` 影响运动学正演和扫描，但仍不是完整频散反演。
- `elastic3d` 更接近真实波场，但目前只是小尺度教学/研究模型，不替代默认定位算法。
- 后续建议扩展：多频带速度约束、FK/扇形滤波、预测滤波绕射增强、从 elastic3d 波场中提取绕射事件、真实 DAS 触发校正、光纤路径标定和通道耦合 QC。
