# 城市道路空洞三维 DAS 正演与定位研究原型

这是一个本地算法研究原型，用于探索“道路一侧 DAS 光纤接收 + 道路另一侧锤击激发”的城市道路空洞/脱空/松散异常体探测方法。项目目标是便于直接运行、修改参数、查看图件、理解公式和验证算法闭环；不追求发布、安装或生产级工程化。

当前有两条正演路线：

- 默认 workflow 使用快速三维等效瑞雷波运动学/属性正演，用于几何验证、合成数据和绕射扫描定位。
- `elastic3d` 子命令提供小尺度三维弹性波全波形有限差分原型，用于理解更接近真实波场的传播、散射和接收现象。

默认 `python main.py` 仍运行快速 workflow；`elastic3d` 是独立实验模块，当前不替代绕射扫描定位主线。

## 推荐入口

如果在 VSCode 中本地调试，推荐直接打开 [main.py](main.py)，只修改顶部的 `LOCAL_RUN_MODE`、`LOCAL_OUTPUT` 和 `LOCAL_WORKFLOW`，然后点击 Run Python File：

```bash
python main.py
```

当 `USE_LOCAL_DEBUG_CONFIG=True` 且没有输入命令行子命令时，`python main.py` 会读取本地 workflow 配置，默认输出进入 `outputs/workflow/`。例如：

```python
LOCAL_RUN_MODE = "workflow"

LOCAL_OUTPUT = dict(
    save=True,
    show=False,
    animate=False,
    save_extra=False,
    clean_output=True,
    outdir="outputs/workflow",
    dpi=150,
)

LOCAL_WORKFLOW = dict(
    geometry=dict(road_width=24.0, road_length=80.0),
    velocity=dict(velocity_mode="layered-effective", rayleigh_velocity=240.0),
    anomaly=dict(cavity_shape="sphere", cavity_x=42.0, cavity_y=8.5, cavity_depth=2.2),
    scan=dict(scan_mode="joint", scan_x_min=20.0, scan_x_max=70.0),
)
```

这种方式只是为了本地算法调试方便，不是 YAML 或软件包配置系统。

如果显式输入子命令，仍然按 argparse 命令行运行：

```bash
python main.py workflow
python main.py scan --road-width 30 --cavity-depth 2.5 --no-save
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
- `wavefield`：调试 workflow 第 6 步的等效运动学波场示意；默认仍写入 `outputs/workflow/`。
- `path`：展示直达路径、绕射路径、公式和理论曲线。
- `scan`：做绕射扫描定位，输出最佳候选和评分图。
- `sensitivity`：做参数敏感性分析，输出少量趋势图和 CSV。
- `tutorial`：生成一套不重复的教学流程图。
- `elastic3d`：运行小尺度三维弹性波全波形有限差分原型，输出速度切片、波场快照和接收记录。
- `fwi-demo`：运行一维 `Vs` 缩放 L2 misfit 曲线演示，不是完整伴随 FWI。
- `numerics-demo`：运行 FEM/SEM/BEM 低维标量教学原型，不替代默认 workflow。
- `numerics-compare`：运行统一 1D 标量波 benchmark，对比 FDTD/FEM/SEM 的接收记录、到时和 L2 差异。
- `workflow`：默认完整流程入口，控制台按算法步骤解释，输出一套代表性图件。
- `all`：当前作为 `workflow` 的别名，保留给习惯使用 `all` 的场景；日常推荐使用 `workflow` 或直接 `python main.py`。

## LOCAL_RUN_MODE 模式说明

`main.py` 顶部已经把常用运行模式写成注释表。这里再给一个阅读版摘要，方便在 VSCode 中修改 `LOCAL_RUN_MODE` 前快速判断该跑哪一步：

| 模式 | 主要用途 | 主要输出 | 重点检查 | 注意事项 |
| --- | --- | --- | --- | --- |
| `workflow` | 推荐主入口，按几何、速度、正演、路径、扫描、总结顺序运行 | `outputs/workflow/01_*.png` 到 `05_*.png` | `LOCAL_WORKFLOW["geometry"]`、`["velocity"]`、`["anomaly"]`、`["scan"]` | 默认不生成 GIF；`--animate` 才输出第 6 步波场 |
| `all` | `workflow` 的别名 | 同 `workflow` | 同 `workflow` | 不额外跑 numerics 或 elastic3d，避免重复输出 |
| `geometry` | 只检查道路、光纤、炮线、异常体几何 | 几何平面/剖面和 3D 图 | `road_width`、`road_length`、`cavity_x/y/depth` | `01_geometry_plan_sections.png` 第三子图是 y-z 横剖面 |
| `velocity` | 只检查当前运动学正演/扫描使用的速度模型 | `velocity_model.png` | `velocity_mode`、`layer_depths`、`layer_velocities` | `layered-effective` 只是计算 `VR_eff`，不是完整频散反演 |
| `forward` | 只生成合成 shot gather | `forward_shot_gather.png` | `VR/VR_eff`、异常体强度、噪声 | 用于看直达波和散射/绕射事件是否合理 |
| `wavefield` | 调试 workflow 第 6 步波场示意 | `06_wavefield_*.png/gif` | `wavefield_view`、`wavefield_mode`、炮号 | 默认是 x-y 平面运动学示意，不是完整 3D 弹性波场 |
| `path` | 只看 S-G 直达路径和 S-D-G 绕射路径 | 路径图和 gather 曲线叠加 | 炮点、异常体、接收点几何 | 适合核查走时公式的几何含义 |
| `scan` | 只运行三维绕射扫描定位 | 残差曲线和评分切片 | `scan_x/y/h/vr` 范围和步长 | 扫描范围不覆盖异常体时不可能找回目标 |
| `sensitivity` | 参数敏感性分析 | 趋势图和 CSV | 速度、噪声、深度、步长 | 适合看趋势，不作为单次定位结论 |
| `tutorial` | 学习型流程样例 | 少量教学图 | 流程顺序 | 不建议作为正式结果判断入口 |
| `elastic3d` | 小尺度 3D elastic FDTD 原型 | 弹性波切片、快照、gather | `dx/dy/dz`、`dt`、`CFL`、`Vp/Vs/rho` | 与道路 workflow 尺度不同，不替代运动学定位主线 |
| `elastic-validate` | 从 workflow 配置裁剪局部 elastic3d 验证模型 | `outputs/elastic3d_validation/` 下的局部模型切片、gather、快照 | 主异常体是否落入小网格、CFL、记录分量 | 局部 sanity check，不是全道路 elastic3d 正演 |
| `fwi-demo` | L2 misfit 曲线演示 | misfit 曲线和 gather 对比 | `Vs` 缩放候选 | 不是完整伴随 FWI，不做模型更新 |
| `numerics-demo` | FEM/SEM/BEM 低维标量教学原型 | 低维教学图 | 方法类型 | 不是三维弹性模拟 |
| `numerics-compare` | FDTD/FEM/SEM 统一 1D benchmark | trace、wavefield、metrics JSON | 到时、峰值、L2 差异 | 不参与道路空洞 workflow |

## VSCode 本地调试：参数一致性

推荐本地调参时只改 [main.py](main.py) 顶部的 `LOCAL_OUTPUT` 和 `LOCAL_WORKFLOW`。现在所有主流程都走同一条参数路径：

```text
LOCAL_OUTPUT + LOCAL_WORKFLOW
        ↓
build_args_from_local_config()
        ↓
build_road_void_config_from_args(args)
        ↓
RoadVoidConfig
        ↓
geometry / forward / wavefield / path / scan / workflow
```

因此修改 `LOCAL_WORKFLOW["geometry"]` 里的 `road_width / road_length / channel_spacing / source_spacing` 后，几何图、正演炮检坐标、波场示意、路径图和扫描都会同步使用同一套几何。程序启动时会打印参数摘要和一致性 warning；如果看到“扫描范围没有覆盖异常体”或“记录长度可能不足”，应先调整参数再解释结果。

异常体可以直接改单异常体参数，也可以写多异常体字符串。如果 `anomalies` 非空，它优先于单异常体参数：

```python
LOCAL_WORKFLOW["anomaly"].update(
    enable_cavity=True,
    cavity_shape="cylinder",
    cavity_x=50.0,
    cavity_y=10.0,
    cavity_depth=3.0,
    cavity_radius=2.0,
    cavity_size_z=5.0,
    scattering_strength=1.0,
    anomalies="",
)
```

```python
LOCAL_WORKFLOW["anomaly"].update(
    anomalies="sphere:42,8.5,2.2,2.0,1.0;box:58,6,1.5,4,3,1,0.8",
)
```

默认 VSCode 本地 workflow 使用 `velocity_mode="layered-effective"`，便于显示道路浅层分层速度，并让 `VR_eff` 真正进入正演和扫描。如果命令行显式指定 `--velocity-mode uniform`，速度图会显示单层均匀速度并标注当前走时使用单一 `VR`。

## 图件保存与交互显示

所有绘图入口支持：

```bash
--save       保存图件
--no-save    不保存图件
--save-extra 保存体积较大的数据文件和额外多炮诊断图
--clean-output 运行前只清理当前 outdir 中的旧图/旧清单
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

默认 workflow 主线输出到：

```text
outputs/workflow/
```

项目不再默认生成 `outputs/local_debug/` 或独立的 `outputs/wavefield/`。`wavefield` 是 workflow 第 6 步的调试入口，即使单独运行 `python main.py wavefield --save`，结果也会写入 workflow 输出体系。`elastic3d`、`numerics` 等实验模块仍保留各自目录，避免和主 workflow 图件混在一起。

`workflow --save` 默认保存一套完整但不重复的科研主线结果，文件名采用步骤编号，便于按算法顺序阅读：

```text
outputs/workflow/01_geometry_plan_sections.png
outputs/workflow/01_geometry_3d.png
outputs/workflow/02_velocity_model.png
outputs/workflow/03_forward_gather.png
outputs/workflow/03b_das_like_gather.png
outputs/workflow/03c_noise_components.png
outputs/workflow/04_diffraction_path.png
outputs/workflow/04_gather_with_curves.png
outputs/workflow/04b_diffraction_attribute.png
outputs/workflow/04c_diffraction_candidates.png
outputs/workflow/05_residual_best_curve.png
outputs/workflow/05_scan_score_slices.png
outputs/workflow/05b_localization_error_summary.png
outputs/workflow/05c_uncertainty_slices.png
outputs/workflow/06_wavefield_frame_early.png
outputs/workflow/06_wavefield_frame_hit_cavity.png
outputs/workflow/06_wavefield_frame_scattered.png
outputs/workflow/06_wavefield_3d_frame_early.png
outputs/workflow/06_wavefield_3d_frame_hit_cavity.png
outputs/workflow/06_wavefield_3d_frame_scattered.png
outputs/workflow/synthetic_dataset_metadata.json
outputs/workflow/research_report.md
outputs/workflow/run_parameters.json
outputs/workflow/output_manifest.txt
```

默认不会保存 GIF，也不会保存体积较大的 `synthetic_dataset.npz` 或多炮诊断图。需要重数据和额外诊断图时显式加：

```bash
python main.py workflow --save --save-extra
```

每次保存运行会打印“本次实际生成文件”，并写入当前输出目录的 `output_manifest.txt`。若担心旧图混淆，可加 `--clean-output`；它只清理当前 `outdir` 中的旧结果文件，不会删除整个 `outputs/`。

`outputs/workflow/02_velocity_model.png` 是当前运动学正演/扫描使用的等效速度模型展示：

- `uniform`：图上显示单层，走时使用单一 `VR`；
- `layered-effective`：图上显示多层速度，并标注 `VR`、`VR_eff`、`source_frequency`、`lambda=VR/f`、敏感深度因子和各层速度。

它不是 `elastic3d` 的 `Vp/Vs/rho` 模型，也不是完整 Rayleigh 频散反演。

默认 `workflow` 不生成 GIF，避免每次运行输出过多。需要查看传播过程时：

```bash
python main.py workflow --animate --save
python main.py wavefield --animate --save
```

如果运行 `python main.py wavefield --save` 而不加 `--animate`，会输出三张关键帧和一个速度上下文图。注意：这些文件仍然写入 `outputs/workflow/`：

```text
outputs/workflow/06_wavefield_frame_early.png
outputs/workflow/06_wavefield_frame_hit_cavity.png
outputs/workflow/06_wavefield_frame_scattered.png
outputs/workflow/06_wavefield_velocity_context.png
```

这些图和 GIF 都是“等效运动学传播示意”，用于理解直达波前、异常体散射触发时刻和 DAS 接收线位置，不是完整弹性波场快照。当前实现中散射波只有在直达波从震源传播到异常体之后才出现，三张关键帧含义为：

- `early`：直达波刚离开震源，异常体散射不应出现；
- `hit_cavity`：直达波前接近或到达第一个异常体；
- `scattered`：异常体散射波已经从异常体位置向外传播。

wavefield 与速度模式的关系：

- `velocity-mode=uniform`：wavefield 使用原始 `VR`；
- `velocity-mode=layered-effective`：wavefield 使用层状速度折算出的 `VR_eff`。

默认 `--wavefield-view plan` 输出的是 **x-y 地表平面运动学波场示意**。这样画是合理的，因为道路、光纤和锤击点主要布置在地表平面，瑞雷波传播路径首先需要从平面上理解；深度 `z` 通过异常体深度进入 `S-D-G` 走时计算，但图上不显示完整三维波场。

可选 `--wavefield-view 3d` 会输出三维运动学等时面示意：

```bash
python main.py wavefield --wavefield-view 3d --save
python main.py wavefield --wavefield-view 3d --animate --save
```

3D kinematic wavefield 会画出地表、DAS 线、炮点、异常体、直达等时半球和散射等时球面，帮助理解 x-y-z 几何关系。它仍然不是弹性波方程快照；真正的 x-y-z 体波场应使用 `python main.py elastic3d`。

但 layered-effective wavefield 仍是等效运动学波场，波前会像均匀介质一样扩散；它只是使用 `VR_eff` 改变传播半径和到时，并在 `06_wavefield_velocity_context.png` 中显示分层速度背景。不要把它解释为严格分层介质弹性波场。真正的分层全波场现象应使用 `elastic3d` 或后续更严格的数值方法检查。

多炮 wavefield 只在显式请求时生成：

```bash
python main.py wavefield --wavefield-mode multi-shot --wavefield-shot-indices 0,5,10 --save
python main.py wavefield --wavefield-mode multi-shot --animate --save
```

不加 `--animate` 时，每个选中炮只输出一张代表性关键帧；加 `--animate` 时输出：

```text
outputs/workflow/06_multishot_wavefield.gif
```

multi-shot wavefield 是多炮覆盖的传播示意，帮助理解炮点位置如何变化；真正用于定位的是 `scan-mode=joint` 的多炮联合评分。不要把 multi-shot wavefield 称为多炮联合反演。

## 科研级合成数据与定位评估

默认 `python main.py workflow --save` 已经输出 scenario、dataset metadata、DAS-like gather、绕射属性、定位误差、wavefield 关键帧和 `research_report.md`，这些结果共同构成一套完整但不重复的科研记录。需要保存体积更大的合成数据体和额外多炮诊断图时，显式运行：

```bash
python main.py workflow --save --save-extra --clean-output
```

`--save-extra` 主要额外生成：

```text
outputs/workflow/synthetic_dataset.npz
outputs/workflow/per_shot_best_x.png
outputs/workflow/per_shot_score_contribution.png
outputs/workflow/single_shot_vs_joint.png
```

地下模型说明、结构化合成 survey dataset metadata、DAS-like 响应近似、绕射/散射属性、joint localization 误差和置信度默认已经进入 workflow 与 `research_report.md`。DAS-like response 只是沿光纤方向差分/平滑的近似，不是真实 DAS 仪器解调结果。

完整依赖链说明见 [docs/workflow_dependency_map_zh.md](docs/workflow_dependency_map_zh.md)。

局部 elastic3d 验证入口：

```bash
python main.py elastic-validate --save
```

它会围绕主异常体裁剪局部区域，把道路坐标平移到小尺度 elastic3d 网格，输出 `outputs/elastic3d_validation/`。这是局部全波场 sanity check，不是全道路 elastic3d 正演，也不替代默认 workflow。

后续 FEM/SEM/BEM/FWI 的科研路线见 [docs/research_algorithm_roadmap_zh.md](docs/research_algorithm_roadmap_zh.md)。

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
python main.py elastic3d --elastic-space-order 4 --no-save
python main.py elastic3d --elastic-record-component strain_rate_xx --no-save
python main.py elastic3d --elastic-abc cpml --no-save
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

当前 elastic3d 增强项：

- `--elastic-space-order 2/4`：二阶或四阶前/后向配对差分；四阶使用 `9/8` 与 `-1/24` 的交错模板系数，但变量仍同尺寸存储，因此不是严格工业级交错网格。
- `--elastic-abc sponge/cpml`：`sponge` 为默认稳定海绵层；`cpml` 是 experimental CPML-like 多项式阻尼，没有完整辅助记忆变量，不能称为严格 CPML。
- `--elastic-record-component vz/vx/strain_xx/strain_rate_xx`：`strain_rate_xx≈dvx/dx` 用于近似沿 x 方向光纤 DAS 应变率响应。
- `--elastic-gauge-length`：用于 DAS 近似记录的有限长度差分。真实 DAS 还受光纤走向、gauge length、耦合和解调方式影响。

注意：`elastic3d` 默认模型尺寸很小，坐标范围与道路 workflow 不完全相同；如果显式传入 `--anomalies`，请确保异常体坐标落在小模型范围内。当前 elastic3d 还不作为扫描定位输入，只用于小模型波场现象验证。

## FWI 最小原型

```bash
python main.py fwi-demo --no-save
python main.py fwi-demo --save
```

当前 `fwi-demo` 只做一件事：改变一个标量 `Vs` 缩放因子，反复调用小尺度 `elastic3d` 正演，并计算：

```text
J(m) = 0.5 * ||d_cal(m) - d_obs||^2
```

输出：

```text
outputs/fwi/misfit_curve.png
outputs/fwi/observed_vs_synthetic_gather.png
```

它没有实现伴随方程、梯度、步长搜索或模型更新，因此不能称为完整三维弹性 FWI。

## 高级数值方法教学原型

本项目新增 `road_void/numerics/`，用于逐步自研 FEM/BEM/SEM/FDTD 等数值方法的教学和研究级原型：

```bash
python main.py numerics-demo --method fem --no-save
python main.py numerics-demo --method sem --no-save
python main.py numerics-demo --method bem --no-save
python main.py numerics-demo --method all --save
python main.py numerics-compare --save
```

当前实现：

- FEM：1D 标量波方程 `M u_tt + K u = f`，线性单元、质量矩阵、刚度矩阵；
- SEM：1D 标量谱元波动方程，GLL 节点、Lagrange 导数矩阵、质量/刚度组装；
- BEM：2D 标量边界积分思想演示，圆形边界离散和简化 Green 函数散射响应；
- FDTD：`road_void/numerics/fdtd.py` 不复制 `elastic3d`，只提供路线说明和 1D 标量波 benchmark 辅助。

这些都是低维标量教学原型，不是完整三维弹性 FEM/BEM/SEM。详细说明见 [docs/advanced_numerical_methods_zh.md](docs/advanced_numerical_methods_zh.md)。

`numerics-compare` 使用同一个 1D 均匀标量波问题对比 FDTD/FEM/SEM：

```text
u_tt = c^2 u_xx + f
```

三种方法使用相同的 `length / velocity / duration / dt / source_position / receiver_position / source_frequency`。输出：

```text
outputs/numerics/compare_1d_traces.png
outputs/numerics/compare_1d_wavefields.png
outputs/numerics/compare_1d_metrics.json
```

该 benchmark 主要用于检查首次到时是否接近、接收记录是否有限、L2 差异是否可解释。它仍是低维标量教学对比，不代表完整三维弹性求解器。BEM 仍保持二维标量边界积分思想演示，不参与该 1D 波动方程对比。

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

如需检查代码层面的 Python warning，可使用：

```bash
python -W default main.py workflow --no-save
python -m pytest -q -W default
```

这里关注的是 `DeprecationWarning`、`RuntimeWarning`、Matplotlib/NumPy warning 等代码运行问题。扫描范围未覆盖异常体、定位置信度低、`layered-effective` 不是完整分层波场等属于物理诊断，应通过控制台摘要和 `research_report.md` 说明，不应依赖 Python warning 刷屏。

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
- `elastic3d` 更接近真实波场，但目前只是小尺度教学/研究模型，不替代默认定位算法；当前四阶差分、cpml-like 边界和 DAS 应变率记录都属于研究原型。
- `fwi-demo` 只是 misfit 曲线演示，不是完整 FWI。
- 后续建议扩展：多频带速度约束、FK/扇形滤波、预测滤波绕射增强、从 elastic3d 波场中提取绕射事件、真实 DAS 触发校正、光纤路径标定和通道耦合 QC。
