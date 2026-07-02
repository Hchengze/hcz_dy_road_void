# main.py 参数中文说明

本项目当前推荐直接通过 `main.py <subcommand> --参数` 修改实验参数。直接运行 `python main.py` 会进入完整 `workflow`，按几何、速度、正演、路径、扫描、总结的顺序跑一遍；单独调试时再使用 `python main.py geometry`、`python main.py forward`、`python main.py scan` 等子命令。YAML 配置仅作为历史兼容，不再是主入口。

## 道路与几何参数

### `--road-width`

道路横向宽度 W，单位 m。它控制锤击线与光纤线之间的横向偏移，是单侧 DAS + 对侧锤击观测孔径的核心参数。道路越宽，直达波和绕射波路径越长，锤击能量越容易衰减，横向位置 `y0` 和深度 `h` 的耦合越明显。

### `--road-length`

道路沿线模拟长度，单位 m。它决定 DAS 通道和锤击点沿 x 方向覆盖范围。长度太短会截断绕射曲线的有效观测孔径。

### `--channel-spacing`

DAS 通道间距，单位 m。通道越密，沿道路方向采样越细，绕射曲线越容易识别；通道过稀会降低 `x0` 定位精度。

### `--source-spacing`

锤击点间距，单位 m。点距越小，多炮约束越强，但采集工作量越大；点距过大会漏掉局部异常最有利的激发位置。

## 速度与频率参数

### `--rayleigh-velocity`

等效瑞雷波速度 `VR`，单位 m/s。直达波和绕射波走时近似为：

```text
t_direct = t0 + distance(S, G) / VR
t_diff = t0 + [distance(S, D) + distance(D, G)] / VR
```

`VR` 是某一频带内的等效速度，不是完整随深度变化的速度模型。真实数据中可由直达波三维几何拟合、DAS-MASW 或交通噪声频散估计。`VR` 偏大可能导致深度偏深，`VR` 偏小可能导致深度偏浅。

### `--velocity-mode`

- `uniform`：使用单一 `VR`，保持原始运动学公式。
- `layered-effective`：根据层状速度、主频和敏感深度近似计算 `VR_eff`，并让正演和扫描都使用该速度。

近似关系为：

```text
lambda = VR / f
z_sensitive ≈ alpha * lambda
w(z) = exp(-z / z_sensitive)
VR_eff = weighted harmonic mean(layer_velocities, w)
```

这只是轻量工程近似，用于让层状速度影响当前走时模型；它不是完整 Rayleigh 频散反演，也不是三维弹性波全波形正演。

### `--layer-depths` / `--layer-velocities`

`--layer-depths` 是层底深度，单位 m；`--layer-velocities` 是每层等效瑞雷速度，单位 m/s。两者数量必须一致。低频波长长、敏感深度大，`VR_eff` 更容易受深层速度影响；高频波长短，更受浅层速度影响。

### `--sensitivity-depth-factor`

敏感深度因子 `alpha`。取值越大，等效速度越受深部层影响；取值越小，越强调浅部速度。

### `--source-frequency`

锤击主频 `f`，单位 Hz。它控制等效波长：

```text
lambda = VR / f
```

频率越高，波长越短，浅部小尺度异常分辨率越高，但衰减更强；频率越低，探测更深但分辨率降低。

## 空洞/异常体参数

`--cavity-x`、`--cavity-y`、`--cavity-depth` 分别表示异常体沿道路位置、横向位置和顶部/主要散射中心深度，单位 m。当前空洞是有效散射体，不是完整真实几何边界。

`--cavity-radius` 控制散射影响范围；`--scattering-strength` 控制绕射/散射事件强度；`--attenuation-strength` 控制异常体附近直达波阴影。

### `--cavity-shape`

单异常体形状。支持 `sphere`、`box`、`cylinder`、`ellipsoid`、`line/zone`。当前 shape 只是等效散射点集合，不代表真实弹性边界散射。

### `--cavity-size-x/y/z` 与 `--cavity-azimuth`

这些参数用于单异常体模式，方便在 VSCode 的 `LOCAL_WORKFLOW["anomaly"]` 中直接改形状尺度：

- `--cavity-size-x`：`box/ellipsoid` 的 x 向尺寸；`line/zone` 的长度；
- `--cavity-size-y`：`box/ellipsoid/zone` 的 y 向尺寸；
- `--cavity-size-z`：`box/ellipsoid/cylinder` 的竖向尺寸或高度；
- `--cavity-azimuth`：`line/zone` 的平面方位角，单位度。

如果没有设置尺寸，代码会用 `cavity_radius` 推导一个默认等效尺度。尺寸变大通常会增加散射点覆盖范围，使散射事件更宽、更复杂；但当前仍是等效散射点模型，不代表真实边界散射强度。

### `--anomalies`

多异常体字符串输入，适合本地快速实验：

```bash
--anomalies "sphere:42,8.5,2.2,2.0,1.0;box:58,6,1.5,4,3,1,0.8"
```

格式：

- `sphere:x,y,h,radius,strength`
- `box:x,y,h,size_x,size_y,size_z,strength`
- `cylinder:x,y,h,radius,height,strength`
- `ellipsoid:x,y,h,size_x,size_y,size_z,strength`
- `line:x,y,h,length,azimuth,strength`
- `zone:x,y,h,length,azimuth,strength`

`zone` 与 `line` 的输入参数相同，但内部会生成一组带宽方向的散射点，因此表示有一定宽度的松散带，而不是一条单线。正演会叠加多个异常体、多个散射点的散射响应。当前扫描默认仍寻找主异常体；多异常联合反演可后续采用“定位一个、减去一个、再扫描”的迭代方式。

## 噪声与耦合参数

- `--noise-level`：随机背景噪声强度。越大，绕射事件越难识别。
- `--traffic-noise-level`：交通类低频干扰强度。
- `--bad-channel-fraction`：坏道比例。
- `--weak-coupling-fraction`：弱耦合通道比例。
- `--coupling-variation`：道间增益随机变化幅度。

这些参数用于模拟真实城市道路 DAS 采集中的噪声、交通干扰、光纤耦合不均和坏道。它们会直接影响绕射拾取、扫描评分和定位置信度。

## 扫描参数

`--scan-x-min/max/step`、`--scan-y-min/max/step`、`--scan-h-min/max/step`、`--scan-vr-min/max/step` 分别控制沿道路、横向、深度和速度的搜索范围与步长。

步长越小，定位网格越细，但计算量越大；步长过大可能导致真实峰值没有被采样到。`--top-k` 用于观察候选点非唯一性，`--uncertainty-threshold` 用于估计接近最高分的参数范围。

`main.py` 顶部的 `LOCAL_WORKFLOW["scan"]` 对这些参数逐项写了中文注释。调参时最重要的检查是：`scan_x/y/h` 范围必须覆盖你设置的异常体位置，否则扫描结果不会回到真实位置；`scan_y_step` 和 `scan_h_step` 不宜过粗，因为单侧 DAS + 对侧锤击几何下 `y-h` 耦合明显，过粗网格会让这种非唯一性被误读成单一最优点。

### `--scan-mode`

- `joint`：默认，多炮联合扫描；
- `single-shot`：只使用 `--shot-index` 指定的一炮；
- `compare`：保留单炮最佳候选和多炮联合结果，便于教学对比。

多炮联合通常能增强沿道路方向 `x0` 的稳定性，因为异常体会在多个炮点记录中产生一致的 `S-D-G` 走时约束。但由于光纤仍在道路单侧，横向 `y0` 与深度 `h` 的耦合不会自动消失。

### `--shot-weight-mode`

- `uniform`：所有炮等权；
- `near-cavity`：靠近候选异常体 `x0` 的炮权重更高；
- `snr`：用记录能量近似信噪比权重。

## 输出参数

- `--save`：保存图件；
- `--show`：交互显示；
- `--no-save`：不保存；
- `--save-extra`：保存体积更大的数据文件和额外诊断图。默认 workflow 已保存完整主链条图件、metadata、`research_report.md`、`run_parameters.json` 和 `output_manifest.txt`；`--save-extra` 主要用于 `synthetic_dataset.npz`、每炮贡献图等额外材料；
- `--clean-output`：运行前清理当前 `outdir` 里的旧结果文件，只影响当前子目录，不会清理整个 `outputs/`；
- `--outdir`：输出目录；
- `--dpi`：图片分辨率。

建议本地调参用 `--show --no-save`，汇报出图用 `--save`。默认主 workflow 和 wavefield 第 6 步都写入 `outputs/workflow/`；只有 `elastic3d`、`numerics` 等独立实验模块保留各自输出目录。

每次保存运行会打印本次实际生成文件，并在当前输出目录写入 `output_manifest.txt`。如果某个目录中历史图很多，建议使用 `--clean-output` 避免把旧图误认为新结果。

默认 workflow 已保存地下模型剖面、synthetic dataset metadata、DAS-like gather、绕射属性、定位误差图和 `research_report.md`，用于形成完整科研记录。`--save-extra` 只保存更重的 `synthetic_dataset.npz` 和额外多炮诊断图；如果只想检查流程而不落盘，请使用 `--no-save`。

## wavefield 参数

默认单炮：

```bash
python main.py wavefield --save
```

输出三个关键帧和一个速度上下文图。速度逻辑为：

```text
outputs/workflow/06_wavefield_frame_early.png
outputs/workflow/06_wavefield_frame_hit_cavity.png
outputs/workflow/06_wavefield_frame_scattered.png
outputs/workflow/06_wavefield_velocity_context.png
```

- `velocity-mode=uniform`：wavefield 使用原始 `VR`；
- `velocity-mode=layered-effective`：wavefield 使用折算后的 `VR_eff`。

`--wavefield-view plan/3d` 控制波场示意的维度：

- `plan`：默认，输出 x-y 地表平面运动学波场示意。它适合看道路平面孔径、炮点、DAS 线和异常体平面位置；异常体深度 `z` 进入 `S-D-G` 走时，但图上不显示完整 z 方向波场。
- `3d`：输出三维运动学等时面示意。它会画地表、DAS 线、炮点、异常体、直达等时半球和散射等时球面，用于理解三维几何关系。

注意：无论 `plan` 还是 `3d`，这里的 wavefield 都是运动学示意，不是严格弹性波方程快照。layered-effective wavefield 仍是等效运动学示意；图中的分层速度只说明 `VR_eff` 的来源，不表示已经模拟了分层介质中的反射、折射或模式转换。

多炮示意参数：

- `--wavefield-mode single-shot/multi-shot`：单炮或多炮顺序激发；
- `--wavefield-shot-index`：单炮模式的炮号；
- `--wavefield-shot-indices`：多炮显式炮号列表，例如 `0,5,10`；
- `--wavefield-max-shots`：最多展示几炮；
- `--wavefield-shot-step`：自动选炮时的炮间隔；
- `--wavefield-shot-interval`：GIF 中相邻炮的示意全局时间间隔。

多炮关键帧：

```bash
python main.py wavefield --wavefield-mode multi-shot --wavefield-shot-indices 0,5,10 --save
```

多炮 GIF：

```bash
python main.py wavefield --wavefield-mode multi-shot --animate --save
```

multi-shot wavefield 用于理解多炮覆盖，不等于 `scan-mode=joint` 的多炮联合定位。

## 小尺度 3D elastic FDTD 参数

`elastic3d` 是独立实验子命令，不是默认 workflow 的一部分：

```bash
python main.py elastic3d --save
python main.py elastic3d --animate --save
```

常用参数：

- `--nx/--ny/--nz`：小模型三维网格数。默认很小，用于本地快速运行；
- `--dx/--dy/--dz`：空间网格间距，单位 m。网格越小，分辨率越高，但 CFL 更严格；
- `--elastic-dt`：时间步长，单位 s。必须满足 CFL 稳定条件；
- `--elastic-nt`：时间步数。越大传播时间越长，计算越慢；
- `--elastic-source-frequency`：Ricker 震源主频，单位 Hz；
- `--elastic-source-amplitude`：垂向力源幅度；
- `--elastic-space-order`：空间差分阶数，支持 2 和 4。四阶使用 `9/8` 与 `-1/24` 的交错模板风格差分，内部数值频散更低，但边界仍会降阶；
- `--elastic-abc`：吸收边界，`sponge` 为默认稳定海绵层，`cpml` 当前为 experimental CPML-like 阻尼，不是完整 CPML；
- `--elastic-record-component`：接收记录分量，支持 `vz`、`vx`、`strain_xx`、`strain_rate_xx`；
- `--elastic-gauge-length`：近似 DAS gauge length，单位 m，用于沿 x 方向应变/应变率差分；
- `--elastic-no-anomaly`：关闭低速低密度异常体，用于和有异常模型对比。

`main.py` 顶部的 `LOCAL_ELASTIC3D` 对这些参数写了更详细的学习型注释。需要特别注意：`elastic3d` 的坐标范围由 `nx*dx`、`ny*dy`、`nz*dz` 决定，通常远小于道路 workflow 的 `road_length/road_width`；因此不要默认把道路运动学 workflow 的异常体坐标直接解释为 elastic3d 网格中的有效异常体位置。

CFL 检查公式为：

```text
CFL = vmax * dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
```

当前要求 `CFL < 0.45`。如果参数不稳定，程序会报错，而不是输出可能误导的波场图。

注意：elastic3d 默认坐标范围是小模型范围，不等同于道路 workflow 的 80 m 沿线范围。如果给 `elastic3d` 显式传入 `--anomalies`，异常体坐标必须落在这个小模型范围内才会影响波场。

## FWI-demo 参数

`fwi-demo` 当前只是误差函数演示，不是完整 FWI：

- `--fwi-vs-scales`：候选 `Vs` 缩放因子列表；
- `--fwi-observed-vs-scale`：生成目标数据的 `Vs` 缩放；
- `--fwi-initial-vs-scale`：绘制目标/初始合成记录对比时使用的初始 `Vs` 缩放。

误差函数为：

```text
J(m) = 0.5 * ||d_cal(m) - d_obs||^2
```

当前没有伴随梯度、步长搜索或模型更新。

## VSCode 本地参数一致性

直接运行 `python main.py` 时，如果 `USE_LOCAL_DEBUG_CONFIG=True`，程序会从 `main.py` 顶部的 `LOCAL_OUTPUT` 和 `LOCAL_WORKFLOW` 构建 argparse 参数，再统一转换为 `RoadVoidConfig`。主要路径为：

```text
LOCAL_OUTPUT + LOCAL_WORKFLOW -> build_args_from_local_config() -> build_road_void_config_from_args() -> RoadVoidConfig
```

因此：

- 修改 `LOCAL_WORKFLOW["geometry"]` 后，geometry/forward/wavefield/path/scan/workflow 会同步使用同一套道路宽度、道路长度、通道间距和炮点间距；
- 修改 `LOCAL_WORKFLOW["anomaly"]` 后，图件、正演和 workflow 会同步显示/使用同一组异常体；
- 如果 `anomalies` 字符串非空，它优先于单异常体参数；
- 修改 `LOCAL_WORKFLOW["velocity"]["velocity_mode"]` 后，速度图、正演走时、扫描速度轴和 `run_parameters.json` 会同步反映。

程序会打印参数摘要，并对常见不一致给出 warning，例如扫描范围未覆盖异常体、记录长度不足或异常体超出道路横向孔径。

## 速度图件含义

`outputs/workflow/02_velocity_model.png` 和 `outputs/velocity/velocity_model.png` 展示的是当前运动学正演/扫描使用的等效瑞雷速度模型：

- `uniform`：单层均匀速度，正演和扫描使用单一 `VR`；
- `layered-effective`：显示分层速度，并把层状速度按 `lambda=VR/f` 和敏感深度因子折算为 `VR_eff`，正演和扫描使用 `VR_eff`。

这张图不是 `elastic3d` 的 `Vp/Vs/rho` 全波场模型，也不是完整 Rayleigh 频散反演。

## 常见错误设置

- 采样率太低：高频锤击信号混叠；
- 记录时长太短：截断直达波或绕射波；
- 扫描范围没覆盖真实异常体；
- 扫描步长太大；
- 噪声很强但仍输出高置信度，需警惕误报；
- `VR` 与真实速度偏差过大；
- `road_width` 与实际锤击线-光纤线距离不一致。
- 对 `elastic3d` 使用道路尺度异常体坐标，例如默认道路异常 `x=42 m`，但小模型长度只有约 28 m，导致异常体不在模型内。
