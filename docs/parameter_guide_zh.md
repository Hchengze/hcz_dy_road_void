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

### `--source-frequency`

锤击主频 `f`，单位 Hz。它控制等效波长：

```text
lambda = VR / f
```

频率越高，波长越短，浅部小尺度异常分辨率越高，但衰减更强；频率越低，探测更深但分辨率降低。

## 空洞/异常体参数

`--cavity-x`、`--cavity-y`、`--cavity-depth` 分别表示异常体沿道路位置、横向位置和顶部/主要散射中心深度，单位 m。当前空洞是有效散射体，不是完整真实几何边界。

`--cavity-radius` 控制散射影响范围；`--scattering-strength` 控制绕射/散射事件强度；`--attenuation-strength` 控制异常体附近直达波阴影。

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

## 输出参数

- `--save`：保存图件；
- `--show`：交互显示；
- `--no-save`：不保存；
- `--outdir`：输出目录；
- `--dpi`：图片分辨率。

建议本地调参用 `--show --no-save`，汇报出图用 `--save --outdir outputs/<功能名>`。

## 常见错误设置

- 采样率太低：高频锤击信号混叠；
- 记录时长太短：截断直达波或绕射波；
- 扫描范围没覆盖真实异常体；
- 扫描步长太大；
- 噪声很强但仍输出高置信度，需警惕误报；
- `VR` 与真实速度偏差过大；
- `road_width` 与实际锤击线-光纤线距离不一致。
