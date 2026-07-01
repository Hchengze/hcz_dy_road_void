# 参数配置中文说明

本文档解释 `configs/*.yaml` 中主要参数的物理意义、单位、推荐范围、对正演和定位的影响，以及真实道路 DAS 场景中如何估计或标定。

## 1. 道路与几何参数

| 参数 | 单位 | 默认值 | 推荐范围 | 物理意义与影响 | 真实数据来源 |
|---|---:|---:|---:|---|---|
| `road_width` | m | 15 | 15-35 | 光纤线与锤击线之间的横向孔径。越宽，传播路径越长，锤击能量越弱，`y0-h` 耦合越明显。 | 道路横断面测量、光纤与锤击线实测 |
| `road_length` | m | 80 | 40-200 | 模拟道路沿线长度。过短会限制可观测绕射曲线。 | 作业区长度 |
| `fiber_y` | m | 0 | 实测 | DAS 光纤横向位置。通常设为道路一侧基准 `y=0`。 | 光纤路由标定 |
| `source_y` | m/null | null | 实测 | 锤击线横向位置。`null` 表示使用 `road_width`。 | 锤击点测量 |
| `channel_spacing` | m | 1 | 0.2-2 | DAS 通道间距。越小接收越密集，`x0` 分辨率更好，但数据量增加。 | DAS 设置 |
| `source_spacing` | m | 4 | 1-10 | 锤击点间距。越小炮点越密集，定位更稳，但外业效率降低。 | 采集设计 |
| `fiber_depth` | m | 0 | 0-2 | 光纤埋深。当前可进入几何坐标，但真实耦合和 gauge length 影响仍需额外处理。 | 管线资料、探测或开挖记录 |
| `source_depth` | m | 0 | 0 | 锤击源深度，通常近似地表。 | 采集方式 |

## 2. 光纤与 DAS 通道参数

`channel_x_min`、`channel_x_max` 和 `channel_spacing` 共同定义 DAS 通道坐标。真实数据中，这些参数不能只用通道号代替，必须通过光纤路径和道号映射表标定。若光纤实际弯曲、绕行或埋深变化明显，简单直线模型会导致走时解释偏差。

## 3. 锤击震源参数

`source_x_min`、`source_x_max` 和 `source_spacing` 定义锤击点序列。真实采集中还需要记录每个锤击点坐标和触发时间。`source_frequency` 是锤击主频，单位 Hz；主频越高，浅部分辨率越好，但传播衰减更强。

## 4. 空洞/异常体参数

| 参数 | 单位 | 默认值 | 影响 |
|---|---:|---:|---|
| `enable_cavity` | - | true | 是否在合成数据中加入异常体。无空洞配置用于误报风险测试。 |
| `cavity_x` | m | 42 | 控制绕射曲线沿道路方向的位置，通常最容易被单侧 DAS 约束。 |
| `cavity_y` | m | 8.5 | 控制横向位置，但与深度 `cavity_h` 容易耦合。 |
| `cavity_h` | m | 2.2 | 空洞顶部或主要散射中心深度。越深，绕射到时更晚、能量更弱。 |
| `cavity_radius` | m | 2 | 控制散射影响范围，不是完整真实几何边界。 |
| `scattering_strength` | - | 1 | 越大，绕射/散射事件越明显。 |
| `attenuation_strength` | - | 0.25 | 越大，空洞附近直达波阴影越明显。 |
| `tail_strength` | - | 1 | 控制散射尾波可见性。 |

当前模型中的空洞是“有效散射体/异常体”，不是完整三维空腔边界。真实解释中，空洞形状、充填状态、含水情况和上覆结构都会改变散射响应。

## 5. 速度与频率参数

| 参数 | 单位 | 默认值 | 说明 |
|---|---:|---:|---|
| `rayleigh_velocity` | m/s | 240 | 当前频带内的等效瑞雷波速度。速度偏大可能导致深度解释偏深，速度偏小可能导致深度解释偏浅。 |
| `velocity_model_type` | - | uniform | `uniform` 进入核心正演和扫描；`layered` 当前主要用于速度模型图展示。 |
| `layer_depths` | m | [0.5,2,6] | 分层模型的层底深度，用于展示。 |
| `layer_velocities` | m/s | [320,260,220] | 分层等效瑞雷速度，用于展示。 |
| `source_frequency` | Hz | 35 | 锤击主频。频率越高，浅部敏感性越强，但衰减更明显。 |
| `wavelet_type` | - | ricker | `ricker` 或 `hammer`。 |
| `bandpass_freqmin/freqmax` | Hz | 10/90 | 预处理带通范围。 |
| `multi_band` | Hz | - | 预留多频带约束。 |

真实数据中，`rayleigh_velocity` 可由直达波三维几何拟合、DAS-MASW 主动源频散、交通噪声互相关或其他面波方法估计。

## 6. 采样与记录参数

| 参数 | 单位 | 默认值 | 常见问题 |
|---|---:|---:|---|
| `sampling_rate` | Hz | 1000 | 太低会导致高频锤击信号混叠。 |
| `duration` | s | 1.0 | 太短会截断直达波、绕射波或尾波。 |
| `t0` | s | 0.02 | 触发时间或系统时延，误差会整体平移走时曲线。 |
| `stack_count` | 次 | 1 | 预留参数，未来可模拟重复锤击叠加。 |
| `random_seed` | - | 2027 | 控制噪声和坏道随机性，保证示例可复现。 |

## 7. 噪声与耦合参数

| 参数 | 默认值 | 影响 |
|---|---:|---|
| `noise_level` | 0.03 | 随机背景噪声。越大，绕射事件越难识别，置信度通常下降。 |
| `traffic_noise_level` | 0.015 | 城市交通类低频干扰。可能污染浅层面波记录。 |
| `bad_channel_fraction` | 0.02 | 坏道比例。过高会破坏相干叠加。 |
| `weak_coupling_fraction` | 0.06 | 弱耦合通道比例。真实运营商光纤常见。 |
| `coupling_variation` | 0.08 | 通道增益随机变化幅度。 |
| `direct_wave_strength` | 1.0 | 直达波强度。过强时可能掩盖绕射波。 |
| `diffraction_strength` | 1.0 | 绕射波相对强度。越低越难定位。 |

## 8. 处理与扫描参数

| 参数 | 单位 | 默认值 | 说明 |
|---|---:|---:|---|
| `direct_wave_mute_width` | s | 0.04 | 直达波压制窗口。过宽可能误伤浅部绕射波。 |
| `direct_wave_subtraction_enable` | - | true | true 表示使用模板减去，通常比宽 mute 更温和。 |
| `scan_x_min/max/step` | m | 32/52/1 | 沿道路方向扫描范围和步长。 |
| `scan_y_min/max/step` | m | 3/14/1 | 横向扫描范围和步长。 |
| `scan_h_min/max/step` | m | 0.8/4/0.4 | 深度扫描范围和步长。步长越小，计算量越大。 |
| `scan_vr_min/max/step` | m/s | 220/260/10 | 等效速度扫描范围。 |
| `score_method` | - | envelope | 当前支持 envelope/energy 形式，semblance-like 为预留方向。 |
| `top_k` | - | 8 | 输出前 k 个候选点。 |
| `uncertainty_threshold` | - | 0.92 | 接近最高评分的候选集合用于估计不确定性范围。 |

## 9. 参数敏感性分析建议

建议优先测试：

1. `road_width`：观察道路变宽后横向和深度不确定性是否增大；
2. `cavity_h`：观察深度变大后绕射事件是否变弱；
3. `rayleigh_velocity`：观察速度误差如何映射为深度偏差；
4. `noise_level`：观察置信度和误报风险；
5. `channel_spacing/source_spacing`：观察采样孔径对定位稳定性的影响；
6. `scattering_strength`：观察弱散射体的可见性。

运行：

```bash
python examples/example_09_parameter_sensitivity.py --config configs/default_road_void.yaml
```

## 10. 常见错误参数设置

- 采样率太低：高频锤击信号混叠，拾取不稳定；
- 记录时长太短：直达波或绕射波被截断；
- 扫描范围没有覆盖真实空洞：评分峰会落在边界；
- scan step 太大：真实峰值可能没有被采样；
- noise_level 过大但仍输出高置信度：需要警惕随机噪声或残余直达波误报；
- rayleigh_velocity 与真实速度偏差过大：深度解释会系统偏浅或偏深；
- road_width 与实际锤击线-光纤线距离不一致：所有三维走时都会偏差；
- 真实光纤路径弯曲但仍按直线建模：x/y 坐标映射可能错误。
