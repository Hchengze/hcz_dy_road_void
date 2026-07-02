# workflow 依赖关系图与协同规则

本文档说明默认 `workflow` 的主链条，以及各模块如何共享同一套参数和同一批数据。它的目的不是增加新的算法功能，而是防止项目再次出现“几何、正演、绕射识别、定位和报告各自使用默认参数”的问题。

## 1. 默认主链条

默认主链条为：

```text
RoadVoidConfig
  -> ScenarioModel
  -> Geometry / Velocity / Anomalies
  -> SyntheticDataset
  -> DAS-like response
  -> Diffraction attributes
  -> Joint localization / inversion
  -> Wavefield visualization
  -> Research report / run_parameters / output_manifest
```

`main.py` 只负责本地参数区、命令行解析和调用高层 runner。真正的 workflow 编排由 `road_void.runner.run_full_workflow()` 完成。

## 2. RoadVoidConfig 如何进入 scenario

`RoadVoidConfig` 是 workflow 的统一配置源。道路宽度、道路长度、速度模式、层速度、异常体位置和扫描范围都先进入 `RoadVoidConfig`，再由 `build_default_subsurface_scenario(cfg)` 生成道路地下场景。

`ScenarioModel` 包含路面层、基层、路基以及异常体信息，用于：

- `02b_subsurface_model_xz.png`
- `02c_subsurface_model_yz.png`
- `02d_subsurface_model_3d.png`
- synthetic dataset metadata
- `research_report.md`
- `elastic-validate` 的局部模型裁剪参考

## 3. scenario 如何进入 dataset

`generate_synthetic_survey_dataset(cfg, workflow)` 使用同一份 `RoadVoidConfig` 和同一次 `run_location_workflow(cfg)` 的结果，生成结构化 synthetic survey dataset。

dataset 包含：

- `shots`
- `receivers`
- `time_axis`
- `data[shot, time, channel]`
- `das_like_data`
- `metadata`
- `labels`

默认 workflow 保存轻量 `synthetic_dataset_metadata.json`。较大的 `synthetic_dataset.npz` 只在 `--save-extra` 时保存。

## 4. dataset 如何进入 diffraction

`detect_diffraction_features(cfg, workflow)` 使用当前 workflow 的残差数据、几何和异常体信息，不重新生成默认数据。

绕射识别结果进入：

- `04b_diffraction_attribute.png`
- `04c_diffraction_candidates.png`
- `research_report.md`

当前 diffraction 是轻量可解释属性，不是机器学习检测器，也不应被解释为“自动确诊空洞”。

## 5. diffraction 如何进入 inversion

当前定位主线仍然是多炮 joint scan。`run_joint_localization_evaluation(cfg, workflow)` 使用同一份扫描结果和真值异常体，输出：

- best estimate
- true position
- x/y/depth 误差
- total error
- confidence
- uncertainty bounds

这些结果进入：

- `05b_localization_error_summary.png`
- `05c_uncertainty_slices.png`
- `research_report.md`

## 6. inversion 如何进入 report

`write_research_report()` 汇总 scenario、dataset、diffraction 和 localization evaluation。报告必须诚实说明：

- 定位结果只是疑似异常范围；
- y-depth 耦合仍然存在；
- 置信度低时不能过度解释；
- 当前模型是运动学/属性近似，不是完整弹性波反演。

## 7. wavefield 如何同步当前参数

wavefield 是 workflow 的第 6 步，不是独立结果体系。它必须使用当前：

- geometry；
- anomalies；
- `velocity_mode`；
- `VR_eff`；
- 代表性炮点；
- scenario/velocity context。

默认 wavefield 是 x-y 平面运动学示意和 3D kinematic 等时面示意，不是完整弹性波场。真正三维弹性波场应使用 `elastic3d`。

## 8. elastic-validate 与 workflow 的关系

`elastic-validate` 是局部全波场验证支线。它从 workflow 的 `RoadVoidConfig` 出发，围绕主异常体裁剪并平移到小尺度 elastic3d 网格。

它不是默认 workflow，也不是全道路尺度 elastic3d 正演。它用于检查局部波场现象是否与运动学假设大致一致。

## 9. 主链条与教学/验证支线

主链条：

```text
workflow
```

验证支线：

```text
elastic3d
elastic-validate
```

教学支线：

```text
numerics
numerics-demo
numerics-compare
fwi-demo
```

教学支线不参与默认道路空洞 workflow，不应影响默认输出目录。

## 10. 如何避免偷偷重建默认数据

新增功能时应优先接收当前 workflow 已有对象，例如 `cfg`、`workflow`、`scenario`、`survey_dataset`、`diffraction_result`、`localization_eval`。不要在下游模块内部重新调用默认 `RoadVoidConfig()` 或重新模拟一套数据。

如果某个模块确实需要独立默认值，必须在 docstring 或控制台提示里说明它是验证/教学支线，而不是默认 workflow 主链条。
