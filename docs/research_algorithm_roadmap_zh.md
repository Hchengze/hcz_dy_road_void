# 科研级道路空洞算法路线图

本项目当前目标不是复刻 SPECFEM3D、SeisSol、Devito、FEniCS 或 Bempp，而是在本地研究原型中逐步形成“正演—识别—定位/反演—验证”的闭环。不同层级服务不同目的，不能互相替代。

## 1. 当前主线

默认 `python main.py workflow` 仍使用快速三维运动学/属性正演和多炮 joint scan：

```text
RoadVoidConfig
  -> 道路几何 / 速度 / 异常体
  -> 三维等效瑞雷波运动学正演
  -> 直达波拟合与压制
  -> 绕射/散射属性
  -> 多炮联合扫描定位
  -> 误差、置信度、不确定性
```

它适合快速生成合成数据、测试几何孔径、评估参数敏感性和验证定位算法趋势。定位结果仍应解释为“疑似异常范围”，不是直接确诊空洞。

## 2. FDTD / elastic3d

近期主力全波场验证路线是小尺度 2D/3D elastic FDTD：

- 规则网格 body anomaly；
- DAS-like strain-rate recording；
- 局部 elastic validation case；
- 与运动学正演中的异常体位置、深度和散射响应做 sanity check。

`python main.py elastic-validate` 会从道路 workflow 配置裁剪局部区域，把异常体平移到 elastic3d 小网格中。它不是全道路全波场模拟，也不替代默认 workflow。

后续可逐步增强：

- 更严格 staggered-grid；
- 更稳定/完整 CPML；
- 局部 2D/3D 弹性模型与 DAS 应变率响应对比；
- 从 elastic gather 中提取可用于 joint scan 或 FWI 的散射事件。

## 3. FEM

FEM 的近期路线不要直接跳到工业级三维弹性：

```text
1D scalar wave FEM
  -> 2D scalar wave FEM
  -> 2D SH wave FEM
  -> 2D P-SV elastic FEM
  -> 小尺度复杂几何空洞验证
```

FEM 的优势是复杂几何和材料界面；代价是网格、形函数、质量矩阵、刚度矩阵和边界条件都需要仔细验证。

## 4. SEM

SEM / 谱元路线适合高精度波动模拟。本项目当前只保留低维教学原型，后续建议：

```text
1D scalar SEM
  -> 2D scalar SEM
  -> 2D elastic SEM
  -> 与 FDTD benchmark 对比
```

可以学习 SPECFEM 的 GLL 点、质量矩阵对角化和高阶基函数思想，但不直接复刻 SPECFEM3D。

## 5. BEM

BEM 对空洞/夹杂体边界散射有理论吸引力，但三维弹性半空间 BEM 难度很高。本项目下一步建议：

```text
2D Helmholtz Green function
  -> circular cavity scattering
  -> boundary condition comparison
  -> 与运动学散射点模型对照
```

当前 BEM demo 仍是标量边界积分思想演示，不是三维弹性边界元。

## 6. FWI

FWI 必须建立在可信 forward solver 和清晰数据残差定义之上。当前 joint scan 是主定位方法，FWI 是后续 refinement，而不是替代入口。

推荐推进顺序：

```text
1. misfit map
2. finite-difference gradient check
3. one-parameter / two-parameter inversion
4. source/receiver geometry sensitivity
5. later adjoint-state method
```

当前 `fwi-demo` 只做 L2 misfit 曲线，不包含伴随梯度、步长搜索或模型更新。

## 7. 本轮新增科研级闭环

默认 `workflow --save` 会输出完整主链条结果：

- 道路地下三层模型 x-z/y-z/3D 图；
- 结构化 synthetic survey dataset metadata；
- DAS-like gather 和噪声/耦合分量；
- 绕射/散射 envelope 属性和候选评分；
- joint localization error summary；
- `research_report.md`。

这些输出用于科研记录与方法对比，并全部写入 `outputs/workflow/`。`--save-extra` 仅额外保存较重的 `synthetic_dataset.npz` 和多炮诊断图，不改变默认 workflow 的主链条。
