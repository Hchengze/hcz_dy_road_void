# 小尺度 3D 弹性波全波形正演调研与实现说明

本文档说明本项目新增 `elastic3d` 原型的依据、实现边界和与默认运动学正演的关系。它面向本地研究和教学展示，不是工业级三维弹性波模拟方案。

## 1. 为什么需要 elastic3d 原型

默认 workflow 使用三维等效瑞雷波运动学/属性正演，优点是快速、可控、适合验证单侧 DAS + 对侧锤击几何下的绕射扫描定位。但它不求解弹性波方程，因此不能真实表达：

- P 波、S 波和面波之间的相互作用；
- 自由表面附近的波型转换；
- 低速体/空洞边界的反射、绕射和散射；
- 复杂速度/密度结构引起的真实波场畸变。

因此，本项目新增一个小尺度 3D elastic FDTD 原型，用来验证“更接近真实波场”时可能出现的现象，并为后续 2.5D/3D 波动方程扩展打基础。它不替代默认绕射扫描定位流程。

## 2. 基本方程

当前实现采用三维各向同性弹性介质的一阶 velocity-stress 形式。速度变量为：

```text
vx, vy, vz
```

应力变量为：

```text
sxx, syy, szz, sxy, sxz, syz
```

介质参数为：

```text
Vp, Vs, rho
mu = rho * Vs^2
lambda = rho * Vp^2 - 2 * mu
```

速度更新近似为：

```text
rho * d vx / dt = d sxx/dx + d sxy/dy + d sxz/dz
rho * d vy / dt = d sxy/dx + d syy/dy + d syz/dz
rho * d vz / dt = d sxz/dx + d syz/dy + d szz/dz
```

应力更新近似为：

```text
d sxx/dt = (lambda + 2mu) d vx/dx + lambda (d vy/dy + d vz/dz)
d syy/dt = (lambda + 2mu) d vy/dy + lambda (d vx/dx + d vz/dz)
d szz/dt = (lambda + 2mu) d vz/dz + lambda (d vx/dx + d vy/dy)
d sxy/dt = mu (d vx/dy + d vy/dx)
d sxz/dt = mu (d vx/dz + d vz/dx)
d syz/dt = mu (d vy/dz + d vz/dy)
```

## 3. 数值格式

经典弹性波有限差分常使用 velocity-stress 交错网格。Virieux (1986) 的 P-SV velocity-stress 有限差分是这一类方法的重要基础；后续三维 staggered-grid elastic FD 工作把同样思想扩展到更多变量和更高阶差分。

本项目为了保持代码短小，采用同尺寸 NumPy 数组保存所有变量，并使用前/后向配对差分避免简单中心差分带来的奇偶网格解耦。它表达一阶弹性波方程的核心变量关系，但不是严格高阶交错网格，也不追求工业模拟精度。

## 4. 自由表面与吸收边界

默认把 `z=0` 作为简化自由表面，在每步更新后令：

```text
szz = sxz = syz = 0
```

这只是一个教学级自由表面近似。真实高精度弹性波自由表面需要更严格的边界处理。

外边界使用 sponge 吸收层：在靠近边界的若干网格内逐步衰减速度和应力，降低边界反射。CPML/PML 吸收边界通常效果更好，但实现复杂度更高，本阶段暂不加入。

## 5. CFL 稳定性

显式三维有限差分必须满足稳定条件。当前使用保守 CFL 数：

```text
CFL = vmax * dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
```

默认要求：

```text
CFL < 0.45
```

若不满足，程序直接报错并提示减小 `dt` 或增大网格间距，避免输出数值爆炸后的假图。

## 6. 震源与接收

当前震源为浅层垂向 Ricker 脉冲，加载到 `vz` 速度场上，用于近似锤击垂向力。接收线记录浅层 `vz` 分量，作为简化 DAS/检波器响应。

真实 DAS 记录的是沿光纤方向的应变或应变率平均，与 `vz` 并不等同。后续若要更接近真实 DAS，需要把位移/速度场转换为沿光纤方向应变，并考虑 gauge length。

## 7. 与默认运动学正演的区别

默认运动学正演：

- 直接使用 `t_direct` 和 `t_diff` 构造事件；
- 计算快，适合扫描定位和参数敏感性实验；
- 振幅和形状只具备定性解释意义。

`elastic3d` 原型：

- 求解三维弹性波 velocity-stress 方程；
- 可以看到更接近真实的体波/面波混合传播、低速体扰动和边界反射；
- 计算量更大，默认只用于小模型教学/研究；
- 当前还不作为默认绕射扫描输入。

## 8. 本项目的最小可行路线

当前 `road_void/elastic3d.py` 已实现：

- 三维各向同性弹性介质；
- `vx/vy/vz` 和六个应力分量；
- 三层 `Vp/Vs/rho` 模型；
- 低速低密度异常体；
- 垂向锤击源；
- 一条浅层接收线；
- 简化自由表面；
- sponge 吸收边界；
- CFL 检查；
- 速度切片、波场快照、接收记录和可选 GIF。

当前不做：

- GPU/MPI；
- 大规模道路模型；
- 高阶严格交错网格；
- CPML；
- 各向异性；
- 黏弹性；
- FWI/RTM；
- 生产级数据格式。

## 9. 参考资料

- Virieux, J. (1986). P-SV wave propagation in heterogeneous media: velocity-stress finite-difference method.
- Devito 相关工作展示了用 DSL 构造有限差分 PDE 求解器和弹性波方程示例：[Devito v3.1.0](https://arxiv.org/abs/1808.01995)、[Vectorial simulations using Devito](https://arxiv.org/abs/2004.10519)。
- PML/CPML 是弹性波无界域模拟中常用吸收边界思路，可参考这篇综述/教程式资料：[Perfectly matched layers for elastodynamics](https://arxiv.org/abs/2104.09854)。
- P-SV velocity-stress 的教学实现可参考：[P-SV wave propagation in heterogeneous media](https://arxiv.org/abs/2107.14727)。
- 三维 staggered-grid elastic finite-difference 的实现思路可参考：[Efficient staggered grid finite-difference method](https://arxiv.org/abs/1706.01915)。
