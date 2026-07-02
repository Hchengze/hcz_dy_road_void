# 高级数值方法路线说明：FDTD / FEM / BEM / SEM / DG

本文档说明本项目新增 `road_void/numerics/` 的定位。它不是要替代现有道路空洞 workflow，而是为后续自研高阶数值方法建立教学和研究级起点。

## 1. 多层次建模路线

本项目当前可以按四个层级理解：

```text
Level 1：point_set 等效散射点模型
Level 2：surface/body sampling 运动学采样模型
Level 3：elastic3d 体属性 FDTD 模型
Level 4：BEM / FEM / SEM / DG 等高级数值方法原型与接口
```

这些层级服务不同目的：

- Level 1 和 Level 2 不是错误，而是快速近似；适合参数扫描、定位算法验证、趋势分析和教学解释；
- Level 3 更接近真实全波场；适合小尺度波场 sanity check，但仍不是工业级三维弹性模拟；
- Level 4 是未来高精度建模路线；适合逐步研究复杂边界、非结构网格、高阶基函数和边界积分方法。

因此，现有运动学散射点模型和 `elastic3d` 不会废弃。高级数值方法是新增研究路线，不能用未验证的高级模块替代默认 `workflow`。

## 2. FDTD

`road_void/elastic3d.py` 属于有限差分路线。FDTD 的核心思想是在规则网格上用差分模板近似空间导数，并显式推进时间。

优点：

- 结构直观；
- 适合规则网格；
- 容易做小尺度三维教学原型；
- 便于观察波场传播、反射和散射。

局限：

- 复杂曲面边界会出现 stair-step 近似；
- 高精度自由表面和完整 CPML 需要更复杂实现；
- 真实 DAS 应变响应还需考虑光纤方向、gauge length 和耦合。

`road_void/numerics/fdtd.py` 另有一个 1D 标量波 FDTD benchmark 辅助，用于和 FEM/SEM 解同一个低维问题。它不是 elastic3d，也不是道路空洞主正演，只是统一对比用的最小有限差分算例。

后续可继续推进严格 staggered-grid、完整 CPML、DAS 应变响应和更系统的边界反射 benchmark。

## 3. FEM

有限元方法适合复杂几何、任意形状空洞和材料界面。基本对象包括：

- 网格和单元；
- 形函数；
- 质量矩阵 `M`；
- 刚度矩阵 `K`；
- 边界条件；
- 时间推进或频域求解。

本轮实现的是 `road_void/numerics/fem.py` 中的 1D 标量波教学原型：

```text
M u_tt + K u = f
```

它使用 1D 线性单元，组装质量矩阵和刚度矩阵，并用质量集总显式推进。它不是二维/三维弹性 FEM，只是帮助理解弱形式、单元矩阵和矩阵组装过程。

## 4. BEM

边界元方法适合边界散射问题。理论上，空洞或夹杂体的边界散射可以用边界积分描述，只需要离散边界而不是整个体域。

优点：

- 对无限域和边界散射有吸引力；
- 只离散边界，维度降低；
- Green 函数天然包含部分传播物理。

难点：

- 三维弹性半空间 Green 函数复杂；
- 奇异积分、自由表面和稠密矩阵处理难度高；
- 大规模 BEM 往往需要快速多极子、H-matrix 等技术。

本轮实现的是 `road_void/numerics/bem.py` 中的二维标量边界积分思想演示。它生成圆形边界点，构造简化 Green 函数矩阵，并计算接收线散射响应。当前 BEM demo 是标量边界积分思想演示，不是三维弹性边界元。

## 5. SEM / 谱元 / 谱元素

谱元方法可以看成高阶有限元：单元内使用高阶 Lagrange 多项式，节点常放在 Gauss-Lobatto-Legendre 点上。GLL 点与 GLL 积分结合时，质量矩阵常接近对角或可质量集总，适合显式波动方程。

SPECFEM3D 是成熟参考，但本项目不直接复刻。当前 `road_void/numerics/sem.py` 只实现 1D 标量谱元教学原型：

- 生成 GLL 节点和权重；
- 构造 Lagrange 导数矩阵；
- 组装 1D 质量和刚度矩阵；
- 运行小型标量波传播。

它不是完整三维弹性 SEM，也不包含真实复杂地形或非结构网格。

## 6. DG / ADER-DG

SeisSol 是高阶 ADER-DG 大规模弹性波模拟的重要参考。DG 方法适合复杂几何、非结构网格、高阶近似和局部时间步长。

本轮不实现 DG，只把它作为未来路线说明。若后续推进 DG，应先从 1D/2D 标量守恒律或声波方程开始，而不是直接写三维弹性 ADER-DG。

## 7. 与现有模型关系

必须强调：

```text
现有运动学散射点模型和 elastic3d 不是废弃，而是继续保留；
高级数值方法是新增研究路线；
不能用未验证的高级模块替代现有 workflow。
```

推荐使用方式：

```bash
python main.py              # 默认/本地 workflow
python main.py numerics-demo --method fem --no-save
python main.py numerics-demo --method sem --no-save
python main.py numerics-demo --method bem --no-save
python main.py numerics-demo --method all --save
python main.py numerics-compare --save
```

输出图件默认位于：

```text
outputs/numerics/fem1d_wavefield.png
outputs/numerics/fem1d_receiver_trace.png
outputs/numerics/sem1d_wavefield.png
outputs/numerics/sem1d_receiver_trace.png
outputs/numerics/bem2d_boundary_points.png
outputs/numerics/bem2d_scattered_response.png
outputs/numerics/compare_1d_traces.png
outputs/numerics/compare_1d_wavefields.png
outputs/numerics/compare_1d_metrics.json
```

## 8. FDTD/FEM/SEM 统一 1D benchmark

为了避免 FEM、SEM、FDTD 各自“能运行”但不能相互比较，本项目新增：

```bash
python main.py numerics-compare --save
```

统一问题为 1D 均匀标量波方程：

```text
u_tt = c^2 u_xx + f
```

统一参数包括：

- `length`：模型长度；
- `velocity`：标量波速度；
- `duration` 和 `dt`：记录时长与时间步长；
- `source_position` 和 `receiver_position`：源点与接收点；
- `source_frequency`：Ricker 点源主频；
- 固定端边界条件。

三种方法分别为：

- FDTD：二阶中心差分显式推进；
- FEM：1D 线性单元，组装质量矩阵和刚度矩阵，质量集总后显式推进；
- SEM：GLL 节点和高阶 Lagrange 基函数，组装谱元质量/刚度并显式推进。

输出的 `compare_1d_metrics.json` 包含：

```json
{
  "fdtd_arrival_time": "...",
  "fem_arrival_time": "...",
  "sem_arrival_time": "...",
  "fem_vs_fdtd_l2": "...",
  "sem_vs_fdtd_l2": "..."
}
```

正确阅读方式是：三种方法的主要到时应在同一量级，波形细节允许不同；如果某条记录爆炸、全零或首次到时严重错位，应先检查数值稳定性和离散参数，而不是把结果解释成物理差异。

该 benchmark 仍然是低维标量波教学对比，不代表完整三维弹性求解器。当前最接近道路三维全波场的仍是 `elastic3d`，而默认定位主线仍是快速三维运动学/属性 workflow。

## 9. 参考资料

- FEM 的质量矩阵、刚度矩阵和弱形式可参考有限元方法教材与概述：[Finite element method](https://en.wikipedia.org/wiki/Finite_element_method)、[Stiffness matrix](https://en.wikipedia.org/wiki/Stiffness_matrix)。
- SEM 的 GLL 节点、高阶 Lagrange 基函数和质量集总思想可参考：[Spectral element method](https://en.wikipedia.org/wiki/Spectral_element_method) 以及 SPECFEM/Komatitsch 系列工作。
- BEM 的边界积分和 Green 函数矩阵思想可参考：[Boundary element method](https://en.wikipedia.org/wiki/Boundary_element_method)。
- Devito 展示了有限差分 PDE DSL 在地震波和反演中的路线：[Devito finite-difference DSL](https://arxiv.org/abs/1609.03361)。
- SeisSol/ADER-DG 代表高阶非结构网格弹性波路线，可参考 ADER-DG 地震波模拟相关资料。
