# FDTD / FEM / SEM 1D 标量波 benchmark 说明

本 benchmark 的目标是把三个低维教学原型放到同一个可比较的问题下，而不是把它们包装成完整三维弹性求解器。

## 统一问题

方程为：

```text
u_tt = c^2 u_xx + f
```

统一参数包括：

- `length`：1D 模型长度，单位 m；
- `velocity`：标量波速度，单位 m/s；
- `duration`：记录时长，单位 s；
- `dt`：时间步长，单位 s；
- `source_position`：点源位置，单位 m；
- `receiver_position`：接收点位置，单位 m；
- `source_frequency`：Ricker 点源主频，单位 Hz。

三种方法都使用固定端边界，因此后期会有边界反射。当前主要比较首次到时和短时波形，不把后期反射当作道路物理现象解释。

## 三种离散方法

- FDTD：二阶中心差分显式推进；
- FEM：1D 线性单元，组装质量矩阵 `M` 和刚度矩阵 `K`，质量集总后显式推进；
- SEM：GLL 节点和高阶 Lagrange 基函数，组装谱元质量/刚度并显式推进。

## 运行方式

```bash
python main.py numerics-compare --save
python main.py numerics-compare --show --no-save
```

输出文件：

```text
outputs/numerics/compare_1d_traces.png
outputs/numerics/compare_1d_wavefields.png
outputs/numerics/compare_1d_metrics.json
```

`compare_1d_metrics.json` 包含首次到时、峰值振幅和归一化 L2 差异。三种方法的主要到时应接近，但波形细节不要求完全一致。

## 解释边界

这个 benchmark 只是数值方法 sanity check：

- 它不是道路空洞主正演；
- 它不是三维弹性模拟；
- 它不替代 `elastic3d`；
- BEM 当前仍是二维标量边界积分思想 demo，不参与该 1D 波动方程对比。

如果某个方法的记录全零、爆炸或首次到时严重错位，应先检查 CFL、时间步长、空间离散和源/接收点位置，而不是把差异解释为物理异常。
