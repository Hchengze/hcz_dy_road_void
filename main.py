"""城市道路空洞 DAS 三维正演与定位研究原型主入口。

本项目定位为“本地可直接运行、测试、修改和理解”的算法研究原型，
因此参数主要放在本文件的 argparse 默认值和中文 help 中，而不是要求
用户到多个配置文件里寻找。运行示例：

    python main.py geometry --show --no-save
    python main.py forward --save
    python main.py scan --road-width 30 --cavity-depth 2.5 --noise-level 0.1 --save
    python main.py tutorial --save
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from shutil import rmtree

import numpy as np

from road_void.config import CavityConfig, GeometryConfig, NoiseConfig, ProcessingConfig, RecordConfig, RoadVoidConfig, VelocityConfig
from road_void.dataset import generate_synthetic_survey_dataset, plot_das_like_gather, plot_noise_components, save_synthetic_survey_dataset
from road_void.diffraction import detect_diffraction_features, plot_diffraction_attribute, plot_diffraction_candidates
from road_void.elastic_bridge import build_local_elastic_validation_case, plot_elastic_validation_outputs, run_elastic_validation_case
from road_void.elastic3d import Elastic3DConfig, animate_elastic3d_wavefield, plot_abc_comparison, plot_elastic3d_outputs, run_elastic3d
from road_void.fwi import plot_fwi_demo_outputs, run_fwi_misfit_demo
from road_void.inversion import plot_localization_error_summary, plot_uncertainty_summary, run_joint_localization_evaluation, write_research_report
from road_void.numerics import run_bem2d_scatter_demo, run_fem1d_wave_demo, run_sem1d_wave_demo
from road_void.numerics.compare import compare_1d_wave_methods
from road_void.scenario import build_default_subsurface_scenario, plot_subsurface_3d, plot_subsurface_sections
from road_void.visualization import (
    animate_kinematic_wavefield,
    animate_kinematic_wavefield_3d,
    animate_multishot_kinematic_wavefield,
    plot_kinematic_wavefield_frames_3d,
    plot_kinematic_wavefield_frames,
    plot_multishot_wavefield_frames,
    plot_diffraction_path_demo,
    plot_geometry_plan_and_sections,
    plot_multishot_scan_diagnostics,
    plot_road_geometry_3d,
    plot_score_slices,
    plot_shot_gather,
    plot_velocity_model,
)
from road_void.workflow import run_location_workflow, simulate_from_config


# ============================================================
# VSCode 本地运行配置：默认运行完整 workflow
# ============================================================
#
# 这个区域是本地算法研究的“唯一常用配置入口”。直接运行 ``python main.py``
# 时会读取 LOCAL_RUN_MODE、LOCAL_OUTPUT 和 LOCAL_WORKFLOW；如果显式输入
# ``python main.py scan --no-save`` 等命令行参数，则仍然优先使用 argparse。
USE_LOCAL_DEBUG_CONFIG = True

# ============================================================
# LOCAL_RUN_MODE 模式说明
# ============================================================
#
# workflow:
#   推荐主入口。运行完整道路空洞定位流程：
#   geometry -> velocity -> forward -> path -> scan -> result plots。
#   默认输出到 outputs/workflow/，适合日常查看完整结果。
#   重点检查 LOCAL_WORKFLOW["geometry"], ["velocity"], ["anomaly"], ["scan"]。
#
# all:
#   workflow 的别名，不额外运行 numerics/elastic3d，也不生成重复输出。
#   保留给习惯输入 all 的场景；日常推荐仍用 workflow。
#
# geometry:
#   只画道路、光纤、炮点、异常体几何图。
#   用于检查 road_width、road_length、cavity_x/y/depth 是否设置正确。
#
# velocity:
#   只画当前运动学正演/扫描使用的速度模型。
#   uniform 显示单一 VR；layered-effective 显示分层速度和 VR_eff。
#
# forward:
#   只生成合成炮集/shot gather。
#   用于检查直达波、散射/绕射事件、噪声和异常体响应。
#
# wavefield:
#   只生成 workflow 第 6 步的波场示意，默认写入 outputs/workflow/。
#   默认是 x-y 地表平面运动学示意；--wavefield-view 3d 只画三维等时面示意。
#
# path:
#   只画 S-G 直达路径与 S-D-G 绕射路径示意，用于核查炮点、异常体、接收点几何关系。
#
# scan:
#   只运行三维绕射扫描定位。重点检查 scan_x/y/h/velocity 范围是否覆盖异常体。
#
# sensitivity:
#   参数敏感性分析入口，适合比较 velocity、noise、depth、scan step 等参数影响。
#
# tutorial:
#   教学示例入口，用于学习流程，不建议作为正式结果判断。
#
# elastic3d:
#   小尺度 3D elastic FDTD 原型，不是默认 workflow。
#   它的网格范围可能和道路几何不同，重点检查 dx/dy/dz、dt、nt、CFL、异常体是否落入小模型。
#
# elastic-validate:
#   从 workflow 的 RoadVoidConfig 裁剪/平移一个局部区域，生成小尺度 elastic3d validation case。
#   用于检查主异常体附近的全波场 sanity check，不是把整条道路直接放进 elastic3d。
#
# fwi-demo:
#   只做 L2 misfit 曲线教学演示，不是完整伴随 FWI，不做模型更新。
#
# numerics-demo:
#   FEM/SEM/BEM 低维标量教学原型，不是三维弹性模拟。
#
# numerics-compare:
#   FDTD/FEM/SEM 统一 1D 标量波 benchmark，用于比较教学原型，不参与道路空洞 workflow。
LOCAL_RUN_MODE_NOTES = """
workflow/all/geometry/velocity/forward/wavefield/path/scan/sensitivity/tutorial/
elastic3d/elastic-validate/fwi-demo/numerics-demo/numerics-compare 模式说明见 main.py 顶部注释。
"""
LOCAL_RUN_MODE = "workflow"

LOCAL_OUTPUT = dict(
    save=True,              # True 时保存 workflow 主线图件；False 时只跑流程和控制台摘要。
    show=False,             # True 时弹出 matplotlib 窗口，适合手动看图调参。
    animate=False,         # True 时才生成 GIF；默认关闭，避免输出过多。
    save_extra=False,       # True 才保存诊断图、run_parameters 等额外文件。
    clean_output=True,      # True 时只清理当前 outdir 中旧图，不碰其它输出目录。
    outdir="outputs/workflow",
    dpi=150,
)

LOCAL_WORKFLOW = dict(
    geometry=dict(
        road_width=30.0,       # 道路横向宽度 W，单位 m；控制锤击线到光纤线的横向孔径。
        road_length=180.0,      # 沿道路模拟长度，单位 m；需覆盖通道、炮点和异常体。
        channel_spacing=1.0,   # DAS 通道间距，单位 m；越小沿 x 采样越密。
        source_spacing=4.0,    # 锤击点距，单位 m；越小多炮约束越强但采集工作量越大。
    ),
    velocity=dict(
        rayleigh_velocity=240.0,          # 等效瑞雷波速度 VR，单位 m/s。
        source_frequency=25.0,            # 锤击主频 f，单位 Hz；lambda=VR/f。
        velocity_mode="layered-effective",# "uniform" 或 "layered-effective"。
        layer_depths="0.4,1.5,6",       # 层底深度，单位 m；layered-effective 使用。
        layer_velocities="180,240,320",   # 每层等效瑞雷速度，单位 m/s。
        sensitivity_depth_factor=0.5,     # z_sensitive≈alpha*lambda。
    ),
    anomaly=dict(
        enable_cavity=True,          # False 时等价于命令行 --no-cavity，用于无异常误报检查。
        cavity_x=42.0,               # 单异常体 x0，单位 m；主要控制绕射顶点沿道路位置。
        cavity_y=15,                # 单异常体 y0，单位 m；单侧 DAS 下与深度 h 存在耦合。
        cavity_depth=4,            # 单异常体顶部/主散射中心深度 h，单位 m。
        cavity_radius=2.0,           # sphere/cylinder 默认半径，单位 m；也作为 line/zone 的辅助尺度。
        cavity_shape="sphere",       # sphere/box/cylinder/ellipsoid/line/zone。
        cavity_size_x=4.0,           # box/ellipsoid 的 x 向尺寸；line/zone 的长度；单位 m。
        cavity_size_y=3.0,           # box/ellipsoid/zone 的 y 向尺寸；单位 m。
        cavity_size_z=2.0,           # box/ellipsoid/cylinder 的竖向尺寸或高度；单位 m。
        cavity_azimuth=0.0,          # line/zone 方位角，单位度；0 表示沿 x 方向。
        scattering_strength=1.0,     # 散射强度；越大绕射/散射事件越明显。
        attenuation_strength=0.25,   # 阴影/衰减强度；越大直达波局部能量下降越明显。
        tail_strength=0.2,           # 散射尾波强度；越大尾波更长，但过大会掩盖主事件。
        anomalies="",                # 非空时优先使用多异常体字符串，忽略上面的单异常体参数。
    ),
    noise=dict(
        noise_level=0.03,
        traffic_noise_level=0.015,
        bad_channel_fraction=0.02,
        weak_coupling_fraction=0.06,
        coupling_variation=0.08,
    ),
    scan=dict(
        # scan_mode: 三维绕射扫描使用哪些炮。joint 为默认多炮联合评分；
        # single-shot 只看 shot_index 指定的一炮，适合观察单炮定位不稳定性；
        # compare 会同时保留单炮与联合结果，适合教学对比。
        scan_mode="joint",
        # shot_index: single-shot 模式使用的炮号，从 0 开始；应选几何上能照亮
        # 异常体的炮点。若离异常体太远，绕射能量弱，定位会更不稳定。
        shot_index=5,
        # shot_weight_mode: 多炮权重。uniform 等权最稳妥；near-cavity 对靠近候选
        # x0 的炮加权；snr 用记录能量近似信噪比，噪声复杂时可能有偏置。
        shot_weight_mode="uniform",
        # scan_x_min / scan_x_max: 沿道路/光纤方向 x0 搜索范围，单位 m。
        # 如果 cavity_x 不在范围内，扫描不可能找回真实异常体。
        scan_x_min=20.0,
        scan_x_max=70.0,
        # scan_x_step: x0 搜索步长，单位 m；越小定位网格越细但计算越慢。
        scan_x_step=2.0,
        # scan_y_min / scan_y_max: 横穿道路方向 y0 搜索范围，单位 m。
        # 应覆盖光纤 y=0 到锤击线 y=road_width 之间的疑似区域。
        scan_y_min=2.0,
        scan_y_max=22.0,
        # scan_y_step: y0 搜索步长，单位 m；单侧 DAS 下 y 与 h 耦合明显，
        # 因此建议看范围和 top-k，而不是只看一个最优点。
        scan_y_step=2.0,
        # scan_h_min / scan_h_max: 顶部埋深/主散射中心深度 h 搜索范围，单位 m。
        # 如果 cavity_depth 不在范围内，深度结果会被截到边界。
        scan_h_min=0.5,
        scan_h_max=5.0,
        # scan_h_step: 深度搜索步长，单位 m；越小深度分辨更细但计算量增加。
        # 过大可能让真实深度附近没有采样点。
        scan_h_step=0.3,
        # scan_vr_min / scan_vr_max: 等效瑞雷速度搜索范围，单位 m/s。
        # 用于考虑直达波速度拟合误差和 layered-effective 的 VR_eff 不确定性。
        scan_vr_min=220.0,
        scan_vr_max=260.0,
        # scan_vr_step: 速度搜索步长，单位 m/s；过粗会低估速度-深度耦合。
        scan_vr_step=10.0,
        # score_method: envelope 对相位误差不敏感，适合合成数据初期验证；
        # energy 更直接但更容易受残余直达波和噪声影响。
        score_method="envelope",
        # top_k: 输出前 k 个候选点，用于观察非唯一性和 y-h tradeoff。
        top_k=8,
        # uncertainty_threshold: 用 max_score * 阈值 定义不确定性范围。
        # 值越高范围越窄，但可能低估受限孔径下的真实不确定性。
        uncertainty_threshold=0.92,
    ),
    wavefield=dict(
        wavefield_view="plan",             # plan: x-y 地表平面运动学示意；3d: 三维等时面几何示意，仍不是 elastic3d。
        wavefield_mode="single-shot",      # single-shot 只看一炮；multi-shot 顺序展示多炮覆盖。
        wavefield_shot_index=None,         # 单炮模式指定炮号；None 时自动选靠近主异常体的炮。
        wavefield_shot_indices="0,5,10",   # 多炮显式炮号，如 "0,5,10"；非空时优先。
        wavefield_max_shots=5,             # 多炮最多展示几炮，避免 GIF 过大。
        wavefield_shot_step=5,             # 未指定 indices 时，每隔几炮选一炮。
        wavefield_shot_interval=0.25,      # 多炮 GIF 的示意全局时间间隔，单位 s。
    ),
)

LOCAL_ELASTIC3D = dict(
    # nx/ny/nz: elastic3d 小模型网格数。物理范围约为
    # (nx*dx, ny*dy, nz*dz)，不等同于 workflow 的 road_length/road_width。
    # 如果把道路尺度异常体直接传入 elastic3d，必须检查坐标是否落在该小模型内。
    nx=56,
    ny=36,
    nz=28,
    # dx/dy/dz: 三维网格间距，单位 m；越小空间分辨率越高，但 CFL 更严格、计算更慢。
    dx=0.5,
    dy=0.5,
    dz=0.5,
    # elastic_dt: 时间步长，单位 s；必须满足 CFL = vmax*dt*sqrt(1/dx^2+1/dy^2+1/dz^2)。
    elastic_dt=0.00012,
    # elastic_nt: 时间步数；越大记录越长，但计算更慢，也更容易看到边界反射。
    elastic_nt=420,
    # elastic3d 内部当前使用三层 Vp/Vs/rho 教学模型：
    # Vp≈520/720/900 m/s，Vs≈220/320/420 m/s，rho≈1800/1900/2050 kg/m3。
    # 这些不是 workflow 的等效瑞雷速度 VR；二者属于不同正演层级。
    # elastic_source_frequency: Ricker 垂向锤击源主频，单位 Hz；越高波长越短、数值频散要求越严格。
    elastic_source_frequency=60.0,
    # elastic_source_amplitude: 速度场中加载的等效垂向力源幅度；过大会导致图件饱和或数值异常。
    elastic_source_amplitude=1.0e5,
    # elastic_space_order: 空间差分阶数。2 阶更稳更快；4 阶内部频散较低，边界附近仍会降阶。
    elastic_space_order=2,
    # elastic_abc: 吸收边界。sponge 是默认稳定海绵层；cpml 是 experimental cpml-like 阻尼，
    # 不是完整带记忆变量的严格 CPML，不应作为工业级边界吸收结论。
    elastic_abc="sponge",
    # elastic_record_component: 接收记录分量。vz/vx 是速度；strain_xx≈du_x/dx；
    # strain_rate_xx≈dv_x/dx，是沿 x 方向光纤 DAS 应变率的粗略近似。
    elastic_record_component="vz",
    # elastic_gauge_length: DAS 近似 gauge length，单位 m；用于应变/应变率的有限长度差分。
    elastic_gauge_length=1.0,
    # elastic3d 默认异常体来自 workflow 的 Cavity 参数；它会被映射为低速低密度体。
    # 由于 elastic3d 是小网格，务必确认 anomaly_x/y/z 和 radius 落入 nx*dx, ny*dy, nz*dz 范围。
)

LOCAL_FWI = dict(
    fwi_vs_scales="0.86,0.92,0.98,1.0,1.04",
    fwi_observed_vs_scale=1.0,
    fwi_initial_vs_scale=0.9,
)

LOCAL_NUMERICS = dict(
    method="all",          # "fem", "sem", "bem", "all"。
    numerics_length=100.0, # 1D FEM/SEM 标量波模型长度，单位 m。
    numerics_velocity=300.0,
    numerics_duration=0.24,
)

LOCAL_NUMERICS_COMPARE = dict(
    numerics_length=100.0,             # 统一 1D 标量波 benchmark 长度，单位 m。
    numerics_velocity=300.0,           # 三种方法使用同一标量波速度，单位 m/s。
    numerics_duration=0.24,            # 记录时长，单位 s；短时避免固定边界反射主导。
    numerics_dt=0.0005,                # 统一时间步长，单位 s；需满足 FDTD/FEM/SEM 稳定性。
    numerics_source_position=25.0,     # 点源位置，单位 m。
    numerics_receiver_position=75.0,   # 接收点位置，单位 m。
    numerics_source_frequency=35.0,    # Ricker 源主频，单位 Hz。
)

WORKFLOW_STEPS = [
    ("geometry", "建立道路三维几何：道路、光纤、炮线、空洞"),
    ("velocity", "展示等效瑞雷波速度/速度模型，并说明 lambda = VR / f"),
    ("forward", "锤击激发三维等效瑞雷波正演，生成单侧 DAS shot gather"),
    ("path", "展示 S-G 直达路径与 S-D-G 绕射路径及走时公式"),
    ("scan", "直达波拟合、残差构建、绕射扫描和疑似空洞定位"),
    ("wavefield", "可选生成等效运动学波场动画，不作为高保真弹性波场"),
    ("summary", "输出结果解释、受限孔径提醒和参数记录"),
]


def build_args_from_local_config(run_mode: str | None = None) -> argparse.Namespace:
    """把 ``LOCAL_*`` 本地调试参数转换为 argparse Namespace。

    这样 VSCode 直接运行和命令行运行复用同一个 parser、同一批 ``run_xxx``
    函数，避免维护两套入口逻辑。
    """

    mode = run_mode or LOCAL_RUN_MODE
    argv = _local_config_to_argv(mode)
    return build_parser().parse_args(argv)


def _local_config_to_argv(mode: str) -> list[str]:
    params: dict[str, object] = {}
    params.update(LOCAL_WORKFLOW["geometry"])
    params.update(LOCAL_WORKFLOW["velocity"])
    params.update(LOCAL_WORKFLOW["anomaly"])
    params.update(LOCAL_WORKFLOW["noise"])
    if mode in {"workflow", "scan", "sensitivity", "tutorial", "all"}:
        params.update(LOCAL_WORKFLOW["scan"])
    if mode in {"workflow", "wavefield", "tutorial", "all"}:
        params.update(LOCAL_WORKFLOW["wavefield"])
    if mode in {"elastic3d", "elastic-validate"}:
        params.update(LOCAL_ELASTIC3D)
    if mode == "fwi-demo":
        params.update(LOCAL_ELASTIC3D)
        params.update(LOCAL_FWI)
    if mode == "numerics-demo":
        params.update(LOCAL_NUMERICS)
    if mode == "numerics-compare":
        params.update(LOCAL_NUMERICS_COMPARE)

    argv = [mode]
    for key, value in params.items():
        if value is None or value == "":
            continue
        if key == "enable_cavity":
            if not bool(value):
                argv.append("--no-cavity")
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    if bool(LOCAL_OUTPUT["save"]):
        argv.append("--save")
    else:
        argv.append("--no-save")
    if bool(LOCAL_OUTPUT["show"]):
        argv.append("--show")
    else:
        argv.append("--no-show")
    if bool(LOCAL_OUTPUT["animate"]) and mode in {"workflow", "wavefield", "tutorial", "elastic3d", "all"}:
        argv.append("--animate")
    if bool(LOCAL_OUTPUT["save_extra"]):
        argv.append("--save-extra")
    if bool(LOCAL_OUTPUT["clean_output"]):
        argv.append("--clean-output")
    argv.extend(["--outdir", str(LOCAL_OUTPUT["outdir"]), "--dpi", str(LOCAL_OUTPUT["dpi"])])
    return argv


def print_local_run_summary(args: argparse.Namespace) -> None:
    """在 VSCode 输出窗口打印当前本地运行模式和关键参数。"""

    print("=" * 64)
    print("VSCode/本地调试模式")
    print(f"当前运行模式：{args.command}")
    print(f"输出模式：show={bool(args.show)}, save={bool(args.save) and not bool(args.no_save)}, animate={getattr(args, 'animate', False)}")
    if hasattr(args, "road_width"):
        print(f"道路宽度 W = {args.road_width} m")
        print(f"等效瑞雷波速度 VR = {args.rayleigh_velocity} m/s")
        anomaly = args.anomalies or f"{args.cavity_shape}@({args.cavity_x},{args.cavity_y},{args.cavity_depth})"
        print(f"异常体 = {anomaly}")
    if hasattr(args, "scan_mode"):
        print(f"扫描模式 = {args.scan_mode}, shot_weight_mode = {args.shot_weight_mode}")
    if args.command == "elastic3d":
        print(
            "elastic3d 参数 = "
            f"space_order={args.elastic_space_order}, abc={args.elastic_abc}, "
            f"record_component={args.elastic_record_component}, gauge_length={args.elastic_gauge_length}"
        )
    if args.command == "numerics-demo":
        print(f"高级数值方法 demo = {args.method}")
    if args.command == "numerics-compare":
        print(
            "数值方法对比 = "
            f"L={args.numerics_length} m, c={args.numerics_velocity} m/s, "
            f"xs={args.numerics_source_position} m, xr={args.numerics_receiver_position} m"
        )
    print("=" * 64)


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """添加通用输出参数。"""

    parser.add_argument("--save", action="store_true", help="保存图件到 --outdir。适合批量运行和写汇报材料。")
    parser.add_argument("--no-save", action="store_true", help="不保存图件。用于快速测试，优先级高于 --save。")
    parser.add_argument("--save-extra", action="store_true", help="额外保存诊断图/对比图/中间图。默认只保存当前子命令的必要图件，避免输出爆炸。")
    parser.add_argument("--clean-output", action="store_true", help="运行前清理当前 outdir 中旧的 png/gif/json/txt 文件；不会清理全局 outputs/ 或其它子目录。")
    parser.add_argument("--show", action="store_true", help="用 matplotlib 交互窗口显示图件，适合本地调参时观察细节。")
    parser.add_argument("--no-show", action="store_true", help="不弹出交互窗口。用于脚本批量运行。")
    parser.add_argument("--outdir", default=None, help="输出目录。若不指定，则按子命令写入 outputs/<subcommand>/。")
    parser.add_argument("--dpi", type=int, default=180, help="保存图片分辨率，单位 dpi；汇报图可设为 200-300。")


def add_geometry_args(parser: argparse.ArgumentParser) -> None:
    """道路、光纤、锤击线和空洞几何参数。"""

    parser.add_argument("--road-width", type=float, default=15.0, help="道路横向宽度 W，单位 m；也是锤击线与光纤线的横向距离。越宽路径越长、能量越弱，y-h 耦合越明显。")
    parser.add_argument("--road-length", type=float, default=80.0, help="道路沿线模拟长度，单位 m；应覆盖锤击点、DAS 通道和疑似异常体所在范围。")
    parser.add_argument("--channel-spacing", type=float, default=1.0, help="DAS 通道间距，单位 m；越小沿道路采样越密，绕射曲线越容易识别，但数据量更大。")
    parser.add_argument("--source-spacing", type=float, default=4.0, help="锤击点间距，单位 m；越小多炮约束越强，过大会漏掉局部异常最佳激发位置。")
    parser.add_argument("--fiber-depth", type=float, default=0.0, help="光纤埋深，单位 m；当前默认为地表，真实数据需由管线资料或标定获得。")
    parser.add_argument("--source-depth", type=float, default=0.0, help="锤击源深度，单位 m；锤击通常近似为地表 0 m。")
    parser.add_argument("--cavity-x", type=float, default=42.0, help="异常体沿道路方向位置 x0，单位 m；主要控制绕射曲线顶点沿 x 的位置。")
    parser.add_argument("--cavity-y", type=float, default=8.5, help="异常体横向位置 y0，单位 m；单侧 DAS 下与深度 h 存在耦合，解释时应给范围。")
    parser.add_argument("--cavity-depth", type=float, default=2.2, help="异常体顶部/主要散射中心深度 h，单位 m；越深绕射到时越晚、能量越弱。")
    parser.add_argument("--cavity-radius", type=float, default=2.0, help="有效异常体半径，单位 m；控制散射影响范围，不代表真实空洞几何边界。")
    parser.add_argument("--cavity-shape", choices=["sphere", "box", "cylinder", "ellipsoid", "line", "zone"], default="sphere", help="单异常体形状。当前只是等效散射点集合：sphere 近圆形空洞，box 井室/箱涵，cylinder 管线/管沟，ellipsoid 脱空，line/zone 长条松散带。")
    parser.add_argument("--cavity-size-x", type=float, default=None, help="单异常体 x 向尺寸，单位 m；box/ellipsoid 使用完整尺寸，line/zone 表示长度，cylinder 可忽略。")
    parser.add_argument("--cavity-size-y", type=float, default=None, help="单异常体 y 向尺寸，单位 m；box/ellipsoid/zone 使用，sphere 通常忽略。")
    parser.add_argument("--cavity-size-z", type=float, default=None, help="单异常体 z 向尺寸或高度，单位 m；cylinder 表示高度，box/ellipsoid 表示竖向尺寸。")
    parser.add_argument("--cavity-azimuth", type=float, default=0.0, help="line/zone 的平面方位角，单位度；0 表示沿 x 方向，90 表示沿 y 方向。")
    parser.add_argument("--anomalies", default=None, help="多个异常体输入，例如 'sphere:42,8.5,2.2,2.0,1.0;box:58,6,1.5,4,3,1,0.8'。若提供，优先于单个 --cavity-* 参数。")
    parser.add_argument("--no-cavity", action="store_true", help="关闭空洞散射，用于检查无异常情况下的误报风险。")


def add_wave_args(parser: argparse.ArgumentParser) -> None:
    """速度、频率、采样和噪声参数。"""

    parser.add_argument("--rayleigh-velocity", type=float, default=240.0, help="等效瑞雷波速度 VR，单位 m/s；t_direct=t0+|S-G|/VR，t_diff=t0+(|S-D|+|D-G|)/VR。VR 偏大可能使深度偏深，偏小可能使深度偏浅。")
    parser.add_argument("--velocity-mode", choices=["uniform", "layered-effective"], default="uniform", help="速度模式。uniform 使用单一 VR；layered-effective 根据 lambda=VR/f 和指数敏感深度权重把层状速度折算为 VR_eff，并用于正演和扫描走时。")
    parser.add_argument("--layer-depths", default="0.4,1.5,4.0", help="层底深度列表，单位 m，例如 0.4,1.5,4.0；用于 layered-effective 和速度图。")
    parser.add_argument("--layer-velocities", default="180,240,320", help="每层等效瑞雷速度，单位 m/s；低频波长长时深层权重更大，高频更受浅层速度控制。")
    parser.add_argument("--sensitivity-depth-factor", type=float, default=0.5, help="敏感深度因子 alpha；z_sensitive≈alpha*lambda。越大表示面波走时受深层速度影响越明显。")
    parser.add_argument("--source-frequency", type=float, default=35.0, help="锤击主频 f，单位 Hz；波长 lambda=VR/f。频率高分辨率好但衰减强，频率低探测更深但分辨率降低。")
    parser.add_argument("--wavelet", choices=["ricker", "hammer"], default="ricker", help="震源子波类型；ricker 便于教学，hammer 更像短促锤击脉冲。")
    parser.add_argument("--sampling-rate", type=float, default=1000.0, help="采样率，单位 Hz；应明显高于主频两倍，否则高频锤击信号会混叠。")
    parser.add_argument("--duration", type=float, default=1.0, help="记录时长，单位 s；太短会截断直达波、绕射波或尾波。")
    parser.add_argument("--t0", type=float, default=0.02, help="触发时间或系统时延，单位 s；误差会整体平移理论走时曲线。")
    parser.add_argument("--random-seed", type=int, default=2027, help="随机种子；控制噪声、坏道和弱耦合通道，使结果可复现。")
    parser.add_argument("--noise-level", type=float, default=0.03, help="随机背景噪声强度；越大绕射事件越难识别，扫描置信度通常下降。")
    parser.add_argument("--traffic-noise-level", type=float, default=0.015, help="交通类低频干扰强度；模拟城市道路车辆和环境振动影响。")
    parser.add_argument("--bad-channel-fraction", type=float, default=0.02, help="坏道比例，0-1；坏道越多，相干扫描越不稳定。")
    parser.add_argument("--weak-coupling-fraction", type=float, default=0.06, help="弱耦合通道比例；模拟运营商光纤耦合不均。")
    parser.add_argument("--coupling-variation", type=float, default=0.08, help="通道增益随机变化幅度；越大，道间振幅越不一致。")
    parser.add_argument("--scattering-strength", type=float, default=1.0, help="异常体散射强度；越大绕射/散射事件越清楚。")
    parser.add_argument("--attenuation-strength", type=float, default=0.25, help="异常体衰减/阴影强度；越大，异常体附近直达波能量下降越明显。")
    parser.add_argument("--tail-strength", type=float, default=1.0, help="异常体散射尾波强度；越大尾波越明显，但过大会掩盖主绕射事件。")


def add_scan_args(parser: argparse.ArgumentParser) -> None:
    """三维绕射扫描参数。"""

    parser.add_argument("--scan-x-min", type=float, default=32.0, help="沿道路方向 x0 扫描最小值，单位 m；扫描范围必须覆盖疑似异常区。")
    parser.add_argument("--scan-x-max", type=float, default=52.0, help="沿道路方向 x0 扫描最大值，单位 m。")
    parser.add_argument("--scan-x-step", type=float, default=1.0, help="x0 扫描步长，单位 m；越小精度越高但计算更慢。")
    parser.add_argument("--scan-y-min", type=float, default=3.0, help="横向位置 y0 扫描最小值，单位 m；不宜超出道路横向范围。")
    parser.add_argument("--scan-y-max", type=float, default=14.0, help="横向位置 y0 扫描最大值，单位 m。")
    parser.add_argument("--scan-y-step", type=float, default=1.0, help="y0 扫描步长，单位 m；单侧 DAS 下 y0 与 h 常耦合，建议关注范围而非单点。")
    parser.add_argument("--scan-h-min", type=float, default=0.8, help="顶部埋深 h 扫描最小值，单位 m。")
    parser.add_argument("--scan-h-max", type=float, default=4.0, help="顶部埋深 h 扫描最大值，单位 m。")
    parser.add_argument("--scan-h-step", type=float, default=0.4, help="深度扫描步长，单位 m；越小深度搜索更细，但计算量增加，过大可能错过峰值。")
    parser.add_argument("--scan-vr-min", type=float, default=220.0, help="扫描等效瑞雷速度最小值，单位 m/s；用于考虑速度估计误差。")
    parser.add_argument("--scan-vr-max", type=float, default=260.0, help="扫描等效瑞雷速度最大值，单位 m/s。")
    parser.add_argument("--scan-vr-step", type=float, default=10.0, help="速度扫描步长，单位 m/s；过大会低估速度-深度耦合。")
    parser.add_argument("--score-method", choices=["envelope", "energy"], default="envelope", help="评分方法；envelope 对相位不敏感，energy 更直接但易受残余直达波影响。")
    parser.add_argument("--top-k", type=int, default=8, help="输出前 k 个候选点，用于观察非唯一性和 y-h tradeoff。")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.92, help="不确定性阈值；保留分数超过 max_score*阈值 的候选范围。")
    parser.add_argument("--direct-wave-mute-width", type=float, default=0.04, help="直达波模板拟合/压制半窗宽，单位 s；过宽可能误伤浅部绕射波。")
    parser.add_argument("--scan-mode", choices=["joint", "single-shot", "compare"], default="joint", help="扫描模式。joint 默认多炮联合；single-shot 只用 --shot-index 指定炮；compare 保留单炮最佳和联合结果用于对比。")
    parser.add_argument("--shot-index", type=int, default=None, help="single-shot 模式使用的炮号，从 0 开始；用于观察单炮孔径下定位不稳定性。")
    parser.add_argument("--shot-weight-mode", choices=["uniform", "near-cavity", "snr"], default="uniform", help="多炮权重。uniform 等权；near-cavity 对靠近候选 x0 的炮加权；snr 用记录能量作简化信噪比权重。")


def add_animation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--animate", action="store_true", help="生成 GIF 动画；适合汇报展示传播过程。")
    parser.add_argument("--no-animate", action="store_true", help="不生成 GIF，只在 --show 时显示关键帧。")
    parser.add_argument("--frames", type=int, default=48, help="GIF 帧数；越大动画更顺滑但文件更大。")
    parser.add_argument("--fps", type=int, default=10, help="GIF 帧率。")


def add_wavefield_args(parser: argparse.ArgumentParser) -> None:
    """等效运动学波场示意参数。"""

    parser.add_argument("--wavefield-view", choices=["plan", "3d"], default="plan", help="波场视图。plan 为默认 x-y 地表平面运动学示意；3d 为三维等时面/等时球面几何示意，仍不是完整弹性波场。")
    parser.add_argument("--wavefield-mode", choices=["single-shot", "multi-shot"], default="single-shot", help="波场示意模式。single-shot 只展示一炮；multi-shot 顺序展示多炮覆盖，但仍只是传播示意，不是联合反演。")
    parser.add_argument("--wavefield-shot-index", type=int, default=None, help="single-shot 使用的炮号；不设时自动选择靠近主异常体的炮。")
    parser.add_argument("--wavefield-shot-indices", default="", help="multi-shot 显式炮号列表，例如 0,5,10；非空时优先于 --wavefield-shot-step。")
    parser.add_argument("--wavefield-max-shots", type=int, default=5, help="multi-shot 最多展示几炮，避免 GIF 或关键帧过多。")
    parser.add_argument("--wavefield-shot-step", type=int, default=5, help="multi-shot 自动选择炮号时的炮间隔，例如每 5 炮取一炮。")
    parser.add_argument("--wavefield-shot-interval", type=float, default=0.25, help="multi-shot GIF 中相邻炮的示意全局时间间隔，单位 s。")


def add_elastic3d_args(parser: argparse.ArgumentParser) -> None:
    """小尺度三维弹性波全波形原型参数。"""

    parser.add_argument("--nx", type=int, default=56, help="elastic3d x 方向网格数；默认较小，保证本地快速运行。")
    parser.add_argument("--ny", type=int, default=36, help="elastic3d y 方向网格数。")
    parser.add_argument("--nz", type=int, default=28, help="elastic3d z 方向网格数，z 为深度。")
    parser.add_argument("--dx", type=float, default=0.5, help="x 网格间距，单位 m；越小分辨率越高但 CFL 更严格。")
    parser.add_argument("--dy", type=float, default=0.5, help="y 网格间距，单位 m。")
    parser.add_argument("--dz", type=float, default=0.5, help="z 网格间距，单位 m。")
    parser.add_argument("--elastic-dt", type=float, default=0.00012, help="elastic3d 时间步长，单位 s；必须满足 CFL 稳定条件。")
    parser.add_argument("--elastic-nt", type=int, default=420, help="elastic3d 时间步数；越大传播时间越长但计算更慢，默认保证波能到达接收线。")
    parser.add_argument("--elastic-source-frequency", type=float, default=60.0, help="elastic3d Ricker 震源主频，单位 Hz。")
    parser.add_argument("--elastic-source-amplitude", type=float, default=1.0e5, help="elastic3d 垂向力源幅度；过大可能导致图件饱和。")
    parser.add_argument("--elastic-space-order", type=int, choices=[2, 4], default=2, help="elastic3d 空间差分阶数。2 阶更稳更快；4 阶使用 9/8 与 -1/24 交错差分模板，内部频散更低，边界附近自动降阶。")
    parser.add_argument("--elastic-abc", choices=["sponge", "cpml"], default="sponge", help="elastic3d 吸收边界。sponge 为默认稳定海绵层；cpml 当前是 experimental CPML-like 阻尼，不是完整 CPML。")
    parser.add_argument("--elastic-record-component", choices=["vz", "vx", "strain_xx", "strain_rate_xx"], default="vz", help="elastic3d 接收记录分量。strain_rate_xx≈dvx/dx，是沿 x 方向光纤 DAS 应变率近似。")
    parser.add_argument("--elastic-gauge-length", type=float, default=1.0, help="DAS 近似 gauge length，单位 m；用于 strain_xx/strain_rate_xx 的有限长度空间差分。")
    parser.add_argument("--elastic-no-anomaly", action="store_true", help="elastic3d 中关闭低速低密度异常体，用于和有异常模型对比。")


def add_fwi_args(parser: argparse.ArgumentParser) -> None:
    """FWI 最小原型参数。"""

    parser.add_argument("--fwi-vs-scales", default="0.86,0.92,0.98,1.0,1.04", help="FWI-demo 候选 Vs 缩放因子列表。当前只做一维 misfit 曲线，不做伴随梯度。")
    parser.add_argument("--fwi-observed-vs-scale", type=float, default=1.0, help="生成目标数据 observed 的 Vs 缩放因子。")
    parser.add_argument("--fwi-initial-vs-scale", type=float, default=0.9, help="绘制 observed vs synthetic 对比时使用的初始模型 Vs 缩放因子。")


def add_numerics_args(parser: argparse.ArgumentParser) -> None:
    """高级数值方法教学 demo 参数。"""

    parser.add_argument("--method", choices=["fem", "sem", "bem", "all"], default="all", help="选择运行 FEM、SEM、BEM 或全部教学原型。")
    parser.add_argument("--numerics-length", type=float, default=100.0, help="FEM/SEM 1D 标量波模型长度，单位 m。")
    parser.add_argument("--numerics-velocity", type=float, default=300.0, help="FEM/SEM 标量波速度，单位 m/s。")
    parser.add_argument("--numerics-duration", type=float, default=0.24, help="FEM/SEM 时间长度，单位 s。")


def add_numerics_compare_args(parser: argparse.ArgumentParser) -> None:
    """FDTD/FEM/SEM 统一 1D 标量波 benchmark 参数。"""

    parser.add_argument("--numerics-length", type=float, default=100.0, help="统一 1D 标量波模型长度，单位 m；三种方法使用同一物理域。")
    parser.add_argument("--numerics-velocity", type=float, default=300.0, help="标量波速度 c，单位 m/s；理论到时约为 |xr-xs|/c。")
    parser.add_argument("--numerics-duration", type=float, default=0.24, help="记录时长，单位 s；短时 benchmark 主要比较首次到时。")
    parser.add_argument("--numerics-dt", type=float, default=0.0005, help="统一时间步长，单位 s；需满足 FDTD/FEM/SEM 显式推进稳定性。")
    parser.add_argument("--numerics-source-position", type=float, default=25.0, help="点源位置，单位 m。")
    parser.add_argument("--numerics-receiver-position", type=float, default=75.0, help="接收点位置，单位 m。")
    parser.add_argument("--numerics-source-frequency", type=float, default=35.0, help="Ricker 点源主频，单位 Hz；三种方法统一使用。")


def output_options(args: argparse.Namespace) -> dict[str, object]:
    save = bool(args.save) and not bool(args.no_save)
    show = bool(args.show) and not bool(args.no_show)
    return {"save": save, "show": show, "dpi": args.dpi}


def command_outdir(args: argparse.Namespace, name: str) -> Path:
    # wavefield 是 workflow 的第 6 步。未显式指定 --outdir 时，也写回
    # outputs/workflow，避免形成另一套独立输出体系。
    if name in {"workflow", "wavefield"} and not args.outdir:
        return Path("outputs") / "workflow"
    if name == "elastic3d_validation" and not args.outdir:
        return Path("outputs") / "elastic3d_validation"
    return Path(args.outdir) if args.outdir else Path("outputs") / name


class OutputManifest:
    """记录本次运行实际生成的文件，帮助区分新图和历史旧图。"""

    def __init__(self, outdir: Path, save: bool) -> None:
        self.outdir = outdir
        self.save = save
        self.files: list[Path] = []

    def add(self, path: Path | str, *, enabled: bool = True) -> Path:
        p = Path(path)
        if self.save and enabled:
            self.files.append(p)
        return p

    def write_and_print(self) -> None:
        if not self.files:
            print("本次实际生成文件：无（save=False 或 --no-save）。")
            return
        print("本次实际生成文件：")
        for idx, path in enumerate(self.files, start=1):
            print(f"{idx}. {path}")
        self.outdir.mkdir(parents=True, exist_ok=True)
        manifest = self.outdir / "output_manifest.txt"
        with manifest.open("w", encoding="utf-8") as f:
            for idx, path in enumerate(self.files, start=1):
                f.write(f"{idx}. {path}\n")
        print(f"输出清单: {manifest}")


def prepare_output_dir(args: argparse.Namespace, name: str) -> tuple[Path, dict[str, object], OutputManifest]:
    """统一处理 outdir、--clean-output 和输出清单。"""

    outdir = command_outdir(args, name)
    opts = output_options(args)
    if bool(getattr(args, "clean_output", False)):
        clean_output_dir(outdir)
    return outdir, opts, OutputManifest(outdir, bool(opts["save"]))


def clean_output_dir(outdir: Path) -> None:
    """只清理当前 outdir 中的常见结果文件，不碰其它 outputs 子目录。"""

    if not outdir.exists():
        return
    allowed = {".png", ".gif", ".mp4", ".json", ".txt", ".csv"}
    for item in outdir.iterdir():
        if item.is_file() and item.suffix.lower() in allowed:
            item.unlink()
        elif item.is_dir() and item.name.startswith("_tmp"):
            rmtree(item)
    print(f"已清理当前输出目录中的旧结果文件: {outdir}")


def parse_float_list(text: str) -> tuple[float, ...]:
    """解析命令行中的逗号分隔浮点数列表。"""

    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析浮点数列表: {text}") from exc
    if not values:
        raise argparse.ArgumentTypeError("浮点数列表不能为空。")
    return values


def parse_int_list(text: str) -> list[int]:
    """解析逗号分隔整数列表；空字符串返回空列表。"""

    if not text:
        return []
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无法解析整数列表: {text}") from exc


def select_wavefield_shots(args: argparse.Namespace, geom, cavities) -> list[int]:
    """根据 wavefield 参数选择单炮或多炮索引。"""

    n = geom.n_shots
    if n <= 0:
        return []
    mode = getattr(args, "wavefield_mode", "single-shot")
    if mode == "single-shot":
        if getattr(args, "wavefield_shot_index", None) is not None:
            return [max(0, min(n - 1, int(args.wavefield_shot_index)))]
        if cavities:
            return [min(range(n), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))]
        return [n // 2]
    explicit = parse_int_list(getattr(args, "wavefield_shot_indices", ""))
    if explicit:
        selected = [idx for idx in explicit if 0 <= idx < n]
    else:
        step = max(1, int(getattr(args, "wavefield_shot_step", 5)))
        selected = list(range(0, n, step))
    max_shots = max(1, int(getattr(args, "wavefield_max_shots", 5)))
    return selected[:max_shots]


def build_road_void_config_from_args(args: argparse.Namespace) -> RoadVoidConfig:
    """把 argparse 或 VSCode LOCAL 参数统一转换为 ``RoadVoidConfig``。

    本项目现在要求所有入口都走同一条参数路径：

    ``LOCAL_OUTPUT/LOCAL_WORKFLOW / argparse`` -> ``Namespace`` -> ``RoadVoidConfig`` ->
    ``geometry / forward / wavefield / scan / workflow``。

    这样改道路宽度、异常体位置或速度模式后，几何图、正演、扫描和波场
    示意不会各自偷偷回到不同默认值。
    """

    road_length = getattr(args, "road_length", 80.0)
    road_width = getattr(args, "road_width", 15.0)
    duration = getattr(args, "duration", 1.0)
    cfg = RoadVoidConfig(
        geometry=GeometryConfig(
            road_width=road_width,
            road_length=road_length,
            source_y=road_width,
            channel_x_min=0.0,
            channel_x_max=road_length,
            channel_spacing=getattr(args, "channel_spacing", 1.0),
            source_x_min=0.0,
            source_x_max=road_length,
            source_spacing=getattr(args, "source_spacing", 4.0),
            fiber_depth=getattr(args, "fiber_depth", 0.0),
            source_depth=getattr(args, "source_depth", 0.0),
        ),
        cavity=CavityConfig(
            enable_cavity=not getattr(args, "no_cavity", False),
            cavity_x=getattr(args, "cavity_x", 42.0),
            cavity_y=getattr(args, "cavity_y", 8.5),
            cavity_h=getattr(args, "cavity_depth", 2.2),
            cavity_radius=getattr(args, "cavity_radius", 2.0),
            scattering_strength=getattr(args, "scattering_strength", 1.0),
            attenuation_strength=getattr(args, "attenuation_strength", 0.25),
            tail_strength=getattr(args, "tail_strength", 1.0),
            shape=getattr(args, "cavity_shape", "sphere"),
            size_x=getattr(args, "cavity_size_x", None),
            size_y=getattr(args, "cavity_size_y", None),
            size_z=getattr(args, "cavity_size_z", None),
            azimuth=getattr(args, "cavity_azimuth", 0.0),
            anomalies=getattr(args, "anomalies", None),
        ),
        velocity=VelocityConfig(
            rayleigh_velocity=getattr(args, "rayleigh_velocity", 240.0),
            velocity_model_type=getattr(args, "velocity_mode", "uniform"),
            layer_depths=parse_float_list(getattr(args, "layer_depths", "0.4,1.5,4.0")),
            layer_velocities=parse_float_list(getattr(args, "layer_velocities", "180,240,320")),
            sensitivity_depth_factor=getattr(args, "sensitivity_depth_factor", 0.5),
            source_frequency=getattr(args, "source_frequency", 35.0),
            wavelet_type=getattr(args, "wavelet", "ricker"),
        ),
        record=RecordConfig(
            sampling_rate=getattr(args, "sampling_rate", 1000.0),
            duration=duration,
            t0=getattr(args, "t0", 0.02),
            random_seed=getattr(args, "random_seed", 2027),
        ),
        noise=NoiseConfig(
            noise_level=getattr(args, "noise_level", 0.03),
            traffic_noise_level=getattr(args, "traffic_noise_level", 0.015),
            bad_channel_fraction=getattr(args, "bad_channel_fraction", 0.02),
            weak_coupling_fraction=getattr(args, "weak_coupling_fraction", 0.06),
            coupling_variation=getattr(args, "coupling_variation", 0.08),
            diffraction_strength=getattr(args, "scattering_strength", 1.0),
        ),
        processing=ProcessingConfig(
            scan_x_min=getattr(args, "scan_x_min", 32.0),
            scan_x_max=getattr(args, "scan_x_max", 52.0),
            scan_x_step=getattr(args, "scan_x_step", 1.0),
            scan_y_min=getattr(args, "scan_y_min", 3.0),
            scan_y_max=getattr(args, "scan_y_max", min(14.0, road_width - 1.0)),
            scan_y_step=getattr(args, "scan_y_step", 1.0),
            scan_h_min=getattr(args, "scan_h_min", 0.8),
            scan_h_max=getattr(args, "scan_h_max", 4.0),
            scan_h_step=getattr(args, "scan_h_step", 0.4),
            scan_vr_min=getattr(args, "scan_vr_min", 220.0),
            scan_vr_max=getattr(args, "scan_vr_max", 260.0),
            scan_vr_step=getattr(args, "scan_vr_step", 10.0),
            score_method=getattr(args, "score_method", "envelope"),
            top_k=getattr(args, "top_k", 8),
            uncertainty_threshold=getattr(args, "uncertainty_threshold", 0.92),
            direct_wave_mute_width=getattr(args, "direct_wave_mute_width", 0.04),
            scan_mode=getattr(args, "scan_mode", "joint"),
            shot_index=getattr(args, "shot_index", None),
            shot_weight_mode=getattr(args, "shot_weight_mode", "uniform"),
        ),
    )
    return cfg


def config_from_args(args: argparse.Namespace) -> RoadVoidConfig:
    """兼容旧测试/旧脚本的别名；新代码优先用 build_road_void_config_from_args。"""

    return build_road_void_config_from_args(args)


def validate_config_consistency(cfg: RoadVoidConfig) -> None:
    """检查同一套配置在几何、异常体、扫描和记录长度上的一致性。

    这里的严重错误仍由 ``road_void.config.validate_config`` 抛出；本函数主要
    给出研究原型中常见的 warning，例如异常体移到了扫描范围之外、记录
    时长不足等，帮助 VSCode 本地调参时马上发现“不一致”的根源。
    """

    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()
    g = cfg.geometry
    p = cfg.processing
    if geom.n_channels < 3:
        print("警告：DAS 通道数过少，绕射曲线识别和扫描结果不可靠。")
    if geom.n_shots < 2:
        print("警告：炮点数过少，多炮联合约束基本失效。")
    if abs(float(geom.shot_y) - (g.source_y if g.source_y is not None else g.road_width)) > 1e-9:
        print("警告：RoadGeometry 中的炮线 y 与配置 source_y/road_width 不一致。")
    if abs(float(geom.fiber_y) - g.fiber_y) > 1e-9:
        print("警告：RoadGeometry 中的光纤 y 与配置 fiber_y 不一致。")

    for idx, cav in enumerate(cavities, start=1):
        prefix = f"警告：异常体 {idx}({cav.shape})"
        if cav.x0 < -0.05 * g.road_length or cav.x0 > 1.05 * g.road_length:
            print(f"{prefix} x={cav.x0:.2f} m 明显超出道路长度 [0, {g.road_length:.2f}]，图件/扫描可能难以解释。")
        if cav.y0 < min(g.fiber_y, geom.shot_y) or cav.y0 > max(g.fiber_y, geom.shot_y):
            print(f"{prefix} y={cav.y0:.2f} m 不在光纤-炮线横向孔径 [{g.fiber_y:.2f}, {float(geom.shot_y):.2f}] 内。")
        if cav.h <= 0:
            print(f"{prefix} depth/h={cav.h:.2f} m 非正，浅层异常体深度设置可能不合理。")
        if cav.radius <= 0:
            print(f"{prefix} radius={cav.radius:.2f} m 非正。")
        if not (p.scan_x_min <= cav.x0 <= p.scan_x_max):
            print("警告：当前扫描 x 范围没有覆盖设置的异常体位置，扫描结果可能找不到目标。")
        if not (p.scan_y_min <= cav.y0 <= p.scan_y_max):
            print("警告：当前扫描 y 范围没有覆盖设置的异常体位置，横向定位可能偏离目标。")
        if not (p.scan_h_min <= cav.h <= p.scan_h_max):
            print("警告：当前扫描 h 范围没有覆盖设置的异常体深度，深度扫描可能找不到目标。")

    vr_eff = cfg.effective_rayleigh_velocity()
    max_sg = float(np.max(geom.source_receiver_distances())) if geom.n_shots and geom.n_channels else 0.0
    needed = cfg.record.t0 + max_sg / vr_eff + 0.08
    if cfg.record.duration < needed:
        print(
            f"警告：记录长度 duration={cfg.record.duration:.3f}s 可能不足；"
            f"按最大炮检距估计至少需要约 {needed:.3f}s，直达波或尾波可能被截断。"
        )


def print_key_parameters(cfg: RoadVoidConfig, command: str | None = None) -> None:
    """打印当前完整参数摘要，便于 VSCode 输出窗口核对配置是否同步。"""

    cavities = cfg.to_cavities()
    vr_eff = cfg.effective_rayleigh_velocity()
    print("-" * 64)
    if command:
        print(f"当前运行模式：{command}")
    primary = "none" if not cavities else f"{cavities[0].shape}@({cavities[0].x0:.1f},{cavities[0].y0:.1f},{cavities[0].h:.1f})"
    print(
        "关键参数："
        f"W={cfg.geometry.road_width:.1f} m, L={cfg.geometry.road_length:.1f} m, "
        f"velocity-mode={cfg.velocity.velocity_model_type}, VR_eff={vr_eff:.1f} m/s, "
        f"anomalies={len(cavities)}, primary={primary}, scan-mode={cfg.processing.scan_mode}, "
        f"noise={cfg.noise.noise_level:.3f}"
    )
    print("道路与观测几何：")
    print(f"  road_width = {cfg.geometry.road_width:.2f} m, road_length = {cfg.geometry.road_length:.2f} m")
    print(f"  channel_spacing = {cfg.geometry.channel_spacing:.2f} m, source_spacing = {cfg.geometry.source_spacing:.2f} m")
    print("异常体：")
    if not cavities:
        print("  无异常体（--no-cavity）。")
    else:
        for idx, cav in enumerate(cavities, start=1):
            print(
                f"  {idx}) shape={cav.shape}, x={cav.x0:.2f}, y={cav.y0:.2f}, depth={cav.h:.2f}, "
                f"radius={cav.radius:.2f}, size=({cav.size_x},{cav.size_y},{cav.size_z}), "
                f"azimuth={cav.azimuth:.1f}, strength={cav.scattering_strength:.2f}"
            )
    wavelength = cfg.velocity.rayleigh_velocity / cfg.velocity.source_frequency
    print("速度与频率：")
    print(f"  velocity_mode = {cfg.velocity.velocity_model_type}")
    print(f"  VR = {cfg.velocity.rayleigh_velocity:.2f} m/s, VR_eff = {vr_eff:.2f} m/s")
    print(f"  source_frequency = {cfg.velocity.source_frequency:.2f} Hz, lambda=VR/f = {wavelength:.2f} m")
    if cfg.velocity.velocity_model_type == "layered-effective":
        print(f"  layer_depths = {cfg.velocity.layer_depths}, layer_velocities = {cfg.velocity.layer_velocities}")
    print("扫描范围：")
    print(f"  x=[{cfg.processing.scan_x_min:.2f}, {cfg.processing.scan_x_max:.2f}], step={cfg.processing.scan_x_step:.2f}")
    print(f"  y=[{cfg.processing.scan_y_min:.2f}, {cfg.processing.scan_y_max:.2f}], step={cfg.processing.scan_y_step:.2f}")
    print(f"  h=[{cfg.processing.scan_h_min:.2f}, {cfg.processing.scan_h_max:.2f}], step={cfg.processing.scan_h_step:.2f}")
    print(f"  scan_mode={cfg.processing.scan_mode}, shot_weight_mode={cfg.processing.shot_weight_mode}, noise={cfg.noise.noise_level:.3f}")
    print("-" * 64)


def prepare_road_void_config(args: argparse.Namespace, command: str | None = None) -> RoadVoidConfig:
    """构建、检查并回显配置；所有主流程入口都应调用这个函数。"""

    cfg = build_road_void_config_from_args(args)
    print_key_parameters(cfg, command or getattr(args, "command", None))
    validate_config_consistency(cfg)
    return cfg


def velocity_plot_info(cfg: RoadVoidConfig) -> dict[str, object]:
    """整理速度图标题/图注需要的参数，确保图件和正演扫描使用同一配置。"""

    vr = cfg.velocity.rayleigh_velocity
    f0 = cfg.velocity.source_frequency
    return {
        "velocity_mode": cfg.velocity.velocity_model_type,
        "rayleigh_velocity": vr,
        "effective_velocity": cfg.effective_rayleigh_velocity(),
        "source_frequency": f0,
        "wavelength": vr / f0,
        "sensitivity_depth_factor": cfg.velocity.sensitivity_depth_factor,
        "layer_depths": cfg.velocity.layer_depths,
        "layer_velocities": cfg.velocity.layer_velocities,
    }


def save_run_parameters(cfg: RoadVoidConfig, outdir: Path, enabled: bool) -> None:
    if not enabled:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    data = {
        "geometry": cfg.geometry.__dict__,
        "cavity": cfg.cavity.__dict__,
        "velocity": {**cfg.velocity.__dict__, "velocity_mode": cfg.velocity.velocity_model_type, "effective_rayleigh_velocity": cfg.effective_rayleigh_velocity()},
        "record": cfg.record.__dict__,
        "noise": cfg.noise.__dict__,
        "processing": cfg.processing.__dict__,
    }
    with (outdir / "run_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_geometry(args: argparse.Namespace) -> None:
    cfg = prepare_road_void_config(args, "geometry")
    outdir, opts, manifest = prepare_output_dir(args, "geometry")
    plot_road_geometry_3d(cfg.to_geometry(), cfg.to_cavities(), manifest.add(outdir / "geometry_3d.png"), **opts)
    plot_geometry_plan_and_sections(cfg.to_geometry(), cfg.to_cavities(), manifest.add(outdir / "geometry_plan_sections.png"), **opts)
    if bool(args.save_extra):
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    manifest.write_and_print()


def run_forward(args: argparse.Namespace) -> None:
    cfg = prepare_road_void_config(args, "forward")
    outdir, opts, manifest = prepare_output_dir(args, "forward")
    ds = simulate_from_config(cfg)
    geom = ds.geometry
    cavities = cfg.to_cavities()
    shot_index = geom.n_shots // 2 if not cavities else min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    plot_shot_gather(
        ds.data,
        geom,
        shot_index=shot_index,
        direct_times=ds.direct_times,
        diffraction_times=ds.diffraction_times[0] if ds.diffraction_times else None,
        title="三维等效瑞雷波正演记录",
        output=manifest.add(outdir / "forward_shot_gather.png"),
        **opts,
    )
    print(f"合成数据形状: {ds.data.shape}")
    if bool(args.save_extra):
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    manifest.write_and_print()


def run_velocity(args: argparse.Namespace) -> None:
    cfg = prepare_road_void_config(args, "velocity")
    outdir, opts, manifest = prepare_output_dir(args, "velocity")
    geom = cfg.to_geometry()
    plot_velocity_model(
        cfg.to_velocity_model(),
        (float(geom.channel_x[0]), float(geom.channel_x[-1])),
        cfg.to_cavities(),
        manifest.add(outdir / "velocity_model.png"),
        effective_velocity=cfg.effective_rayleigh_velocity(),
        velocity_info=velocity_plot_info(cfg),
        **opts,
    )
    print(f"说明：velocity-mode={cfg.velocity.velocity_model_type}，当前正演和扫描实际使用 VR_eff={cfg.effective_rayleigh_velocity():.1f} m/s。layered-effective 是轻量近似，不是完整频散反演。")
    if bool(args.save_extra):
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    manifest.write_and_print()


def run_wavefield(args: argparse.Namespace) -> None:
    cfg = prepare_road_void_config(args, "wavefield")
    outdir, opts, manifest = prepare_output_dir(args, "wavefield")
    print(f"wavefield 是 workflow 的第 6 步，本次结果写入 {outdir}。")
    cavities = cfg.to_cavities()
    if not cavities:
        print("当前关闭空洞，仅能展示直达波前；建议去掉 --no-cavity。")
        return
    geom = cfg.to_geometry()
    vr_eff = cfg.effective_rayleigh_velocity()
    velocity_info = velocity_plot_info(cfg)
    shot_indices = select_wavefield_shots(args, geom, cavities)
    if not shot_indices:
        print("没有可用炮点，无法生成 wavefield。")
        return
    shot_index = shot_indices[0]
    animate = bool(args.animate) and not bool(args.no_animate)
    print(f"wavefield 使用速度: velocity_mode={cfg.velocity.velocity_model_type}, VR={cfg.velocity.rayleigh_velocity:.1f} m/s, VR_eff={vr_eff:.1f} m/s")
    print(f"wavefield-view = {args.wavefield_view}；plan 为 x-y 地表平面示意，3d 为三维等时面几何示意。")
    if cfg.velocity.velocity_model_type == "layered-effective":
        print("说明：layered-effective 只折算 VR_eff；x-y 波场仍是等效运动学示意，不伪造成严格分层介质波场。")
    else:
        print("说明：uniform 使用单一 VR；本图仍是等效运动学传播示意，不是弹性波场快照。")
    plot_velocity_model(
        cfg.to_velocity_model(),
        (float(geom.channel_x[0]), float(geom.channel_x[-1])),
        cavities,
        manifest.add(outdir / "06_wavefield_velocity_context.png"),
        effective_velocity=vr_eff,
        velocity_info=velocity_info,
        **opts,
    )
    if args.wavefield_view == "3d":
        if getattr(args, "wavefield_mode", "single-shot") == "multi-shot":
            print("提醒：3d + multi-shot 当前只输出所选第一炮的三维关键帧/GIF；多炮覆盖建议使用 plan 视图。")
        if animate:
            frame_outputs = plot_kinematic_wavefield_frames_3d(
                geom,
                cavities,
                shot_index,
                vr_eff,
                outdir,
                t0=cfg.record.t0,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
                velocity_info=velocity_info,
                filename_prefix="06_wavefield_3d",
            )
            for output in frame_outputs:
                manifest.add(output)
            animate_kinematic_wavefield_3d(
                geom,
                cavities,
                shot_index,
                vr_eff,
                manifest.add(outdir / "06_wavefield_3d.gif"),
                t0=cfg.record.t0,
                n_frames=args.frames,
                fps=args.fps,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                velocity_info=velocity_info,
            )
            print("已生成 06_wavefield_3d_frame_*.png 和 06_wavefield_3d.gif：三维运动学等时面示意，不是完整弹性波场。")
        else:
            frame_outputs = plot_kinematic_wavefield_frames_3d(
                geom,
                cavities,
                shot_index,
                vr_eff,
                outdir,
                t0=cfg.record.t0,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
                velocity_info=velocity_info,
                filename_prefix="06_wavefield_3d",
            )
            for output in frame_outputs:
                manifest.add(output)
            print("未启用 --animate：已输出/显示三张 3D 运动学等时面关键帧。")
    elif getattr(args, "wavefield_mode", "single-shot") == "multi-shot":
        print(f"multi-shot wavefield 选择炮号: {shot_indices}")
        if animate:
            animate_multishot_kinematic_wavefield(
                geom,
                cavities,
                shot_indices,
                vr_eff,
                manifest.add(outdir / "06_multishot_wavefield.gif"),
                t0=cfg.record.t0,
                n_frames=args.frames,
                fps=args.fps,
                shot_interval=args.wavefield_shot_interval,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                velocity_info=velocity_info,
            )
            print("已生成 06_multishot_wavefield.gif：多炮顺序激发示意，不是多炮联合反演。")
        else:
            outputs = plot_multishot_wavefield_frames(
                geom,
                cavities,
                shot_indices,
                vr_eff,
                outdir,
                t0=cfg.record.t0,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
                velocity_info=velocity_info,
                filename_prefix="06_multishot_frame",
            )
            for output in outputs:
                manifest.add(output)
            print("未启用 --animate：multi-shot 只输出每个选中炮的一张代表性关键帧，避免大量单帧。")
    elif animate:
        animate_kinematic_wavefield(
            geom,
            cavities,
            shot_index,
            vr_eff,
            manifest.add(outdir / "06_kinematic_wavefield.gif"),
            t0=cfg.record.t0,
            n_frames=args.frames,
            fps=args.fps,
            save=bool(opts["save"]),
            show=bool(opts["show"]),
            velocity_info=velocity_info,
        )
        print("已生成 06_kinematic_wavefield.gif：时间从震源激发连续推进，异常体散射只在 S→D 到时之后出现。")
    else:
        frame_save = bool(opts["save"])
        frame_paths = [
            outdir / "06_wavefield_frame_early.png",
            outdir / "06_wavefield_frame_hit_cavity.png",
            outdir / "06_wavefield_frame_scattered.png",
        ]
        plot_kinematic_wavefield_frames(
            geom,
            cavities,
            shot_index,
            vr_eff,
            outdir,
            t0=cfg.record.t0,
            save=frame_save,
            show=bool(opts["show"]),
            dpi=int(opts["dpi"]),
            velocity_info=velocity_info,
            filename_prefix="06_wavefield",
        )
        for frame_path in frame_paths:
            manifest.add(frame_path, enabled=frame_save)
        print("未启用 --animate：已输出/显示三个等效运动学关键帧。")
        print("early: 直达波刚离开震源，散射波不应出现。")
        print("hit_cavity: 直达波前接近/到达第一个异常体。")
        print("scattered: 异常体散射波已开始从异常体位置向外传播。")
    if bool(args.save_extra):
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    print("说明：该波场是等效运动学传播示意，不是严格弹性波场快照。")
    manifest.write_and_print()


def run_path(args: argparse.Namespace) -> None:
    cfg = prepare_road_void_config(args, "path")
    outdir, opts, manifest = prepare_output_dir(args, "path")
    ds = simulate_from_config(cfg)
    geom = ds.geometry
    cavities = cfg.to_cavities()
    if not cavities:
        print("当前关闭空洞，无法绘制 S-D-G 绕射路径。")
        return
    cavity = cavities[0]
    shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavity.x0))
    channel_index = min(range(geom.n_channels), key=lambda i: abs(geom.channel_x[i] - cavity.x0))
    plot_diffraction_path_demo(geom, cavity, shot_index, channel_index, manifest.add(outdir / "diffraction_path_formula.png"), **opts)
    plot_shot_gather(
        ds.data,
        geom,
        shot_index,
        direct_times=ds.direct_times,
        diffraction_times=ds.diffraction_times[0],
        title="直达波与绕射波理论曲线叠加",
        output=manifest.add(outdir / "path_gather_curves.png"),
        **opts,
    )
    if bool(args.save_extra):
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    manifest.write_and_print()


def run_scan(args: argparse.Namespace) -> None:
    cfg = prepare_road_void_config(args, "scan")
    outdir, opts, manifest = prepare_output_dir(args, "scan")
    wf = run_location_workflow(cfg)
    geom = wf.dataset.geometry
    cavities = cfg.to_cavities()
    shot_index = geom.n_shots // 2 if not cavities else min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    best = wf.scan_result.best
    best_times = geom.diffraction_times((best.x0, best.y0, best.h), best.velocity, wf.velocity_fit.t0)
    plot_shot_gather(wf.residual, geom, shot_index, diffraction_times=best_times, title="残差记录与最佳三维绕射曲线", output=manifest.add(outdir / "scan_residual_best_curve.png"), **opts)
    true_x = cavities[0].x0 if cavities else None
    true_y = cavities[0].y0 if cavities else None
    true_h = cavities[0].h if cavities else None
    plot_score_slices(wf.scan_result, true_x=true_x, true_y=true_y, true_h=true_h, output=manifest.add(outdir / "scan_score_slices.png"), **opts)
    if bool(args.save_extra):
        plot_multishot_scan_diagnostics(wf.scan_result, outdir, **opts)
        for name in ("per_shot_best_x.png", "per_shot_score_contribution.png", "single_shot_vs_joint.png"):
            manifest.add(outdir / name)
    fit = wf.velocity_fit
    print(f"直达波拟合: VR={fit.velocity:.1f} m/s, t0={fit.t0:.4f} s, RMS={fit.residual_rms:.4f} s")
    print(f"最佳疑似异常体: x={best.x0:.1f} m, y={best.y0:.1f} m, h={best.h:.1f} m, VR={best.velocity:.1f} m/s, score={best.score:.4f}")
    print(f"不确定性范围: {wf.scan_result.uncertainty}")
    if wf.scan_result.consistency:
        c = wf.scan_result.consistency
        mode_label = "单炮扫描结果" if cfg.processing.scan_mode == "single-shot" else "多炮联合扫描结果"
        print(f"{mode_label}：")
        print(f"best x/y/h = {best.x0:.1f} / {best.y0:.1f} / {best.h:.1f}")
        print(f"单炮结果离散程度：x_std={c['x_std']:.2f}, y_std={c['y_std']:.2f}, h_std={c['h_std']:.2f}")
        print("解释：多炮联合通常让 x 更稳定，但单侧 DAS 下 y-h 耦合仍然存在。")
    if bool(args.save_extra):
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    manifest.write_and_print()


def run_sensitivity(args: argparse.Namespace) -> None:
    from examples.example_parameter_sensitivity import main as sensitivity_main

    cfg = prepare_road_void_config(args, "sensitivity")
    outdir = command_outdir(args, "sensitivity")
    sensitivity_main(outdir, cfg, save=not args.no_save)


def run_tutorial(args: argparse.Namespace) -> None:
    """生成一套不重复的教学流程图：几何、正演、扫描评分和可选动画。"""

    cfg = prepare_road_void_config(args, "tutorial")
    outdir, opts, manifest = prepare_output_dir(args, "tutorial")
    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()
    plot_geometry_plan_and_sections(geom, cavities, manifest.add(outdir / "01_geometry_plan_sections.png"), **opts)
    ds = simulate_from_config(cfg)
    shot_index = geom.n_shots // 2 if not cavities else min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    plot_shot_gather(ds.data, geom, shot_index, direct_times=ds.direct_times, diffraction_times=ds.diffraction_times[0] if ds.diffraction_times else None, title="02 正演 shot gather", output=manifest.add(outdir / "02_forward_gather.png"), **opts)
    wf = run_location_workflow(cfg)
    best = wf.scan_result.best
    best_times = geom.diffraction_times((best.x0, best.y0, best.h), best.velocity, wf.velocity_fit.t0)
    plot_shot_gather(wf.residual, geom, shot_index, diffraction_times=best_times, title="03 残差与最佳绕射曲线", output=manifest.add(outdir / "03_scan_residual_best_curve.png"), **opts)
    plot_score_slices(wf.scan_result, true_x=cavities[0].x0 if cavities else None, true_y=cavities[0].y0 if cavities else None, true_h=cavities[0].h if cavities else None, output=manifest.add(outdir / "04_scan_score_slices.png"), **opts)
    if cavities and bool(args.animate) and not bool(args.no_animate):
        animate_kinematic_wavefield(geom, cavities, shot_index, cfg.effective_rayleigh_velocity(), manifest.add(outdir / "05_kinematic_wavefield.gif"), save=bool(opts["save"]), show=False, velocity_info=velocity_plot_info(cfg))
    if bool(args.save_extra):
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    print(f"教学流程完成。最佳疑似异常体: x={best.x0:.1f}, y={best.y0:.1f}, h={best.h:.1f}")
    manifest.write_and_print()


def run_workflow(args: argparse.Namespace) -> None:
    """按算法逻辑顺序执行完整流程，且只正演一次、扫描一次。

    workflow 与 tutorial 的区别：workflow 是默认主入口，控制台会按步骤解释
    算法路线；tutorial 更像少量图件教学输出。这里统一上下文，避免重复调用
    run_forward/run_scan 导致重复正演、重复扫描和重复图件。
    """

    cfg = prepare_road_void_config(args, "workflow")
    outdir, opts, manifest = prepare_output_dir(args, "workflow")
    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()
    scenario = build_default_subsurface_scenario(cfg)
    survey_dataset = None
    diffraction_result = None
    localization_eval = None

    print("完整 workflow 开始。步骤顺序：")
    for idx, (_, desc) in enumerate(WORKFLOW_STEPS, start=1):
        print(f"  Step {idx}: {desc}")

    print("\nStep 1：构建三维道路场景。")
    plot_geometry_plan_and_sections(
        geom,
        cavities,
        manifest.add(outdir / "01_geometry_plan_sections.png"),
        **opts,
    )
    plot_road_geometry_3d(
        geom,
        cavities,
        manifest.add(outdir / "01_geometry_3d.png"),
        **opts,
    )
    print("坐标说明：x 沿道路/光纤方向，y 横穿道路方向，z 为深度；光纤位于 y=0，锤击线位于 y=W。")

    print("\nStep 2：展示速度/频率参数。")
    vr_eff = cfg.effective_rayleigh_velocity()
    wavelength = cfg.velocity.rayleigh_velocity / cfg.velocity.source_frequency
    print(f"速度模式 velocity-mode = {cfg.velocity.velocity_model_type}")
    print(f"输入参考瑞雷波速度 VR = {cfg.velocity.rayleigh_velocity:.1f} m/s")
    print(f"当前实际用于正演/扫描的 VR_eff = {vr_eff:.1f} m/s")
    print(f"锤击主频 f = {cfg.velocity.source_frequency:.1f} Hz")
    print(f"等效波长 lambda = VR / f = {wavelength:.2f} m")
    print("说明：layered-effective 会用指数敏感深度权重折算层状速度，但仍不是完整频散反演或弹性波场。")
    plot_velocity_model(
        cfg.to_velocity_model(),
        (float(geom.channel_x[0]), float(geom.channel_x[-1])),
        cavities,
        manifest.add(outdir / "02_velocity_model.png"),
        effective_velocity=vr_eff,
        velocity_info=velocity_plot_info(cfg),
        **opts,
    )
    if bool(args.save_extra):
        plot_subsurface_sections(
            scenario,
            manifest.add(outdir / "02b_subsurface_model_xz.png"),
            manifest.add(outdir / "02c_subsurface_model_yz.png"),
            **opts,
        )
        plot_subsurface_3d(
            scenario,
            manifest.add(outdir / "02d_subsurface_model_3d.png"),
            **opts,
        )

    print("\nStep 3：正演模拟。")
    print(f"当前异常体数量: {len(cavities)}；多异常体散射在正演中按等效散射点叠加。")
    workflow = run_location_workflow(cfg)
    dataset = workflow.dataset
    shot_index = geom.n_shots // 2 if not cavities else min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    print(f"合成数据形状: {dataset.data.shape} = shots x times x channels")
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index,
        direct_times=dataset.direct_times,
        diffraction_times=dataset.diffraction_times[0] if dataset.diffraction_times else None,
        title="03 正演 shot gather：直达波与绕射/散射事件",
        output=manifest.add(outdir / "03_forward_gather.png"),
        **opts,
    )
    if bool(args.save_extra):
        survey_dataset = generate_synthetic_survey_dataset(cfg, workflow)
        if bool(opts["save"]):
            npz_path, metadata_path = save_synthetic_survey_dataset(survey_dataset, outdir)
            manifest.add(npz_path)
            manifest.add(metadata_path)
        plot_das_like_gather(
            survey_dataset,
            shot_index,
            manifest.add(outdir / "03b_das_like_gather.png"),
            **opts,
        )
        plot_noise_components(
            survey_dataset,
            manifest.add(outdir / "03c_noise_components.png"),
            **opts,
        )
        print("save-extra：已生成结构化 synthetic survey dataset 与 DAS-like response 摘要。")

    print("\nStep 4：传播路径说明。")
    print("t_direct = t0 + |S-G| / VR")
    print("t_diff   = t0 + (|S-D| + |D-G|) / VR")
    if cavities:
        cavity = cavities[0]
        channel_index = min(range(geom.n_channels), key=lambda i: abs(geom.channel_x[i] - cavity.x0))
        plot_diffraction_path_demo(
            geom,
            cavity,
            shot_index,
            channel_index,
            manifest.add(outdir / "04_diffraction_path.png"),
            **opts,
        )
        plot_shot_gather(
            dataset.data,
            geom,
            shot_index,
            direct_times=dataset.direct_times,
            diffraction_times=dataset.diffraction_times[0] if dataset.diffraction_times else None,
            title="04 直达波与绕射波理论曲线叠加",
            output=manifest.add(outdir / "04_gather_with_curves.png"),
            **opts,
        )
    else:
        print("当前关闭空洞，跳过 S-D-G 绕射路径图。")
    if bool(args.save_extra):
        diffraction_result = detect_diffraction_features(cfg, workflow)
        plot_diffraction_attribute(
            diffraction_result,
            geom,
            manifest.add(outdir / "04b_diffraction_attribute.png"),
            **opts,
        )
        plot_diffraction_candidates(
            diffraction_result,
            manifest.add(outdir / "04c_diffraction_candidates.png"),
            **opts,
        )
        print("save-extra：已输出绕射/散射 envelope 属性和候选体评分。")

    print("\nStep 5：定位扫描。")
    print(f"扫描模式 scan-mode = {cfg.processing.scan_mode}；炮权重 shot-weight-mode = {cfg.processing.shot_weight_mode}")
    fit = workflow.velocity_fit
    scan = workflow.scan_result
    best = scan.best
    best_times = geom.diffraction_times((best.x0, best.y0, best.h), best.velocity, fit.t0)
    plot_shot_gather(
        workflow.residual,
        geom,
        shot_index,
        diffraction_times=best_times,
        title="05 残差记录与最佳三维绕射曲线",
        output=manifest.add(outdir / "05_residual_best_curve.png"),
        **opts,
    )
    plot_score_slices(
        scan,
        true_x=cavities[0].x0 if cavities else None,
        true_y=cavities[0].y0 if cavities else None,
        true_h=cavities[0].h if cavities else None,
        output=manifest.add(outdir / "05_scan_score_slices.png"),
        **opts,
    )
    if bool(args.save_extra):
        plot_multishot_scan_diagnostics(scan, outdir, **opts)
        for name in ("per_shot_best_x.png", "per_shot_score_contribution.png", "single_shot_vs_joint.png"):
            manifest.add(outdir / name)
        localization_eval = run_joint_localization_evaluation(cfg, workflow)
        plot_localization_error_summary(
            localization_eval,
            manifest.add(outdir / "05b_localization_error_summary.png"),
            **opts,
        )
        plot_uncertainty_summary(
            localization_eval,
            manifest.add(outdir / "05c_uncertainty_slices.png"),
            **opts,
        )
    print(f"直达波拟合 VR = {fit.velocity:.1f} m/s")
    print(f"拟合 t0 = {fit.t0:.4f} s, RMS = {fit.residual_rms:.4f} s")
    print(f"最佳疑似异常体 x/y/h/VR = {best.x0:.1f} / {best.y0:.1f} / {best.h:.1f} / {best.velocity:.1f}")
    if cavities:
        c = cavities[0]
        print(f"真实空洞位置 x/y/h = {c.x0:.1f} / {c.y0:.1f} / {c.h:.1f}")
    print(f"不确定性范围 = {scan.uncertainty}")
    if scan.consistency:
        cns = scan.consistency
        print(f"单炮结果离散程度：x_std={cns['x_std']:.2f}, y_std={cns['y_std']:.2f}, h_std={cns['h_std']:.2f}")
    print("解释提醒：单侧 DAS + 对侧锤击下，x 通常更稳定，y-h 存在耦合，应输出范围而非唯一确诊点。")

    print("\nStep 6：可选运动学波场动画。")
    animate = bool(args.animate) and not bool(args.no_animate)
    if animate and cavities:
        if args.wavefield_view == "3d":
            frame_outputs = plot_kinematic_wavefield_frames_3d(
                geom,
                cavities,
                shot_index,
                vr_eff,
                outdir,
                t0=cfg.record.t0,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
                velocity_info=velocity_plot_info(cfg),
                filename_prefix="06_wavefield_3d",
            )
            for output in frame_outputs:
                manifest.add(output)
            animate_kinematic_wavefield_3d(
                geom,
                cavities,
                shot_index,
                vr_eff,
                manifest.add(outdir / "06_wavefield_3d.gif"),
                t0=cfg.record.t0,
                n_frames=args.frames,
                fps=args.fps,
                save=bool(opts["save"]),
                show=False,
                velocity_info=velocity_plot_info(cfg),
            )
            print("已生成 06_wavefield_3d_frame_*.png 和 06_wavefield_3d.gif。该图是三维运动学等时面示意，不是完整弹性波场。")
        else:
            plot_kinematic_wavefield_frames(
                geom,
                cavities,
                shot_index,
                vr_eff,
                outdir,
                t0=cfg.record.t0,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
                velocity_info=velocity_plot_info(cfg),
                filename_prefix="06_wavefield",
            )
            for frame_name in (
                "06_wavefield_frame_early.png",
                "06_wavefield_frame_hit_cavity.png",
                "06_wavefield_frame_scattered.png",
            ):
                manifest.add(outdir / frame_name)
            if getattr(args, "wavefield_mode", "single-shot") == "multi-shot":
                shot_indices = select_wavefield_shots(args, geom, cavities)
                animate_multishot_kinematic_wavefield(
                    geom,
                    cavities,
                    shot_indices,
                    vr_eff,
                    manifest.add(outdir / "06_multishot_wavefield.gif"),
                    t0=cfg.record.t0,
                    n_frames=args.frames,
                    fps=args.fps,
                    shot_interval=args.wavefield_shot_interval,
                    save=bool(opts["save"]),
                    show=False,
                    velocity_info=velocity_plot_info(cfg),
                )
                print("已生成 06_wavefield_frame_*.png 和 06_multishot_wavefield.gif。该动画是多炮顺序激发示意，不是多炮联合反演。")
            else:
                animate_kinematic_wavefield(
                    geom,
                    cavities,
                    shot_index,
                    vr_eff,
                    manifest.add(outdir / "06_kinematic_wavefield.gif"),
                    t0=cfg.record.t0,
                    n_frames=args.frames,
                    fps=args.fps,
                    save=bool(opts["save"]),
                    show=False,
                    velocity_info=velocity_plot_info(cfg),
                )
                print("已生成 06_wavefield_frame_*.png 和 06_kinematic_wavefield.gif。该动画是 x-y 平面运动学示意，不是严格弹性波场快照。")
    else:
        print("未生成动画。如需生成等效波场动图，请运行：")
        print("python main.py workflow --animate --save")
        print("或：")
        print("python main.py wavefield --animate --save")
        print("输出写入 outputs/workflow/，该动画是等效运动学传播示意，不是严格弹性波场快照。")

    print("\nStep 7：结果总结。")
    if bool(args.save_extra):
        if survey_dataset is None:
            survey_dataset = generate_synthetic_survey_dataset(cfg, workflow)
        if diffraction_result is None:
            diffraction_result = detect_diffraction_features(cfg, workflow)
        if localization_eval is None:
            localization_eval = run_joint_localization_evaluation(cfg, workflow)
        if bool(opts["save"]):
            report_path = write_research_report(
                cfg,
                scenario,
                survey_dataset,
                diffraction_result,
                localization_eval,
                outdir / "research_report.md",
            )
            manifest.add(report_path)
        save_run_parameters(cfg, outdir, bool(opts["save"]))
        manifest.add(outdir / "run_parameters.json")
    print("本次流程完成。")
    print(f"速度总结：velocity-mode={cfg.velocity.velocity_model_type}, VR_eff={vr_eff:.1f} m/s。")
    print(f"扫描总结：scan-mode={cfg.processing.scan_mode}，当前扫描默认定位主异常；多异常联合反演可后续用迭代减去方式扩展。")
    print(f"异常体总结：本次异常体数量={len(cavities)}，shape 仅表示等效散射几何，不是真实弹性边界散射。")
    print("本方法输出的是疑似异常范围，不是直接确诊空洞。")
    print("真实数据应用前仍需光纤路径标定、锤击触发校正、通道耦合 QC、浅层速度估计和管线/井盖干扰核查。")
    manifest.write_and_print()


def run_elastic3d_command(args: argparse.Namespace) -> None:
    """运行小尺度 3D elastic FDTD 原型，不替代默认运动学 workflow。"""

    road_cfg = prepare_road_void_config(args, "elastic3d")
    outdir, opts, manifest = prepare_output_dir(args, "elastic3d")
    cavities = None
    if not args.elastic_no_anomaly:
        cavities = road_cfg.to_cavities()
    cfg = Elastic3DConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        dt=args.elastic_dt,
        nt=args.elastic_nt,
        source_frequency=args.elastic_source_frequency,
        source_amplitude=args.elastic_source_amplitude,
        space_order=args.elastic_space_order,
        abc=args.elastic_abc,
        record_component=args.elastic_record_component,
        gauge_length=args.elastic_gauge_length,
        with_anomaly=not args.elastic_no_anomaly,
    )
    print("elastic3d：三维弹性波全波形有限差分原型，小尺度教学/研究模型，不是工业级模拟。")
    print(f"dx/dy/dz = {cfg.dx:.3f}/{cfg.dy:.3f}/{cfg.dz:.3f} m, dt = {cfg.dt:.6f} s, nt = {cfg.nt}")
    print(f"space_order = {cfg.space_order}；abc = {cfg.abc}；record_component = {cfg.record_component}；gauge_length = {cfg.gauge_length:.2f} m")
    if cfg.abc == "cpml":
        print("提醒：当前 cpml 是 experimental CPML-like 阻尼，尚不是完整带记忆变量的严格 CPML。")
    if cavities:
        print(f"elastic3d 使用同一 RoadVoidConfig 中的异常体，共 {len(cavities)} 个；请注意坐标需落在小模型范围内。")
    result = run_elastic3d(cfg, cavities)
    print(f"vmax = {float(result.model.vp.max()):.1f} m/s")
    print(f"CFL number = {result.cfl:.3f}，满足稳定条件。")
    print(f"gather shape = {result.gather.shape} = time x receiver")
    plot_elastic3d_outputs(result, outdir, save=bool(opts["save"]), show=bool(opts["show"]), dpi=int(opts["dpi"]))
    for name in (
        "velocity_model_slice.png",
        "wavefield_snapshot_early.png",
        "wavefield_snapshot_mid.png",
        "wavefield_snapshot_late.png",
        f"elastic3d_gather_{cfg.record_component}.png",
    ):
        manifest.add(outdir / name)
    if cfg.record_component == "vz":
        manifest.add(outdir / "elastic3d_gather.png")
    if bool(opts["save"]) and cfg.abc == "cpml" and bool(args.save_extra):
        plot_abc_comparison(cfg, outdir, save=True, show=False, dpi=int(opts["dpi"]))
        manifest.add(outdir / "abc_compare_sponge_vs_cpml.png")
        print("已输出 abc_compare_sponge_vs_cpml.png，用于边界吸收 sanity check。")
    if bool(args.animate) and not bool(args.no_animate):
        animate_elastic3d_wavefield(result, outdir, save=bool(opts["save"]), show=bool(opts["show"]), fps=args.fps)
        manifest.add(outdir / "elastic3d_wavefield.gif")
        print("已生成 elastic3d_wavefield.gif。")
    print("输出: velocity_model_slice.png, wavefield_snapshot_*.png, elastic3d_gather_<component>.png；--animate 时额外输出 elastic3d_wavefield.gif。")
    manifest.write_and_print()


def run_elastic_validate(args: argparse.Namespace) -> None:
    """从道路 workflow 配置裁剪局部区域，运行小尺度 elastic3d validation。"""

    road_cfg = prepare_road_void_config(args, "elastic-validate")
    outdir, opts, manifest = prepare_output_dir(args, "elastic3d_validation")
    elastic_cfg = Elastic3DConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        dt=args.elastic_dt,
        nt=min(args.elastic_nt, 260),
        source_frequency=args.elastic_source_frequency,
        source_amplitude=args.elastic_source_amplitude,
        space_order=args.elastic_space_order,
        abc=args.elastic_abc,
        record_component=args.elastic_record_component,
        gauge_length=args.elastic_gauge_length,
        with_anomaly=not args.elastic_no_anomaly,
    )
    case = build_local_elastic_validation_case(road_cfg, elastic_config=elastic_cfg)
    print("elastic-validate：从 RoadVoidConfig 生成局部 elastic3d validation case。")
    print(f"局部坐标原点 origin_x/origin_y = {case.origin_x:.2f} / {case.origin_y:.2f} m")
    print(f"异常体是否落入 elastic3d 小网格 = {case.in_bounds}")
    print(case.note)
    result = run_elastic_validation_case(case)
    print(f"local elastic gather shape = {result.gather.shape}, CFL={result.cfl:.3f}")
    outputs = plot_elastic_validation_outputs(
        case,
        result,
        outdir,
        save=bool(opts["save"]),
        show=bool(opts["show"]),
        dpi=int(opts["dpi"]),
    )
    for path in outputs:
        manifest.add(path)
    for name in (
        "velocity_model_slice.png",
        "wavefield_snapshot_early.png",
        "wavefield_snapshot_mid.png",
        "wavefield_snapshot_late.png",
        f"elastic3d_gather_{elastic_cfg.record_component}.png",
    ):
        manifest.add(outdir / name)
    print("输出: local_model_slices.png, elastic_gather.png, elastic_wavefield_snapshot.png。")
    print("说明：这是局部全波场 sanity check，不替代默认道路 workflow。")
    manifest.write_and_print()


def run_fwi_demo(args: argparse.Namespace) -> None:
    """运行 FWI 最小原型：一维 Vs 缩放 misfit 曲线。"""

    prepare_road_void_config(args, "fwi-demo")
    outdir = command_outdir(args, "fwi")
    opts = output_options(args)
    cfg = Elastic3DConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        dt=args.elastic_dt,
        nt=min(args.elastic_nt, 120),
        source_frequency=args.elastic_source_frequency,
        source_amplitude=args.elastic_source_amplitude,
        space_order=args.elastic_space_order,
        abc=args.elastic_abc,
        record_component=args.elastic_record_component,
        gauge_length=args.elastic_gauge_length,
        with_anomaly=not args.elastic_no_anomaly,
    )
    scales = parse_float_list(args.fwi_vs_scales)
    print("fwi-demo：只计算一维 Vs 缩放 L2 misfit 曲线，不包含伴随梯度或模型更新。")
    print(f"候选 Vs scales = {scales}")
    result = run_fwi_misfit_demo(
        vs_scales=scales,
        observed_vs_scale=args.fwi_observed_vs_scale,
        initial_vs_scale=args.fwi_initial_vs_scale,
        config=cfg,
    )
    print(f"最小 misfit 对应 Vs scale = {result.best_vs_scale:.3f}")
    plot_fwi_demo_outputs(result, outdir, save=bool(opts["save"]), show=bool(opts["show"]), dpi=int(opts["dpi"]))
    print("输出: misfit_curve.png, observed_vs_synthetic_gather.png。当前不是完整三维弹性伴随 FWI。")


def run_numerics_demo(args: argparse.Namespace) -> None:
    """运行 FEM/SEM/BEM 高级数值方法教学原型。"""

    outdir, opts, manifest = prepare_output_dir(args, "numerics")
    methods = ["fem", "sem", "bem"] if args.method == "all" else [args.method]
    print("numerics-demo：高级数值方法教学/研究原型，不替代默认 workflow。")
    print(f"method = {args.method}; 输出目录 = {outdir}")
    for method in methods:
        if method == "fem":
            result = run_fem1d_wave_demo(
                length=args.numerics_length,
                velocity=args.numerics_velocity,
                duration=args.numerics_duration,
                outdir=outdir,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
            )
            manifest.add(outdir / "fem1d_wavefield.png")
            manifest.add(outdir / "fem1d_receiver_trace.png")
            print(f"FEM 1D 完成：nodes={result.x.size}, snapshots={result.snapshots.shape}, receiver_samples={result.receiver_trace.size}")
        elif method == "sem":
            result = run_sem1d_wave_demo(
                length=args.numerics_length,
                velocity=args.numerics_velocity,
                duration=args.numerics_duration,
                outdir=outdir,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
            )
            manifest.add(outdir / "sem1d_wavefield.png")
            manifest.add(outdir / "sem1d_receiver_trace.png")
            print(f"SEM 1D 完成：nodes={result.x.size}, GLL/order_nodes={result.gll_nodes.size}, receiver_samples={result.receiver_trace.size}")
        elif method == "bem":
            result = run_bem2d_scatter_demo(
                outdir=outdir,
                save=bool(opts["save"]),
                show=bool(opts["show"]),
                dpi=int(opts["dpi"]),
            )
            manifest.add(outdir / "bem2d_boundary_points.png")
            manifest.add(outdir / "bem2d_scattered_response.png")
            print(f"BEM 2D 完成：boundary_points={result.boundary.shape[0]}, receivers={result.receivers.shape[0]}")
    print("说明：FEM/SEM/BEM 均为标量低维教学原型，不是完整三维弹性模拟。")
    manifest.write_and_print()


def run_numerics_compare(args: argparse.Namespace) -> None:
    """运行 FDTD/FEM/SEM 统一 1D 标量波 benchmark。"""

    outdir, opts, manifest = prepare_output_dir(args, "numerics")
    print("numerics-compare：统一 1D homogeneous scalar wave benchmark。")
    print(
        f"L={args.numerics_length:.1f} m, c={args.numerics_velocity:.1f} m/s, "
        f"dt={args.numerics_dt:.6f} s, duration={args.numerics_duration:.3f} s"
    )
    print(
        f"source={args.numerics_source_position:.1f} m, receiver={args.numerics_receiver_position:.1f} m, "
        f"f0={args.numerics_source_frequency:.1f} Hz"
    )
    result = compare_1d_wave_methods(
        length=args.numerics_length,
        velocity=args.numerics_velocity,
        duration=args.numerics_duration,
        dt=args.numerics_dt,
        source_position=args.numerics_source_position,
        receiver_position=args.numerics_receiver_position,
        source_frequency=args.numerics_source_frequency,
        outdir=outdir,
        save=bool(opts["save"]),
        show=bool(opts["show"]),
        dpi=int(opts["dpi"]),
    )
    print("到时与差异指标：")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6g}" if isinstance(value, float) else f"  {key}: {value}")
    print("输出: compare_1d_traces.png, compare_1d_wavefields.png, compare_1d_metrics.json。")
    print("说明：该 benchmark 是低维标量波教学对比，不代表三维弹性道路空洞正演。")
    for name in ("compare_1d_traces.png", "compare_1d_wavefields.png", "compare_1d_metrics.json"):
        manifest.add(outdir / name)
    manifest.write_and_print()


def run_all(args: argparse.Namespace) -> None:
    """all 作为 workflow 的别名，避免维护两套完整流程。"""

    run_workflow(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="城市道路空洞 DAS 三维正演与定位研究原型")
    sub = parser.add_subparsers(dest="command", required=True)
    handlers = {
        "workflow": run_workflow,
        "geometry": run_geometry,
        "forward": run_forward,
        "velocity": run_velocity,
        "wavefield": run_wavefield,
        "path": run_path,
        "scan": run_scan,
        "sensitivity": run_sensitivity,
        "tutorial": run_tutorial,
        "elastic3d": run_elastic3d_command,
        "elastic-validate": run_elastic_validate,
        "fwi-demo": run_fwi_demo,
        "numerics-demo": run_numerics_demo,
        "numerics-compare": run_numerics_compare,
        "all": run_all,
    }
    for name, handler in handlers.items():
        p = sub.add_parser(name, help=f"运行 {name} 功能")
        add_geometry_args(p)
        add_wave_args(p)
        if name in {"scan", "sensitivity", "tutorial", "workflow", "all"}:
            add_scan_args(p)
        if name in {"wavefield", "tutorial", "workflow", "all"}:
            add_animation_args(p)
            add_wavefield_args(p)
        if name in {"elastic3d", "elastic-validate"}:
            add_elastic3d_args(p)
            add_animation_args(p)
        if name == "fwi-demo":
            add_elastic3d_args(p)
            add_fwi_args(p)
        if name == "numerics-demo":
            add_numerics_args(p)
        if name == "numerics-compare":
            add_numerics_compare_args(p)
        add_output_args(p)
        p.set_defaults(func=handler)
    return parser


def main() -> None:
    argv = sys.argv[1:]
    if not argv and USE_LOCAL_DEBUG_CONFIG:
        args = build_args_from_local_config()
        print_local_run_summary(args)
    else:
        if not argv or argv[0].startswith("--"):
            argv = ["workflow", *argv]
        args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
