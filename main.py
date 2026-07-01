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

from road_void.config import CavityConfig, GeometryConfig, NoiseConfig, ProcessingConfig, RecordConfig, RoadVoidConfig, VelocityConfig
from road_void.visualization import (
    animate_kinematic_wavefield,
    plot_diffraction_path_demo,
    plot_geometry_plan_and_sections,
    plot_road_geometry_3d,
    plot_score_slices,
    plot_shot_gather,
    plot_velocity_model,
)
from road_void.workflow import run_location_workflow, simulate_from_config


WORKFLOW_STEPS = [
    ("geometry", "建立道路三维几何：道路、光纤、炮线、空洞"),
    ("velocity", "展示等效瑞雷波速度/速度模型，并说明 lambda = VR / f"),
    ("forward", "锤击激发三维等效瑞雷波正演，生成单侧 DAS shot gather"),
    ("path", "展示 S-G 直达路径与 S-D-G 绕射路径及走时公式"),
    ("scan", "直达波拟合、残差构建、绕射扫描和疑似空洞定位"),
    ("wavefield", "可选生成等效运动学波场动画，不作为高保真弹性波场"),
    ("summary", "输出结果解释、受限孔径提醒和参数记录"),
]


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """添加通用输出参数。"""

    parser.add_argument("--save", action="store_true", help="保存图件到 --outdir。适合批量运行和写汇报材料。")
    parser.add_argument("--no-save", action="store_true", help="不保存图件。用于快速测试，优先级高于 --save。")
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
    parser.add_argument("--no-cavity", action="store_true", help="关闭空洞散射，用于检查无异常情况下的误报风险。")


def add_wave_args(parser: argparse.ArgumentParser) -> None:
    """速度、频率、采样和噪声参数。"""

    parser.add_argument("--rayleigh-velocity", type=float, default=240.0, help="等效瑞雷波速度 VR，单位 m/s；t_direct=t0+|S-G|/VR，t_diff=t0+(|S-D|+|D-G|)/VR。VR 偏大可能使深度偏深，偏小可能使深度偏浅。")
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


def add_animation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--animate", action="store_true", help="生成 GIF 动画；适合汇报展示传播过程。")
    parser.add_argument("--no-animate", action="store_true", help="不生成 GIF，只在 --show 时显示关键帧。")
    parser.add_argument("--frames", type=int, default=48, help="GIF 帧数；越大动画更顺滑但文件更大。")
    parser.add_argument("--fps", type=int, default=10, help="GIF 帧率。")


def output_options(args: argparse.Namespace) -> dict[str, object]:
    save = bool(args.save) and not bool(args.no_save)
    show = bool(args.show) and not bool(args.no_show)
    return {"save": save, "show": show, "dpi": args.dpi}


def command_outdir(args: argparse.Namespace, name: str) -> Path:
    return Path(args.outdir) if args.outdir else Path("outputs") / name


def config_from_args(args: argparse.Namespace) -> RoadVoidConfig:
    """把 argparse 参数转换为内部配置对象。

    这里使用 dataclass 只是为了复用已有 workflow，不再要求用户修改 YAML。
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
        ),
        velocity=VelocityConfig(
            rayleigh_velocity=getattr(args, "rayleigh_velocity", 240.0),
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
        ),
    )
    return cfg


def print_key_parameters(cfg: RoadVoidConfig) -> None:
    print(
        "关键参数："
        f"W={cfg.geometry.road_width:.1f} m, L={cfg.geometry.road_length:.1f} m, "
        f"dx_rec={cfg.geometry.channel_spacing:.1f} m, dx_src={cfg.geometry.source_spacing:.1f} m, "
        f"VR={cfg.velocity.rayleigh_velocity:.1f} m/s, f={cfg.velocity.source_frequency:.1f} Hz, "
        f"cavity=({cfg.cavity.cavity_x:.1f}, {cfg.cavity.cavity_y:.1f}, {cfg.cavity.cavity_h:.1f}) m, "
        f"noise={cfg.noise.noise_level:.3f}"
    )


def save_run_parameters(cfg: RoadVoidConfig, outdir: Path, enabled: bool) -> None:
    if not enabled:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    data = {
        "geometry": cfg.geometry.__dict__,
        "cavity": cfg.cavity.__dict__,
        "velocity": cfg.velocity.__dict__,
        "record": cfg.record.__dict__,
        "noise": cfg.noise.__dict__,
        "processing": cfg.processing.__dict__,
    }
    with (outdir / "run_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_geometry(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    outdir = command_outdir(args, "geometry")
    opts = output_options(args)
    print_key_parameters(cfg)
    plot_road_geometry_3d(cfg.to_geometry(), cfg.to_cavities(), outdir / "geometry_3d.png", **opts)
    plot_geometry_plan_and_sections(cfg.to_geometry(), cfg.to_cavities(), outdir / "geometry_plan_sections.png", **opts)
    save_run_parameters(cfg, outdir, bool(opts["save"]))


def run_forward(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    outdir = command_outdir(args, "forward")
    opts = output_options(args)
    print_key_parameters(cfg)
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
        output=outdir / "forward_shot_gather.png",
        **opts,
    )
    print(f"合成数据形状: {ds.data.shape}")
    save_run_parameters(cfg, outdir, bool(opts["save"]))


def run_velocity(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    outdir = command_outdir(args, "velocity")
    opts = output_options(args)
    print_key_parameters(cfg)
    geom = cfg.to_geometry()
    plot_velocity_model(cfg.to_velocity_model(), (float(geom.channel_x[0]), float(geom.channel_x[-1])), cfg.to_cavities(), outdir / "velocity_model.png", **opts)
    print("说明：layered 速度模型目前主要用于教学展示；核心正演/扫描仍使用等效 VR。")
    save_run_parameters(cfg, outdir, bool(opts["save"]))


def run_wavefield(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    outdir = command_outdir(args, "wavefield")
    opts = output_options(args)
    cavities = cfg.to_cavities()
    if not cavities:
        print("当前关闭空洞，仅能展示直达波前；建议去掉 --no-cavity。")
        return
    geom = cfg.to_geometry()
    shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    animate = bool(args.animate) and not bool(args.no_animate)
    animate_kinematic_wavefield(
        geom,
        cavities[0],
        shot_index,
        cfg.velocity.rayleigh_velocity,
        outdir / "kinematic_wavefield.gif",
        t0=cfg.record.t0,
        n_frames=args.frames,
        fps=args.fps,
        save=bool(opts["save"]) and animate,
        show=bool(opts["show"]),
    )
    save_run_parameters(cfg, outdir, bool(opts["save"]))
    print("说明：该波场是等效运动学传播示意，不是严格弹性波场快照。")


def run_path(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    outdir = command_outdir(args, "path")
    opts = output_options(args)
    ds = simulate_from_config(cfg)
    geom = ds.geometry
    cavities = cfg.to_cavities()
    if not cavities:
        print("当前关闭空洞，无法绘制 S-D-G 绕射路径。")
        return
    cavity = cavities[0]
    shot_index = min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavity.x0))
    channel_index = min(range(geom.n_channels), key=lambda i: abs(geom.channel_x[i] - cavity.x0))
    plot_diffraction_path_demo(geom, cavity, shot_index, channel_index, outdir / "diffraction_path_formula.png", **opts)
    plot_shot_gather(
        ds.data,
        geom,
        shot_index,
        direct_times=ds.direct_times,
        diffraction_times=ds.diffraction_times[0],
        title="直达波与绕射波理论曲线叠加",
        output=outdir / "path_gather_curves.png",
        **opts,
    )
    save_run_parameters(cfg, outdir, bool(opts["save"]))


def run_scan(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    outdir = command_outdir(args, "scan")
    opts = output_options(args)
    print_key_parameters(cfg)
    wf = run_location_workflow(cfg)
    geom = wf.dataset.geometry
    cavities = cfg.to_cavities()
    shot_index = geom.n_shots // 2 if not cavities else min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    best = wf.scan_result.best
    best_times = geom.diffraction_times((best.x0, best.y0, best.h), best.velocity, wf.velocity_fit.t0)
    plot_shot_gather(wf.residual, geom, shot_index, diffraction_times=best_times, title="残差记录与最佳三维绕射曲线", output=outdir / "scan_residual_best_curve.png", **opts)
    true_x = cavities[0].x0 if cavities else None
    true_y = cavities[0].y0 if cavities else None
    true_h = cavities[0].h if cavities else None
    plot_score_slices(wf.scan_result, true_x=true_x, true_y=true_y, true_h=true_h, output=outdir / "scan_score_slices.png", **opts)
    fit = wf.velocity_fit
    print(f"直达波拟合: VR={fit.velocity:.1f} m/s, t0={fit.t0:.4f} s, RMS={fit.residual_rms:.4f} s")
    print(f"最佳疑似异常体: x={best.x0:.1f} m, y={best.y0:.1f} m, h={best.h:.1f} m, VR={best.velocity:.1f} m/s, score={best.score:.4f}")
    print(f"不确定性范围: {wf.scan_result.uncertainty}")
    save_run_parameters(cfg, outdir, bool(opts["save"]))


def run_sensitivity(args: argparse.Namespace) -> None:
    from examples.example_parameter_sensitivity import main as sensitivity_main

    cfg = config_from_args(args)
    outdir = command_outdir(args, "sensitivity")
    sensitivity_main(outdir, cfg, save=not args.no_save)


def run_tutorial(args: argparse.Namespace) -> None:
    """生成一套不重复的教学流程图：几何、正演、扫描评分和可选动画。"""

    cfg = config_from_args(args)
    outdir = command_outdir(args, "tutorial")
    opts = output_options(args)
    print_key_parameters(cfg)
    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()
    plot_geometry_plan_and_sections(geom, cavities, outdir / "01_geometry_plan_sections.png", **opts)
    ds = simulate_from_config(cfg)
    shot_index = geom.n_shots // 2 if not cavities else min(range(geom.n_shots), key=lambda i: abs(geom.shot_x[i] - cavities[0].x0))
    plot_shot_gather(ds.data, geom, shot_index, direct_times=ds.direct_times, diffraction_times=ds.diffraction_times[0] if ds.diffraction_times else None, title="02 正演 shot gather", output=outdir / "02_forward_gather.png", **opts)
    wf = run_location_workflow(cfg)
    best = wf.scan_result.best
    best_times = geom.diffraction_times((best.x0, best.y0, best.h), best.velocity, wf.velocity_fit.t0)
    plot_shot_gather(wf.residual, geom, shot_index, diffraction_times=best_times, title="03 残差与最佳绕射曲线", output=outdir / "03_scan_residual_best_curve.png", **opts)
    plot_score_slices(wf.scan_result, true_x=cavities[0].x0 if cavities else None, true_y=cavities[0].y0 if cavities else None, true_h=cavities[0].h if cavities else None, output=outdir / "04_scan_score_slices.png", **opts)
    if cavities and bool(args.animate) and not bool(args.no_animate):
        animate_kinematic_wavefield(geom, cavities[0], shot_index, cfg.velocity.rayleigh_velocity, outdir / "05_kinematic_wavefield.gif", save=bool(opts["save"]), show=False)
    save_run_parameters(cfg, outdir, bool(opts["save"]))
    print(f"教学流程完成。最佳疑似异常体: x={best.x0:.1f}, y={best.y0:.1f}, h={best.h:.1f}")


def run_workflow(args: argparse.Namespace) -> None:
    """按算法逻辑顺序执行完整流程，且只正演一次、扫描一次。

    workflow 与 tutorial 的区别：workflow 是默认主入口，控制台会按步骤解释
    算法路线；tutorial 更像少量图件教学输出。这里统一上下文，避免重复调用
    run_forward/run_scan 导致重复正演、重复扫描和重复图件。
    """

    cfg = config_from_args(args)
    outdir = command_outdir(args, "workflow")
    opts = output_options(args)
    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()

    print("完整 workflow 开始。步骤顺序：")
    for idx, (_, desc) in enumerate(WORKFLOW_STEPS, start=1):
        print(f"  Step {idx}: {desc}")
    print_key_parameters(cfg)

    print("\nStep 1：构建三维道路场景。")
    plot_geometry_plan_and_sections(
        geom,
        cavities,
        outdir / "01_geometry_plan_sections.png",
        **opts,
    )
    plot_road_geometry_3d(
        geom,
        cavities,
        outdir / "01_geometry_3d.png",
        **opts,
    )
    print("坐标说明：x 沿道路/光纤方向，y 横穿道路方向，z 为深度；光纤位于 y=0，锤击线位于 y=W。")

    print("\nStep 2：展示速度/频率参数。")
    wavelength = cfg.velocity.rayleigh_velocity / cfg.velocity.source_frequency
    print(f"等效瑞雷波速度 VR = {cfg.velocity.rayleigh_velocity:.1f} m/s")
    print(f"锤击主频 f = {cfg.velocity.source_frequency:.1f} Hz")
    print(f"等效波长 lambda = VR / f = {wavelength:.2f} m")
    print("说明：当前 VR 是某一频带内的等效瑞雷波速度，不是完整随深度变化的速度模型。")
    plot_velocity_model(
        cfg.to_velocity_model(),
        (float(geom.channel_x[0]), float(geom.channel_x[-1])),
        cavities,
        outdir / "02_velocity_model.png",
        **opts,
    )

    print("\nStep 3：正演模拟。")
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
        output=outdir / "03_forward_gather.png",
        **opts,
    )

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
            outdir / "04_diffraction_path.png",
            **opts,
        )
        plot_shot_gather(
            dataset.data,
            geom,
            shot_index,
            direct_times=dataset.direct_times,
            diffraction_times=dataset.diffraction_times[0] if dataset.diffraction_times else None,
            title="04 直达波与绕射波理论曲线叠加",
            output=outdir / "04_gather_with_curves.png",
            **opts,
        )
    else:
        print("当前关闭空洞，跳过 S-D-G 绕射路径图。")

    print("\nStep 5：定位扫描。")
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
        output=outdir / "05_residual_best_curve.png",
        **opts,
    )
    plot_score_slices(
        scan,
        true_x=cavities[0].x0 if cavities else None,
        true_y=cavities[0].y0 if cavities else None,
        true_h=cavities[0].h if cavities else None,
        output=outdir / "05_scan_score_slices.png",
        **opts,
    )
    print(f"直达波拟合 VR = {fit.velocity:.1f} m/s")
    print(f"拟合 t0 = {fit.t0:.4f} s, RMS = {fit.residual_rms:.4f} s")
    print(f"最佳疑似异常体 x/y/h/VR = {best.x0:.1f} / {best.y0:.1f} / {best.h:.1f} / {best.velocity:.1f}")
    if cavities:
        c = cavities[0]
        print(f"真实空洞位置 x/y/h = {c.x0:.1f} / {c.y0:.1f} / {c.h:.1f}")
    print(f"不确定性范围 = {scan.uncertainty}")
    print("解释提醒：单侧 DAS + 对侧锤击下，x 通常更稳定，y-h 存在耦合，应输出范围而非唯一确诊点。")

    print("\nStep 6：可选运动学波场动画。")
    animate = bool(args.animate) and not bool(args.no_animate)
    if animate and cavities:
        animate_kinematic_wavefield(
            geom,
            cavities[0],
            shot_index,
            cfg.velocity.rayleigh_velocity,
            outdir / "06_kinematic_wavefield.gif",
            t0=cfg.record.t0,
            n_frames=args.frames,
            fps=args.fps,
            save=bool(opts["save"]),
            show=bool(opts["show"]),
        )
        print("已生成 06_kinematic_wavefield.gif。该动画是等效运动学传播示意，不是严格弹性波场快照。")
    else:
        print("未生成动画。需要动画时运行：python main.py workflow --animate --save，输出为 outputs/workflow/06_kinematic_wavefield.gif")

    print("\nStep 7：结果总结。")
    save_run_parameters(cfg, outdir, bool(opts["save"]))
    print("本次流程完成。")
    print("本方法输出的是疑似异常范围，不是直接确诊空洞。")
    print("真实数据应用前仍需光纤路径标定、锤击触发校正、通道耦合 QC、浅层速度估计和管线/井盖干扰核查。")


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
        add_output_args(p)
        p.set_defaults(func=handler)
    return parser


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("--"):
        argv = ["workflow", *argv]
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
