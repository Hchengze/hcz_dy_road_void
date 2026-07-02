"""高层 workflow runner。

``main.py`` 只负责参数解析和调用本模块；本模块负责把 scenario、dataset、
diffraction、inversion、wavefield 和 report 串成同一条主链。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RoadVoidConfig
from .config_builder import select_wavefield_shots, velocity_plot_info
from .dataset import (
    ResearchSurveyDataset,
    generate_synthetic_survey_dataset,
    plot_das_like_gather,
    plot_noise_components,
    save_synthetic_survey_dataset,
)
from .diffraction import detect_diffraction_features, plot_diffraction_attribute, plot_diffraction_candidates
from .inversion import (
    plot_localization_error_summary,
    plot_uncertainty_summary,
    run_joint_localization_evaluation,
    write_research_report,
)
from .output import OutputManifest, save_run_parameters
from .scenario import build_default_subsurface_scenario, plot_subsurface_3d, plot_subsurface_sections
from .visualization import (
    animate_kinematic_wavefield,
    animate_kinematic_wavefield_3d,
    animate_multishot_kinematic_wavefield,
    plot_diffraction_path_demo,
    plot_geometry_plan_and_sections,
    plot_kinematic_wavefield_frames,
    plot_kinematic_wavefield_frames_3d,
    plot_multishot_scan_diagnostics,
    plot_road_geometry_3d,
    plot_score_slices,
    plot_shot_gather,
    plot_velocity_model,
)
from .workflow import run_location_workflow


WORKFLOW_STEPS = [
    ("geometry", "建立道路三维几何：道路、光纤、炮线、空洞"),
    ("velocity", "展示等效瑞雷波速度/速度模型，并说明 lambda = VR / f"),
    ("forward", "锤击激发三维等效瑞雷波正演，生成单侧 DAS shot gather"),
    ("dataset", "生成结构化 synthetic survey dataset 与 DAS-like response"),
    ("diffraction", "提取绕射/散射属性和候选评分"),
    ("scan", "直达波拟合、残差构建、多炮联合扫描和疑似异常体定位"),
    ("wavefield", "输出 2D/3D 运动学波场关键帧，按需生成 GIF"),
    ("summary", "输出研究报告、参数记录和输出清单"),
]


def write_dataset_metadata(dataset: ResearchSurveyDataset, output: Path, enabled: bool) -> Path:
    """保存轻量 metadata/labels，不默认保存体积更大的 npz 数据体。"""

    if not enabled:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(_json_ready({"metadata": dataset.metadata, "labels": dataset.labels}), f, ensure_ascii=False, indent=2)
    return output


def _json_ready(value: Any) -> Any:
    """把 numpy 标量/数组递归转换为 JSON 可写对象。"""

    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def run_full_workflow(
    args: Any,
    cfg: RoadVoidConfig,
    outdir: Path,
    opts: dict[str, object],
    manifest: OutputManifest,
) -> None:
    """运行完整科研 workflow。

    所有步骤复用同一个 ``RoadVoidConfig``、同一次 ``run_location_workflow`` 结果和同一个
    ``ResearchSurveyDataset``。这能避免 scenario、dataset、diffraction、inversion 各自
    偷偷重建默认数据。
    """

    geom = cfg.to_geometry()
    cavities = cfg.to_cavities()
    scenario = build_default_subsurface_scenario(cfg)
    workflow = run_location_workflow(cfg)
    dataset = workflow.dataset
    fit = workflow.velocity_fit
    scan = workflow.scan_result
    best = scan.best
    survey_dataset = generate_synthetic_survey_dataset(cfg, workflow)
    diffraction_result = detect_diffraction_features(cfg, workflow)
    localization_eval = run_joint_localization_evaluation(cfg, workflow)
    vr_eff = cfg.effective_rayleigh_velocity()
    velocity_info = velocity_plot_info(cfg)
    shot_index = _representative_shot(args, geom, cavities)

    print("完整 workflow 开始。步骤顺序：")
    print(f"输出写入 {outdir}")
    print(f"output_dir={outdir.as_posix()}")
    for idx, (_, desc) in enumerate(WORKFLOW_STEPS, start=1):
        print(f"  Step {idx}: {desc}")

    print("\nStep 1：构建三维道路场景。")
    print("坐标说明：x 沿道路/光纤方向，y 横穿道路方向，z 为深度；光纤位于 y=0，锤击线位于 y=W。")
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

    print("\nStep 2：展示速度/频率与地下模型。")
    wavelength = cfg.velocity.rayleigh_velocity / cfg.velocity.source_frequency
    print(f"速度模式 velocity-mode = {cfg.velocity.velocity_model_type}")
    print(f"输入参考瑞雷波速度 VR = {cfg.velocity.rayleigh_velocity:.1f} m/s")
    print(f"当前实际用于正演/扫描的 VR_eff = {vr_eff:.1f} m/s")
    print(f"锤击主频 f = {cfg.velocity.source_frequency:.1f} Hz, lambda = {wavelength:.2f} m")
    plot_velocity_model(
        cfg.to_velocity_model(),
        (float(geom.channel_x[0]), float(geom.channel_x[-1])),
        cavities,
        manifest.add(outdir / "02_velocity_model.png"),
        effective_velocity=vr_eff,
        velocity_info=velocity_info,
        **opts,
    )
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

    print("\nStep 3：正演数据与 DAS-like response。")
    print(f"合成数据形状: {dataset.data.shape} = shots x times x channels")
    plot_shot_gather(
        dataset.data,
        geom,
        shot_index=shot_index,
        direct_times=dataset.direct_times,
        diffraction_times=dataset.diffraction_times[0] if dataset.diffraction_times else None,
        title="03 三维等效瑞雷波正演记录",
        output=manifest.add(outdir / "03_forward_gather.png"),
        **opts,
    )
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

    print("\nStep 4：传播路径和绕射/散射属性。")
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
        print("当前关闭异常体，跳过 S-D-G 绕射路径图。")
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

    print("\nStep 5：多炮联合定位与误差评估。")
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
    if bool(getattr(args, "save_extra", False)):
        plot_multishot_scan_diagnostics(scan, outdir, **opts)
        for name in ("per_shot_best_x.png", "per_shot_score_contribution.png", "single_shot_vs_joint.png"):
            manifest.add(outdir / name)
    _print_scan_summary(cfg, fit, scan, best, cavities)

    print("\nStep 6：运动学波场示意关键帧。")
    if cavities:
        _plot_workflow_wavefield(args, cfg, geom, cavities, outdir, opts, manifest, shot_index, vr_eff, velocity_info)
    else:
        print("当前无异常体，wavefield 只保留几何/速度上下文；报告中会说明未生成散射关键帧。")

    print("\nStep 7：研究记录。")
    if bool(getattr(args, "save_extra", False)) and bool(opts["save"]):
        npz_path, _ = save_synthetic_survey_dataset(survey_dataset, outdir)
        manifest.add(npz_path)
    metadata_path = write_dataset_metadata(
        survey_dataset,
        outdir / "synthetic_dataset_metadata.json",
        bool(opts["save"]),
    )
    if bool(opts["save"]):
        manifest.add(metadata_path)
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
    params_path = save_run_parameters(cfg, outdir, bool(opts["save"]))
    manifest.add(params_path)

    print("本次流程完成。")
    print(f"速度总结：velocity-mode={cfg.velocity.velocity_model_type}, VR_eff={vr_eff:.1f} m/s。")
    print(f"异常体总结：本次异常体数量={len(cavities)}，shape 仅表示等效散射几何，不是真实弹性边界散射。")
    print("定位结果应解释为疑似异常范围，不是直接确诊空洞。")
    manifest.write_and_print()


def _representative_shot(args: Any, geom: Any, cavities: list[Any]) -> int:
    shots = select_wavefield_shots(args, geom, cavities)
    if shots:
        return shots[0]
    return geom.n_shots // 2


def _print_scan_summary(cfg: RoadVoidConfig, fit: Any, scan: Any, best: Any, cavities: list[Any]) -> None:
    print(f"扫描模式 scan-mode = {cfg.processing.scan_mode}；炮权重 shot-weight-mode = {cfg.processing.shot_weight_mode}")
    print(f"直达波拟合 VR = {fit.velocity:.1f} m/s")
    print(f"拟合 t0 = {fit.t0:.4f} s, RMS = {fit.residual_rms:.4f} s")
    print(f"最佳疑似异常体 x/y/h/VR = {best.x0:.1f} / {best.y0:.1f} / {best.h:.1f} / {best.velocity:.1f}")
    if cavities:
        c = cavities[0]
        print(f"真实主异常体位置 x/y/h = {c.x0:.1f} / {c.y0:.1f} / {c.h:.1f}")
    print(f"不确定性范围 = {scan.uncertainty}")
    if scan.consistency:
        cns = scan.consistency
        print(f"单炮结果离散程度：x_std={cns['x_std']:.2f}, y_std={cns['y_std']:.2f}, h_std={cns['h_std']:.2f}")
    print("解释提醒：单侧 DAS + 对侧锤击下，x 通常更稳定，y-h 存在耦合，应输出范围而非唯一确诊点。")


def _plot_workflow_wavefield(
    args: Any,
    cfg: RoadVoidConfig,
    geom: Any,
    cavities: list[Any],
    outdir: Path,
    opts: dict[str, object],
    manifest: OutputManifest,
    shot_index: int,
    vr_eff: float,
    velocity_info: dict[str, object],
) -> None:
    """输出 workflow 的代表性 2D plan 和 3D kinematic wavefield 关键帧。"""

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
        velocity_info=velocity_info,
        filename_prefix="06_wavefield",
    )
    for frame_name in (
        "06_wavefield_frame_early.png",
        "06_wavefield_frame_hit_cavity.png",
        "06_wavefield_frame_scattered.png",
    ):
        manifest.add(outdir / frame_name)

    for output in plot_kinematic_wavefield_frames_3d(
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
    ):
        manifest.add(output)

    animate = bool(getattr(args, "animate", False)) and not bool(getattr(args, "no_animate", False))
    if not animate:
        print("no_animation_generated")
        print("已输出 2D plan 与 3D kinematic wavefield 关键帧；未生成 GIF。")
        return

    if getattr(args, "wavefield_view", "plan") == "3d":
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
            velocity_info=velocity_info,
        )
        print("已生成 06_wavefield_3d.gif；它是三维运动学等时面示意，不是完整弹性波场。")
    elif getattr(args, "wavefield_mode", "single-shot") == "multi-shot":
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
            velocity_info=velocity_info,
        )
        print("已生成 06_multishot_wavefield.gif；它是多炮顺序激发示意，不是多炮联合反演。")
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
            velocity_info=velocity_info,
        )
        print("已生成 06_kinematic_wavefield.gif；它是 x-y 平面运动学示意，不是严格弹性波场快照。")
