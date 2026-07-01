"""配置驱动的正演、处理和定位工作流。

这里不引入新的反演算法，只把已有模块按配置文件串起来，方便 main.py、
示例脚本、参数敏感性分析和单元测试复用同一条闭环。
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from .config import RoadVoidConfig
from .forward import RayleighKinematicForwardModel, SyntheticDataset
from .processing import fit_direct_velocity, mute_direct_wave, pick_direct_arrivals, subtract_direct_wave_template, trace_normalize
from .scan import CavityScanResult, scan_cavity_diffraction


FloatArray = NDArray


@dataclass
class WorkflowResult:
    """配置驱动定位流程的输出集合。"""

    dataset: SyntheticDataset
    picks: FloatArray
    velocity_fit: object
    residual: FloatArray
    scan_result: CavityScanResult


def simulate_from_config(config: RoadVoidConfig) -> SyntheticDataset:
    """根据配置生成合成数据。"""

    geom = config.to_geometry()
    forward_cfg = config.to_forward_config()
    cavities = config.to_cavities()
    return RayleighKinematicForwardModel(geom, forward_cfg).simulate(cavities)


def run_location_workflow(config: RoadVoidConfig) -> WorkflowResult:
    """运行“正演 -> 直达波拟合 -> 直达波压制 -> 绕射扫描”闭环。"""

    dataset = simulate_from_config(config)
    geom = dataset.geometry
    picks = pick_direct_arrivals(
        dataset.data,
        geom,
        velocity_hint=config.velocity.rayleigh_velocity,
        t0_hint=config.record.t0,
        search_half_width=max(0.03, config.processing.direct_wave_mute_width),
    )
    fit = fit_direct_velocity(picks, geom)
    if config.processing.direct_wave_subtraction_enable:
        residual = subtract_direct_wave_template(
            dataset.data,
            geom,
            fit.velocity,
            fit.t0,
            frequency=config.velocity.source_frequency,
            wavelet=config.velocity.wavelet_type,
            fit_half_width=config.processing.direct_wave_mute_width,
        )
    else:
        residual = mute_direct_wave(
            dataset.data,
            geom,
            fit.velocity,
            fit.t0,
            half_width=config.processing.direct_wave_mute_width,
        )
    residual = trace_normalize(residual)
    use_envelope = config.processing.score_method in {"envelope", "coherence", "semblance-like"}
    scan_result = scan_cavity_diffraction(
        residual,
        geom,
        config.to_scan_grid(),
        t0=fit.t0,
        use_envelope=use_envelope,
        half_window=0.012,
        top_k=config.processing.top_k,
        confidence_fraction=config.processing.uncertainty_threshold,
    )
    return WorkflowResult(
        dataset=dataset,
        picks=picks,
        velocity_fit=fit,
        residual=residual,
        scan_result=scan_result,
    )
