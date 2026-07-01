"""正演模型兼容入口。

项目核心正演类仍定义在 ``forward.py``。本文件提供更直观的模块名，
方便教学展示和后续文档引用。
"""

from .forward import ForwardModelConfig, RayleighKinematicForwardModel, SyntheticDataset

__all__ = ["ForwardModelConfig", "RayleighKinematicForwardModel", "SyntheticDataset"]
