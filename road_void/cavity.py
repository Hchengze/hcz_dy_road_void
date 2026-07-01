"""空洞模型兼容入口。

项目最初把异常体模型放在 ``anomaly.py``。本文件只做中文语义更直观的
导入转发，便于后续阅读代码时从 ``road_void.cavity`` 找到 ``Cavity``。
"""

from .anomaly import Cavity

__all__ = ["Cavity"]
