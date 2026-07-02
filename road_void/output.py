"""输出目录、清单和运行参数记录的轻量工具。

本模块只处理“文件写到哪里、哪些文件是本次生成的”这类工程性逻辑。
把这些函数从 ``main.py`` 中下沉出来，可以避免 workflow、wavefield、elastic3d
等入口各自维护一套输出规则。
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import rmtree
from typing import Any

from .config import RoadVoidConfig


def output_options(args: Any) -> dict[str, object]:
    """统一解释 ``--save/--no-save/--show``。

    默认 workflow 是项目主线：如果用户运行 ``python main.py workflow`` 且没有显式
    ``--no-save``，就保存完整 workflow 结果。其它专家子命令仍保持“显式 --save 才保存”
    的轻量调试习惯。
    """

    command = getattr(args, "command", "")
    default_save = command in {"workflow", "all"} and not bool(getattr(args, "no_save", False))
    save = (bool(getattr(args, "save", False)) or default_save) and not bool(getattr(args, "no_save", False))
    show = bool(getattr(args, "show", False)) and not bool(getattr(args, "no_show", False))
    return {"save": save, "show": show, "dpi": getattr(args, "dpi", 150)}


def command_outdir(args: Any, name: str) -> Path:
    """返回子命令输出目录。

    ``wavefield`` 是 workflow 第 6 步，因此默认也写入 ``outputs/workflow``，
    避免重新生成独立的 ``outputs/wavefield`` 体系。
    """

    if name in {"workflow", "wavefield"} and not getattr(args, "outdir", None):
        return Path("outputs") / "workflow"
    if name == "elastic3d_validation" and not getattr(args, "outdir", None):
        return Path("outputs") / "elastic3d_validation"
    return Path(args.outdir) if getattr(args, "outdir", None) else Path("outputs") / name


class OutputManifest:
    """记录本次运行实际生成的文件。

    只记录当前 run 写出的路径，帮助区分新结果和历史旧图。``save=False`` 时仍把
    路径返回给绘图函数，但不会写入 manifest。
    """

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


def clean_output_dir(outdir: Path) -> None:
    """只清理当前输出目录中的常见结果文件。

    这个函数不会清理整个 ``outputs``，也不会跨目录删除 numerics、elastic3d_validation
    等支线结果。
    """

    if not outdir.exists():
        return
    allowed = {".png", ".gif", ".mp4", ".json", ".txt", ".csv", ".npz"}
    for item in outdir.iterdir():
        if item.is_file() and item.suffix.lower() in allowed:
            item.unlink()
        elif item.is_dir() and item.name.startswith("_tmp"):
            rmtree(item)
    print(f"已清理当前输出目录中的旧结果文件: {outdir}")


def prepare_output_dir(args: Any, name: str) -> tuple[Path, dict[str, object], OutputManifest]:
    """统一处理 outdir、clean-output 和输出清单。"""

    outdir = command_outdir(args, name)
    opts = output_options(args)
    if bool(getattr(args, "clean_output", False)):
        clean_output_dir(outdir)
    return outdir, opts, OutputManifest(outdir, bool(opts["save"]))


def save_run_parameters(cfg: RoadVoidConfig, outdir: Path, enabled: bool) -> Path:
    """把本次 workflow 使用的核心参数保存为 JSON 运行记录。"""

    path = outdir / "run_parameters.json"
    if not enabled:
        return path
    outdir.mkdir(parents=True, exist_ok=True)
    data = {
        "geometry": cfg.geometry.__dict__,
        "cavity": cfg.cavity.__dict__,
        "velocity": {
            **cfg.velocity.__dict__,
            "velocity_mode": cfg.velocity.velocity_model_type,
            "effective_rayleigh_velocity": cfg.effective_rayleigh_velocity(),
        },
        "record": cfg.record.__dict__,
        "noise": cfg.noise.__dict__,
        "processing": cfg.processing.__dict__,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
