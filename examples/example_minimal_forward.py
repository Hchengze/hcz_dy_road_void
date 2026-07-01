"""最小正演示例：直接复用 main.py 的 forward 子命令。

日常使用建议优先运行：

    python main.py forward --show --no-save
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("--"):
        sys.argv.insert(1, "forward")
    main()
