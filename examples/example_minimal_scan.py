"""最小定位扫描示例：直接复用 main.py 的 scan 子命令。

运行方式：

    python examples/example_minimal_scan.py scan --no-save
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("--"):
        sys.argv.insert(1, "scan")
    main()
