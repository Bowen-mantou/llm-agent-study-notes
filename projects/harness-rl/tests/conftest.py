"""让 pytest 在未安装本目录为包的情况下也能 import 09-harness 顶层模块。

pytest 默认只把测试文件所在目录加进 sys.path（本仓库没有把 09-harness 做成 package），
所以这里显式把项目根目录插到最前面。运行方式：

    cd 09-harness
    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
