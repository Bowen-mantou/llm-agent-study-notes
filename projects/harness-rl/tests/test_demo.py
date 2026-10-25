"""demo.py 的集成测试：真实运行一遍 demo，断言退出码和关键输出。

这一条测试等价于「python demo.py 能跑通」的可回归版本。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_DIR / "demo.py"


def test_demo_runs_end_to_end() -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    completed = subprocess.run(
        [sys.executable, str(DEMO_PATH)],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        shell=False,
        env=env,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "part 1/3" in output
    assert "part 2/3" in output
    assert "part 3/3" in output
    assert "demo finished: all checks passed" in output
    # 三个单条 episode 都成功，且批跑指标出现了 solved_rate
    assert "solved=True" in output
    assert "harness/solved_rate" in output
    # 沙箱边界检查四项关键项
    for check in ("normal stdout", "timeout is killed", "output truncated", "non-zero exit code"):
        assert check in output
