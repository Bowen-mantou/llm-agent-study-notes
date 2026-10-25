"""sandbox.py 的单元测试：子进程隔离、超时、截断、退出码、UTF-8、shell=False。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sandbox import SubprocessSandbox


def test_run_python_ok() -> None:
    sandbox = SubprocessSandbox()
    result = sandbox.run_python('print("child says", 6 * 7)')
    assert result.ok is True
    assert result.timed_out is False
    assert result.returncode == 0
    assert "42" in result.stdout
    # Windows 上 \r\n 已在沙箱层统一成 \n
    assert "\r" not in result.stdout
    assert result.latency_s >= 0


def test_run_python_unicode_output() -> None:
    sandbox = SubprocessSandbox()
    result = sandbox.run_python('print("沙箱-中文-ok")')
    assert result.ok is True
    assert "沙箱-中文-ok" in result.stdout


def test_run_python_nonzero_exit_keeps_stderr() -> None:
    sandbox = SubprocessSandbox()
    result = sandbox.run_python('raise ValueError("boom")')
    assert result.ok is False
    assert result.returncode != 0
    assert "boom" in result.stderr


def test_run_python_timeout_kills_process() -> None:
    sandbox = SubprocessSandbox(timeout_s=0.5)
    result = sandbox.run_python("import time\ntime.sleep(5)")
    assert result.timed_out is True
    assert result.ok is False
    assert result.latency_s < 4.0


def test_run_python_truncates_output_flood() -> None:
    sandbox = SubprocessSandbox(max_output_chars=256)
    result = sandbox.run_python('print("x" * 20000)')
    assert result.ok is True
    assert result.truncated is True
    assert len(result.stdout) <= 256
    assert "truncated" in result.stdout


def test_run_generic_argv_is_not_shell_interpreted() -> None:
    sandbox = SubprocessSandbox()
    result = sandbox.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            "utf8",
            "-c",
            "import sys; print(sys.argv[1])",
            "a; echo injected",
        ]
    )
    assert result.ok is True
    assert result.stdout.strip() == "a; echo injected"
    assert result.argv[-1] == "a; echo injected"


def test_run_records_argv() -> None:
    sandbox = SubprocessSandbox()
    result = sandbox.run([sys.executable, "-I", "-c", "print('hi')"])
    assert result.argv[0] == sys.executable
    assert result.argv[-1] == "print('hi')"


def test_sandbox_rejects_invalid_arguments() -> None:
    sandbox = SubprocessSandbox()
    with pytest.raises(ValueError):
        sandbox.run([])
    with pytest.raises(ValueError):
        sandbox.run_python("   ")
    with pytest.raises(ValueError):
        sandbox.run_python("print(1)", timeout_s=0)
    with pytest.raises(ValueError):
        sandbox.run_python("x" * 20_000)


def test_sandbox_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        SubprocessSandbox(timeout_s=0)
    with pytest.raises(ValueError):
        SubprocessSandbox(max_output_chars=0)


def test_sandbox_stats_metrics() -> None:
    sandbox = SubprocessSandbox(timeout_s=0.4, max_workers=2)
    sandbox.run_python("print(1)")
    sandbox.run_python("import time\ntime.sleep(5)")
    metrics = sandbox.stats.metrics()
    assert metrics["sandbox/calls"] == 2
    assert metrics["sandbox/success_rate"] == pytest.approx(0.5)
    assert metrics["sandbox/timeout_rate"] == pytest.approx(0.5)
    assert sandbox.stats.timeouts == 1


def test_format_for_model() -> None:
    sandbox = SubprocessSandbox(timeout_s=0.4)
    ok_result = sandbox.run_python("print(42)")
    assert sandbox.format_for_model(ok_result).strip() == "42"

    timeout_result = sandbox.run_python("import time\ntime.sleep(5)")
    assert "sandbox_timeout" in sandbox.format_for_model(timeout_result)

    error_result = sandbox.run_python('raise ValueError("bad")')
    formatted = sandbox.format_for_model(error_result)
    assert "sandbox_error" in formatted
    assert "bad" in formatted


def test_sandbox_does_not_write_files(tmp_path: Path) -> None:
    """子进程 cwd 默认继承父进程；这里只验证 run(cwd=...) 参数能正常工作。"""
    sandbox = SubprocessSandbox()
    result = sandbox.run(
        [sys.executable, "-I", "-c", "import os; print(os.getcwd())"],
        cwd=tmp_path,
    )
    assert result.ok is True
    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()
