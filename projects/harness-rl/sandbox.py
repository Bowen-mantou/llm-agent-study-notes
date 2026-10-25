"""09-harness 的子进程沙箱：隔离执行 + 超时 + 输出截断 + 退出码处理。

为什么必须要有这一层
====================

Agent RL 的 rollout 里，工具参数是**模型生成的不可信输入**。哪怕只是
``calculator``，也可能收到 ``__import__("os").system(...)``。所以真实项目里
工具执行必须和训练进程隔开：训练进程挂了，整个 rollout 队列就废了。

本文件提供一个最小可用的本地沙箱（不要求 docker / firejail）：

1. **隔离**：``subprocess.Popen(argv, shell=False)``，argv 用 list 传参，
   模型给的代码通过 ``-c`` 传入，不经过 shell，没有注入面；
2. **超时**：wall-clock 超时后杀进程。POSIX 上 ``start_new_session`` 建独立进程组后
   ``killpg``；Windows 上没有 SIGKILL/进程组语义，用 ``taskkill /F /T`` 连子树一起杀；
3. **截断**：stdout/stderr 写到匿名临时文件而不是 PIPE，只从文件尾部读有限字节，
   防止 ``print("x" * 10**8)`` 这类输出洪水把父进程内存打爆；
4. **退出码**：非 0 退出码 + stderr 原样回喂给模型（论文里「自我纠错」就靠这个信号）。

Windows 兼容要点（这个文件踩过的坑）
------------------------------------

- 没有 ``os.killpg``，直接用 ``process.kill()`` 杀不掉子进程树，必须走 taskkill；
- ``python -c`` 的输出重定向到文件时默认用 locale 编码（中文 Windows 是 cp936），
  父进程按 UTF-8 解码就会乱码，所以子进程统一加 ``-X utf8``；
- 命令行长度有限（Windows 约 32K 字符），过长的 code 直接拒绝并回喂错误；
- 沙箱测试自己的子进程不能继承 pytest 的 ``sys.path``，用 ``-I`` 隔离模式启动。

真实项目（比如 ReTool / SWE-agent）会把这一层换成 SandboxFusion / Docker /
gVisor；本文件的接口 :meth:`SubprocessSandbox.run` 不变，替换实现即可。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from protocol import clip_text, sanitize_control_markers

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_OUTPUT_CHARS = 4096
# 最多从输出文件尾部读这么多字节再按字符截断；先限字节是为了防输出洪水撑爆内存。
_READ_CAP_BYTES = 64 * 1024
# Windows 命令行长度上限约 32767，留一半余量给 argv 其它部分。
_MAX_CODE_CHARS = 12_000
# 子进程环境变量：强制 UTF-8 + 把 BLAS 线程压到 1，避免并发沙箱把机器打满。
_CHILD_ENV_OVERRIDES = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class SandboxResult:
    """一次沙箱执行的完整结果。

    :param argv: 实际执行的命令行（list 形式，便于复盘）；
    :param ok: 进程正常退出且退出码为 0；
    :param stdout: 截断后的标准输出；
    :param stderr: 截断后的标准错误；
    :param returncode: 退出码；进程被信号杀死时为负值，未回收时为 None；
    :param timed_out: 是否因 wall-clock 超时被杀死；
    :param truncated: 输出是否被截断；
    :param latency_s: 从启动到回收的耗时（秒）。
    """

    argv: tuple[str, ...]
    ok: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    truncated: bool
    latency_s: float


@dataclass
class SandboxStats:
    """累计沙箱调用指标，字段命名对齐 repo 里 SwanLab 的 ``sandbox/*`` 指标。"""

    calls: int = 0
    successes: int = 0
    timeouts: int = 0
    errors: int = 0
    truncated: int = 0
    latency_total: float = 0.0

    def metrics(self) -> dict[str, float]:
        """转换成比例指标，方便直接打进训练日志。"""
        denominator = max(self.calls, 1)
        return {
            "sandbox/calls": float(self.calls),
            "sandbox/success_rate": self.successes / denominator,
            "sandbox/error_rate": self.errors / denominator,
            "sandbox/timeout_rate": self.timeouts / denominator,
            "sandbox/truncated_rate": self.truncated / denominator,
            "sandbox/latency_mean": self.latency_total / denominator,
        }


def _read_tail_capped(file: object, cap_bytes: int) -> tuple[str, bool]:
    """从输出文件尾部读取有限字节，返回 ``(文本, 是否超过 cap)``。

    读的是文件而不是 PIPE：子进程写多少都不会让父进程内存跟着涨。
    """
    file.seek(0, os.SEEK_END)  # type: ignore[attr-defined]
    size = file.tell()  # type: ignore[attr-defined]
    file.seek(max(0, size - cap_bytes))  # type: ignore[attr-defined]
    data = file.read()  # type: ignore[attr-defined]
    text = data.decode("utf-8", errors="replace")
    # Windows 子进程 stdout 重定向到文件时会写 \r\n，统一成 \n，让 observation 跨平台一致。
    text = text.replace("\r\n", "\n")
    return text, size > cap_bytes


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    """杀死进程及其子进程。

    Windows 没有 POSIX 的进程组信号：``taskkill /T`` 才是把子树一起带走的正确做法，
    ``process.kill()`` 只保证杀死直接子进程，作为兜底再调一次。
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            process.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


class SubprocessSandbox:
    """本机 Python 子进程沙箱。

    线程安全：内部用 ``threading.BoundedSemaphore`` 限并发（不用 asyncio.Semaphore，
    因为它会绑定到单个事件循环，而训练每个 step 都可能新建 loop）。
    """

    def __init__(
        self,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        python: str = sys.executable,
        max_workers: int = 4,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")
        self.timeout_s = float(timeout_s)
        self.max_output_chars = int(max_output_chars)
        self.python = python
        self.stats = SandboxStats()
        self._semaphore = threading.BoundedSemaphore(max(1, int(max_workers)))

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def run_python(self, code: str, *, timeout_s: float | None = None) -> SandboxResult:
        """在全新 Python 子进程里执行一段代码。

        用 ``-I``（隔离模式，不加载 cwd / 用户 site-packages）+ ``-X utf8`` 保证
        结果可复现；代码不落盘，直接通过 argv 传给子进程。
        """
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if len(code) > _MAX_CODE_CHARS:
            raise ValueError(f"code exceeds {_MAX_CODE_CHARS} characters")
        argv = [self.python, "-I", "-B", "-X", "utf8", "-c", code]
        return self.run(argv, timeout_s=timeout_s)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float | None = None,
        cwd: str | Path | None = None,
    ) -> SandboxResult:
        """执行一条命令（list 形式的 argv，``shell=False``）。

        :param argv: 命令与参数，例如 ``[sys.executable, "-c", "print(42)"]``；
        :param timeout_s: 本次调用的超时，缺省用实例默认值；
        :param cwd: 子进程工作目录，缺省继承父进程。
        """
        command = [str(part) for part in argv]
        if not command:
            raise ValueError("argv must not be empty")
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")

        env = {**os.environ, **_CHILD_ENV_OVERRIDES}
        popen_kwargs: dict[str, object] = {
            "stdout": None,
            "stderr": None,
            "stdin": subprocess.DEVNULL,
            "shell": False,
            "env": env,
            "cwd": str(cwd) if cwd is not None else None,
        }
        # Windows 没有 start_new_session；CREATE_NEW_PROCESS_GROUP 让子进程成为独立组，
        # 便于 taskkill /T 精确清理；POSIX 用 start_new_session + killpg。
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        started = time.perf_counter()
        with self._semaphore:
            # 输出落临时文件而不是 PIPE：防输出洪水撑爆父进程内存。
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                popen_kwargs["stdout"] = stdout_file
                popen_kwargs["stderr"] = stderr_file
                process = subprocess.Popen(command, **popen_kwargs)  # type: ignore[arg-type]
                timed_out = False
                try:
                    returncode: int | None = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_process_tree(process)
                    try:
                        returncode = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:  # 理论上不会发生，保底防挂死
                        returncode = None
                stdout_raw, stdout_capped = _read_tail_capped(stdout_file, _READ_CAP_BYTES)
                stderr_raw, stderr_capped = _read_tail_capped(stderr_file, _READ_CAP_BYTES)

        stdout, stdout_clipped = clip_text(stdout_raw, self.max_output_chars)
        stderr, stderr_clipped = clip_text(stderr_raw, self.max_output_chars)
        truncated = stdout_capped or stderr_capped or stdout_clipped or stderr_clipped
        latency = time.perf_counter() - started
        ok = (not timed_out) and returncode == 0

        self.stats.calls += 1
        self.stats.latency_total += latency
        if timed_out:
            self.stats.timeouts += 1
        elif ok:
            self.stats.successes += 1
        else:
            self.stats.errors += 1
        if truncated:
            self.stats.truncated += 1

        return SandboxResult(
            argv=tuple(command),
            ok=ok,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            truncated=truncated,
            latency_s=latency,
        )

    # ------------------------------------------------------------------ #
    # 回喂格式
    # ------------------------------------------------------------------ #

    def format_for_model(self, result: SandboxResult) -> str:
        """把执行结果格式化成可回填进上下文的文本（已做特殊标记无害化）。"""
        if result.timed_out:
            return f"Error [sandbox_timeout]: execution exceeded {self.timeout_s:g}s and was killed."
        if result.ok:
            return sanitize_control_markers(result.stdout) if result.stdout else "(no output)"
        detail = result.stderr.strip() or f"process exited with code {result.returncode}"
        return "Error [sandbox_error]: " + sanitize_control_markers(detail)
