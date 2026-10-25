"""09-harness 的最小 demo：无 GPU、无 API key、无第三方运行时依赖。

demo 证明三件事：

1. **完整 agent loop 能跑通**：脚本化假 LLM 按预设输出模型行为，
   完整经历「生成 → 解析 → 工具执行 → observation 回填 → 下一轮」；
2. **harness 能并发批跑并出指标**：同一批任务用 ``run_batch`` 并发执行，
   聚合 solved_rate / reward / turns / 错误数；
3. **沙箱边界是真的**：子进程隔离、超时杀进程、输出截断、非零退出码、UTF-8 中文。

三个用例：

====================  ==========================================================
任务                  演示的 harness 能力
====================  ==========================================================
``calc-chain``        多轮 ReAct + calculator + 最终答案
``docs-lookup``       search_docs 检索 + 受限 read_file + 答案
``resilience``        未知工具 / 非法 JSON 的纠错，以及子进程 run_python
====================  ==========================================================

运行方式（两种都可以）::

    cd 09-harness
    python demo.py

    # 或者从仓库根目录
    python 09-harness/demo.py

退出码：全部内部检查通过为 0，否则为 1（所以它本身也是一个 smoke test）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from harness import Harness, LoopConfig, Task  # noqa: E402
from sandbox import SubprocessSandbox  # noqa: E402


class ScriptedPolicy:
    """脚本化假 LLM：按顺序返回预设的 assistant 文本。

    它不读 observation，只负责「像模型一样一轮一轮输出」，让 demo 在没有模型、
    没有网络的环境下也能确定性地复现一条 agent 轨迹。接真模型时，
    把 :meth:`__call__` 换成一次 chat completion 即可（见 readme.md）。
    """

    def __init__(self, script: list[str], *, name: str = "scripted-policy") -> None:
        self._script = list(script)
        self._cursor = 0
        self.name = name

    def __call__(self, messages: list[dict[str, object]], tools: list[object]) -> str:
        if self._cursor >= len(self._script):
            # 脚本耗尽时返回空串：会被协议层判成 empty_output，
            # 既能暴露脚本写漏的问题，又不会把 demo 直接炸掉。
            return ""
        text = self._script[self._cursor]
        self._cursor += 1
        return text


# --------------------------------------------------------------------------- #
# 三个 demo 用例
# --------------------------------------------------------------------------- #

TASKS: list[Task] = [
    Task(
        task_id="calc-chain",
        prompt=(
            "Compute (12 + 30) * 2 / 2 step by step. "
            "Use the calculator tool, then answer with the final number."
        ),
        expected_answer="42",
    ),
    Task(
        task_id="docs-lookup",
        prompt=(
            "Search the local docs for the harness component that isolates tool execution, "
            "read workspace/notes.md, then answer with that component name (one word)."
        ),
        expected_answer="sandbox",
    ),
    Task(
        task_id="resilience",
        prompt=(
            "Compute 6 * 7 with run_python. If a tool call fails, read the error and retry "
            "with a different tool. Answer with the number."
        ),
        expected_answer="42",
    ),
]

SCRIPTS: dict[str, list[str]] = {
    "calc-chain": [
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "(12 + 30) * 2"}}\n</tool_call>',
        '<tool_call>\n{"name": "calculator", "arguments": {"expression": "84 / 2"}}\n</tool_call>',
        "The chain gives 84 then 42.\n<answer>42</answer>",
    ],
    "docs-lookup": [
        (
            '<tool_call>\n{"name": "search_docs", '
            '"arguments": {"query": "execution isolation timeout", "top_k": 2}}\n</tool_call>'
        ),
        (
            '<tool_call>\n{"name": "read_file", '
            '"arguments": {"path": "workspace/notes.md", "max_chars": 1200}}\n</tool_call>'
        ),
        "<answer>sandbox</answer>",
    ],
    "resilience": [
        # 1) 调用一个不存在的工具：工具层返回 unknown_tool，模型应改正
        '<tool_call>\n{"name": "multiply", "arguments": {"a": 6, "b": 7}}\n</tool_call>',
        # 2) 故意缺少右花括号：协议层返回 invalid_json，loop 回喂纠错提示
        '<tool_call>\n{"name": "run_python", "arguments": {"code": "print(6 * 7)"}\n</tool_call>',
        # 3) 这次格式正确，代码在子进程沙箱里真的执行
        '<tool_call>\n{"name": "run_python", "arguments": {"code": "print(6 * 7)"}}\n</tool_call>',
        "<answer>42</answer>",
    ],
}


def _short(value: object, limit: int = 92) -> str:
    """把任意值压成一行短文本，方便打印轨迹。"""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _print_episode(index: int, total: int, result: object) -> None:
    """打印一条 episode 的轨迹摘要。"""
    trajectory = result.trajectory  # type: ignore[attr-defined]
    print(f"\n--- [{index}/{total}] {trajectory.task_id} ---")
    print(f"task: {_short(trajectory.prompt, 110)}")
    for turn in trajectory.turns:
        head = f"  turn {turn.index + 1:02d} | {turn.action_kind:<7}"
        if turn.action_kind == "tool":
            head += f" | {turn.tool_name}({_short(turn.tool_arguments, 60)})"
            head += f" -> {_short(turn.observation, 60)}"
            if turn.observation_ok is False:
                head += f"  [error={turn.error_code}]"
        elif turn.action_kind == "answer":
            head += f" | {_short(turn.observation, 70)}"
        else:
            head += f" | error={turn.error_code} -> {_short(turn.observation, 60)}"
        print(head)
    print(
        f"  reward={result.reward:.2f} solved={trajectory.solved} "  # type: ignore[attr-defined]
        f"terminated_by={trajectory.terminated_by} turns={len(trajectory.turns)} "
        f"tool_calls={trajectory.tool_calls} "
        f"errors=(protocol={trajectory.protocol_errors}, tool={trajectory.tool_errors}) "
        f"time={trajectory.wall_time_s:.3f}s"
    )
    print(f"  final_answer: {trajectory.final_answer}")


def _run_sandbox_checks(sandbox: SubprocessSandbox) -> list[tuple[str, bool, str]]:
    """直接对沙箱做边界检查（超时/截断/退出码/UTF-8）。"""
    checks: list[tuple[str, bool, str]] = []

    normal = sandbox.run_python('print("child says", 6 * 7)')
    checks.append(
        (
            "normal stdout",
            normal.ok and "42" in normal.stdout,
            _short(normal.stdout),
        )
    )

    unicode_run = sandbox.run_python('print("沙箱-中文-ok")')
    checks.append(
        (
            "utf-8 output",
            unicode_run.ok and "沙箱-中文-ok" in unicode_run.stdout,
            _short(unicode_run.stdout),
        )
    )

    timeout = sandbox.run_python("import time\ntime.sleep(5)", timeout_s=0.5)
    checks.append(
        (
            "timeout is killed",
            timeout.timed_out and not timeout.ok and timeout.latency_s < 4.0,
            f"timed_out={timeout.timed_out} latency={timeout.latency_s:.2f}s",
        )
    )

    flood = sandbox.run_python('print("x" * 20000)')
    checks.append(
        (
            "output truncated",
            flood.ok and flood.truncated and len(flood.stdout) <= sandbox.max_output_chars,
            f"truncated={flood.truncated} chars={len(flood.stdout)}",
        )
    )

    failing = sandbox.run_python('raise SystemExit(3)')
    checks.append(
        (
            "non-zero exit code",
            (not failing.ok) and failing.returncode == 3,
            f"ok={failing.ok} returncode={failing.returncode}",
        )
    )
    return checks


def main() -> int:
    """跑完整 demo，返回进程退出码。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("09-harness · Harness-RL minimal demo (no GPU / no API key)")
    print("=" * 72)
    print(f"workspace root: {PROJECT_DIR}")

    loop_config = LoopConfig(max_turns=8, max_protocol_errors=2, max_context_chars=12_000)
    harness = Harness(root=PROJECT_DIR, loop_config=loop_config)
    print(f"tools: {', '.join(harness.registry.names())}")
    print(
        f"loop: max_turns={loop_config.max_turns}, "
        f"max_protocol_errors={loop_config.max_protocol_errors}, "
        f"max_context_chars={loop_config.max_context_chars}"
    )

    print("\n" + "=" * 72)
    print("part 1/3 · single episodes (scripted fake LLM)")
    print("=" * 72)
    results = []
    for index, task in enumerate(TASKS, start=1):
        policy = ScriptedPolicy(SCRIPTS[task.task_id], name=f"scripted::{task.task_id}")
        result = harness.run_episode(task, policy)
        results.append(result)
        _print_episode(index, len(TASKS), result)

    print("\ntrajectory record (first episode, keys + counts):")
    record = results[0].trajectory.record()
    print(
        json.dumps(
            {
                "schema_version": record["schema_version"],
                "task_id": record["task_id"],
                "terminated_by": record["terminated_by"],
                "final_answer": record["final_answer"],
                "counts": record["counts"],
                "turns": [
                    {
                        "index": turn["index"],
                        "action_kind": turn["action_kind"],
                        "tool_name": turn["tool_name"],
                        "observation_ok": turn["observation_ok"],
                        "error_code": turn["error_code"],
                        "truncated": turn["truncated"],
                    }
                    for turn in record["turns"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n" + "=" * 72)
    print("part 2/3 · batch rollout (ThreadPoolExecutor)")
    print("=" * 72)
    batch_tasks: list[Task] = []
    for repeat in range(2):
        for task in TASKS:
            batch_tasks.append(
                Task(
                    task_id=f"{task.task_id}-r{repeat}",
                    prompt=task.prompt,
                    expected_answer=task.expected_answer,
                )
            )

    def policy_factory(task: Task) -> ScriptedPolicy:
        base_id = task.task_id.rsplit("-r", 1)[0]
        return ScriptedPolicy(SCRIPTS[base_id], name=f"scripted::{base_id}")

    batch_started = time.perf_counter()
    batch_results = harness.run_batch(batch_tasks, policy_factory, max_workers=4)
    batch_wall = time.perf_counter() - batch_started
    stats = harness.summarize(batch_results)
    print(f"episodes={len(batch_results)} batch_wall_time={batch_wall:.3f}s")
    print(stats.format_table())
    print("swanlab-style metrics:")
    print(json.dumps(stats.metrics(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print("part 3/3 · sandbox boundary checks")
    print("=" * 72)
    checks = _run_sandbox_checks(harness.sandbox)
    failed = 0
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        failed += 0 if passed else 1
        print(f"[{status}] {name:<22} {detail}")
    print("\nsandbox stats:", json.dumps(harness.sandbox.stats.metrics(), ensure_ascii=False))

    answered = sum(1 for result in results if result.trajectory.solved)
    if answered != len(results):
        failed += 1
        print(f"[FAIL] only {answered}/{len(results)} single episodes solved")

    print("\n" + "=" * 72)
    if failed:
        print(f"demo finished with {failed} failing check(s)")
        return 1
    print("demo finished: all checks passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
