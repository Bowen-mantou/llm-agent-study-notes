"""harness.py 的单元测试：单条 rollout、并发批跑、reward 与聚合指标。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import (
    EpisodeResult,
    Harness,
    HarnessStats,
    LoopConfig,
    Task,
    exact_match_reward,
    summarize,
)
from sandbox import SubprocessSandbox


class ScriptedPolicy:
    """按顺序吐出预设文本的假 LLM。"""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self, messages: list[dict[str, object]], tools: list[object]) -> str:
        self.calls += 1
        return self.script[self.calls - 1] if self.calls <= len(self.script) else ""


def _answer(text: str) -> str:
    return f"<answer>{text}</answer>"


def _tool_call(name: str, arguments: dict) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    return Harness(
        root=tmp_path,
        loop_config=LoopConfig(max_turns=6),
        sandbox=SubprocessSandbox(timeout_s=2.0, max_workers=2),
    )


# --------------------------------------------------------------------------- #
# 单条 episode
# --------------------------------------------------------------------------- #


def test_run_episode_success(harness: Harness) -> None:
    task = Task("calc", "compute", expected_answer="42")
    result = harness.run_episode(
        task,
        ScriptedPolicy([_tool_call("calculator", {"expression": "6 * 7"}), _answer("42")]),
    )
    assert result.reward == 1.0
    assert result.trajectory.solved is True
    assert result.trajectory.tool_calls == 1
    assert result.task is task
    assert result.wall_time_s >= 0


def test_run_episode_wrong_answer(harness: Harness) -> None:
    task = Task("calc", "compute", expected_answer="42")
    result = harness.run_episode(task, ScriptedPolicy([_answer("41")]))
    assert result.reward == 0.0
    assert result.trajectory.solved is True  # 正常作答但答案错：solved 与 reward 是两件事


def test_run_episode_no_answer(harness: Harness) -> None:
    task = Task("calc", "compute", expected_answer="42")
    result = harness.run_episode(task, ScriptedPolicy([]))
    assert result.trajectory.solved is False
    assert result.reward == 0.0


def test_exact_match_reward_without_expected_answer(harness: Harness) -> None:
    task = Task("open", "just answer something")
    result = harness.run_episode(task, ScriptedPolicy([_answer("whatever")]))
    assert result.reward == 1.0

    failed = harness.run_episode(Task("open2", "loop"), ScriptedPolicy([]))
    assert failed.reward == 0.0


def test_exact_match_reward_normalizes_text() -> None:
    task = Task("t", "p", expected_answer="  Hello   World ")
    from agent_loop import Trajectory

    trajectory = Trajectory(task_id="t", prompt="p", solved=True, final_answer="hello world")

    assert exact_match_reward(task, trajectory) == 1.0


def test_episode_record_is_json_safe(harness: Harness) -> None:
    task = Task("calc", "compute", expected_answer="2")
    result = harness.run_episode(task, ScriptedPolicy([_answer("2")]))
    encoded = json.dumps(result.record(), ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["task_id"] == "calc"
    assert decoded["reward"] == 1.0
    assert decoded["trajectory"]["task_id"] == "calc"
    assert decoded["trajectory"]["counts"]["turns"] == 1


# --------------------------------------------------------------------------- #
# 并发批跑
# --------------------------------------------------------------------------- #


def test_run_batch_preserves_order_and_runs_each_task(harness: Harness) -> None:
    tasks = [
        Task("a", "prompt a", expected_answer="1"),
        Task("b", "prompt b", expected_answer="2"),
        Task("c", "prompt c", expected_answer="3"),
    ]

    def policy_factory(task: Task):
        return ScriptedPolicy([_answer(task.expected_answer or "")])

    results = harness.run_batch(tasks, policy_factory, max_workers=3)
    assert [result.task.task_id for result in results] == ["a", "b", "c"]
    assert all(result.reward == 1.0 for result in results)


def test_run_batch_empty(harness: Harness) -> None:
    results = harness.run_batch([], lambda task: ScriptedPolicy([]))
    assert results == []
    stats = summarize(results)
    assert stats.episodes == 0
    assert stats.metrics()["harness/solved_rate"] == 0.0


def test_run_batch_rejects_bad_worker_count(harness: Harness) -> None:
    with pytest.raises(ValueError):
        harness.run_batch([Task("a", "p")], lambda task: ScriptedPolicy([]), max_workers=0)


# --------------------------------------------------------------------------- #
# 聚合指标
# --------------------------------------------------------------------------- #


def test_harness_stats_from_results(harness: Harness) -> None:
    tasks = [Task("a", "prompt a", expected_answer="1"), Task("b", "prompt b", expected_answer="2")]

    def policy_factory(task: Task):
        if task.task_id == "a":
            return ScriptedPolicy([_tool_call("echo", {"text": "x"}), _answer("1")])
        return ScriptedPolicy([_answer("wrong")])

    results = harness.run_batch(tasks, policy_factory, max_workers=2)
    stats = HarnessStats.from_results(results)
    assert stats.episodes == 2
    assert stats.solved == 2
    assert stats.solved_rate == pytest.approx(1.0)
    assert stats.avg_reward == pytest.approx(0.5)
    assert stats.total_tool_calls == 1
    assert stats.avg_turns == pytest.approx((2 + 1) / 2)

    metrics = stats.metrics()
    assert metrics["harness/episodes"] == 2.0
    assert metrics["harness/reward"] == pytest.approx(0.5)
    assert metrics["harness/tool_calls"] == 1.0
    assert set(metrics) >= {"harness/solved_rate", "harness/timeouts", "harness/protocol_errors"}

    table = stats.format_table()
    assert "episodes" in table and "solved_rate" in table


def test_harness_stats_empty() -> None:
    stats = HarnessStats.from_results([])
    assert stats.episodes == 0
    assert stats.solved_rate == 0.0
    assert stats.format_table()


def test_harness_uses_default_reward_and_registry(tmp_path: Path) -> None:
    # 不传 sandbox/registry 时构造默认 harness；只验证可构造与默认 reward 绑定。
    harness = Harness(root=tmp_path)
    assert "calculator" in harness.registry
    assert harness.reward_fn is exact_match_reward
    assert harness.root == tmp_path.resolve()


def test_shared_loop_config(harness: Harness) -> None:
    assert harness.loop.config.max_turns == 6
