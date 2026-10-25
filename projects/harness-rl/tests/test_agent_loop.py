"""agent_loop.py 的单元测试：终止条件、错误恢复、截断、上下文裁剪、轨迹格式。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop import (
    TERMINATED_ANSWER,
    TERMINATED_MAX_TURNS,
    TERMINATED_POLICY_ERROR,
    TERMINATED_PROTOCOL_ERROR,
    AgentLoop,
    LoopConfig,
    Trajectory,
    trim_messages,
)
from sandbox import SubprocessSandbox
from tools import ECHO_SPEC, Tool, ToolRegistry, build_default_registry


class ScriptedPolicy:
    """按顺序吐出预设文本的假 LLM；脚本耗尽返回空串（触发 empty_output）。"""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self, messages: list[dict[str, object]], tools: list[object]) -> str:
        self.calls += 1
        if self.calls > len(self.script):
            return ""
        return self.script[self.calls - 1]


def _tool_call(name: str, arguments: dict) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}\n</tool_call>'


@pytest.fixture()
def registry(tmp_path: Path):
    sandbox = SubprocessSandbox(timeout_s=0.5, max_workers=2)
    return build_default_registry(tmp_path, sandbox)


# --------------------------------------------------------------------------- #
# 正常路径
# --------------------------------------------------------------------------- #


def test_tool_then_answer(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run(
        "t1",
        "compute",
        ScriptedPolicy(
            [
                _tool_call("calculator", {"expression": "(12 + 30) * 2"}),
                _tool_call("calculator", {"expression": "84 / 2"}),
                "<answer>42</answer>",
            ]
        ),
    )
    assert trajectory.solved is True
    assert trajectory.terminated_by == TERMINATED_ANSWER
    assert trajectory.final_answer == "42"
    assert trajectory.tool_calls == 2
    assert trajectory.tool_errors == 0
    assert len(trajectory.turns) == 3
    assert trajectory.turns[0].tool_name == "calculator"
    assert trajectory.turns[0].observation == "84"
    assert trajectory.turns[0].observation_ok is True


def test_implicit_answer(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run("t1", "say hi", ScriptedPolicy(["hi"]))
    assert trajectory.solved is True
    assert trajectory.final_answer == "hi"


# --------------------------------------------------------------------------- #
# 协议错误 / 工具错误
# --------------------------------------------------------------------------- #


def test_protocol_error_then_recover(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run(
        "t1",
        "compute",
        ScriptedPolicy(
            [
                "<tool_call>not valid json</tool_call>",
                "<answer>ok</answer>",
            ]
        ),
    )
    assert trajectory.solved is True
    assert trajectory.protocol_errors == 1
    assert trajectory.terminated_by == TERMINATED_ANSWER
    assert trajectory.turns[0].error_code == "invalid_json"
    # 非法输出后必须回喂一条纠错提示
    assert any("protocol-error" in str(message.get("content", "")) for message in trajectory.messages)


def test_protocol_error_budget_terminates(registry) -> None:
    loop = AgentLoop(registry, LoopConfig(max_protocol_errors=1))
    trajectory = loop.run(
        "t1",
        "compute",
        ScriptedPolicy(["<tool_call>bad json</tool_call>", "<answer>late</answer>"]),
    )
    assert trajectory.solved is False
    assert trajectory.terminated_by == TERMINATED_PROTOCOL_ERROR
    assert trajectory.final_answer is None
    assert len(trajectory.turns) == 1


def test_unknown_tool_is_recorded_and_recovered(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run(
        "t1",
        "compute",
        ScriptedPolicy([_tool_call("multiply", {"a": 6, "b": 7}), "<answer>42</answer>"]),
    )
    assert trajectory.solved is True
    assert trajectory.tool_errors == 1
    assert trajectory.turns[0].error_code == "unknown_tool"
    assert "unknown_tool" in (trajectory.turns[0].observation or "")


def test_bad_arguments_are_recorded(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run(
        "t1",
        "compute",
        ScriptedPolicy([_tool_call("calculator", {}), "<answer>42</answer>"]),
    )
    assert trajectory.solved is True
    assert trajectory.tool_errors == 1
    assert trajectory.turns[0].error_code == "missing_argument"


def test_tool_execution_error_is_recorded(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run(
        "t1",
        "compute",
        ScriptedPolicy(
            [
                _tool_call("calculator", {"expression": "1/0"}),
                "<answer>0</answer>",
            ]
        ),
    )
    assert trajectory.solved is True
    assert trajectory.tool_errors == 1
    assert trajectory.turns[0].error_code == "calc_division_by_zero"


def test_sandbox_timeout_is_recorded(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run(
        "t1",
        "run it",
        ScriptedPolicy(
            [
                _tool_call("run_python", {"code": "import time; time.sleep(5)"}),
                _tool_call("run_python", {"code": "print(6 * 7)"}),
                "<answer>42</answer>",
            ]
        ),
    )
    assert trajectory.solved is True
    assert trajectory.timeouts == 1
    assert trajectory.tool_errors == 1
    assert trajectory.turns[0].timed_out is True
    assert trajectory.turns[0].error_code == "sandbox_timeout"
    # Windows 子进程 stdout 末尾是换行，比较时忽略首尾空白
    assert trajectory.turns[1].observation.strip() == "42"


def test_observation_truncation(registry) -> None:
    loop = AgentLoop(registry, LoopConfig(max_observation_chars=100))
    trajectory = loop.run(
        "t1",
        "echo",
        ScriptedPolicy(
            [
                _tool_call("echo", {"text": "x" * 5000}),
                "<answer>done</answer>",
            ]
        ),
    )
    assert trajectory.truncated_observations == 1
    assert trajectory.turns[0].truncated is True
    observation = trajectory.turns[0].observation or ""
    assert "truncated" in observation
    assert len(observation) < 300


# --------------------------------------------------------------------------- #
# 终止条件
# --------------------------------------------------------------------------- #


def test_max_turns_terminates(registry) -> None:
    loop = AgentLoop(registry, LoopConfig(max_turns=2))
    trajectory = loop.run(
        "t1",
        "loop",
        ScriptedPolicy(
            [
                _tool_call("echo", {"text": "a"}),
                _tool_call("echo", {"text": "b"}),
                "<answer>never reached</answer>",
            ]
        ),
    )
    assert trajectory.solved is False
    assert trajectory.terminated_by == TERMINATED_MAX_TURNS
    assert len(trajectory.turns) == 2
    assert trajectory.tool_calls == 2


def test_policy_exception_is_contained(registry) -> None:
    def broken_policy(messages, tools):
        raise RuntimeError("sampler down")

    loop = AgentLoop(registry)
    trajectory = loop.run("t1", "compute", broken_policy)
    assert trajectory.terminated_by == TERMINATED_POLICY_ERROR
    assert trajectory.solved is False
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].error_code == TERMINATED_POLICY_ERROR


def test_empty_policy_output_is_protocol_error(registry) -> None:
    loop = AgentLoop(registry, LoopConfig(max_protocol_errors=1))
    trajectory = loop.run("t1", "compute", ScriptedPolicy([]))
    assert trajectory.terminated_by == TERMINATED_PROTOCOL_ERROR
    assert trajectory.turns[0].error_code == "empty_output"


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        LoopConfig(max_turns=0).validate()
    with pytest.raises(ValueError):
        LoopConfig(max_context_chars=0).validate()
    with pytest.raises(ValueError):
        LoopConfig(max_observation_chars=0).validate()
    with pytest.raises(ValueError):
        LoopConfig(max_protocol_errors=-1).validate()


# --------------------------------------------------------------------------- #
# 轨迹格式
# --------------------------------------------------------------------------- #


def test_trajectory_record_is_json_safe(registry) -> None:
    loop = AgentLoop(registry)
    trajectory = loop.run(
        "t1",
        "compute",
        ScriptedPolicy([_tool_call("echo", {"text": "hi"}), "<answer>hi</answer>"]),
    )
    record = trajectory.record()
    encoded = json.dumps(record, ensure_ascii=False)
    assert json.loads(encoded)["task_id"] == "t1"
    assert record["schema_version"] == 1
    assert record["counts"]["turns"] == 2
    assert record["counts"]["tool_calls"] == 1
    assert record["turns"][0]["tool_name"] == "echo"
    assert record["turns"][0]["observation"] == "hi"
    assert record["turns"][1]["action_kind"] == "answer"
    assert isinstance(record["messages"], list)


def test_trajectory_defaults() -> None:
    trajectory = Trajectory(task_id="x", prompt="p")
    assert trajectory.solved is False
    assert trajectory.record()["counts"]["turns"] == 0


# --------------------------------------------------------------------------- #
# 上下文裁剪
# --------------------------------------------------------------------------- #


def test_trim_messages_under_budget_is_unchanged() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    trimmed, was_trimmed = trim_messages(messages, 1000)
    assert trimmed == messages
    assert was_trimmed is False


def test_trim_messages_keeps_head_and_recent_turns() -> None:
    messages = [
        {"role": "system", "content": "S" * 10},
        {"role": "user", "content": "U" * 10},
        {"role": "assistant", "content": "A" * 100},
        {"role": "tool", "content": "T" * 100},
        {"role": "assistant", "content": "B" * 10},
        {"role": "tool", "content": "C" * 10},
    ]
    trimmed, was_trimmed = trim_messages(messages, 50)
    assert was_trimmed is True
    assert trimmed[0] is messages[0]
    assert trimmed[1] is messages[1]
    assert trimmed[2]["role"] == "system"
    assert "[context-trimmed]" in trimmed[2]["content"]
    # 保留下来的尾部不能以孤儿 tool 消息开头
    tail = trimmed[3:]
    assert tail and tail[0]["role"] != "tool"
    assert tail[-1]["content"] == "C" * 10


def test_trim_messages_drops_orphan_tool_messages() -> None:
    messages = [
        {"role": "system", "content": "S" * 10},
        {"role": "user", "content": "U" * 10},
        {"role": "assistant", "content": "A" * 100},
        {"role": "tool", "content": "T" * 10},
    ]
    trimmed, was_trimmed = trim_messages(messages, 25)
    assert was_trimmed is True
    assert all(message["role"] != "tool" for message in trimmed[2:])


def test_trim_messages_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError):
        trim_messages([{"role": "system", "content": "x"}], 0)


def test_loop_trims_context_and_records_it() -> None:
    # 用小注册表（只有 echo）让 system prompt 足够短，才能在小预算下触发裁剪。
    tiny_registry = ToolRegistry([Tool(ECHO_SPEC, lambda args: args["text"])])
    loop = AgentLoop(tiny_registry, LoopConfig(max_context_chars=900))
    script = [_tool_call("echo", {"text": "y" * 500}) for _ in range(3)]
    script.append("<answer>done</answer>")
    trajectory = loop.run("t1", "trim me", ScriptedPolicy(script))
    assert trajectory.solved is True
    assert trajectory.context_trims >= 1
    assert any("[context-trimmed]" in str(message.get("content", "")) for message in trajectory.messages)
