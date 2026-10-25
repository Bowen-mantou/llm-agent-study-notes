"""09-harness 的 ReAct agent loop。

一次 episode 就是一个状态机：

.. code-block:: text

    build messages
        │
        ▼
    policy(messages) ──► parse_action
        │                  │
        │        tool ─────┼──► execute tool ──► append observation ──┐
        │        answer ───┴──► finish                                  │
        │        invalid ──► append protocol feedback ──────────────────┘
        ▼
    trajectory（每条路径都记录终止原因）

这一层负责四件在 Agent RL 里非常容易写错的事：

1. **轮数控制**：``max_turns`` 到顶必须终止，否则单条 rollout 可能无限循环烧 token；
2. **协议错误预算**：连续输出非法格式超过阈值就终止，避免模型一直发同一种坏格式；
3. **上下文裁剪**：多轮工具轨迹会迅速变长，超过字符预算时丢最早的中间轮，
   但保留 system prompt 和任务描述（训练时这些 token 的 loss mask 也要保留）；
4. **轨迹记录**：每轮 assistant 原文、解析结果、observation、错误码、耗时、
   终止原因全部落进 :class:`Trajectory`，可 JSON 序列化，供 reward 打分和复盘。

policy 抽象成可调用对象（:class:`Policy`），不依赖任何模型 SDK：
demo 用脚本化假 LLM；真实项目里换成 PyTRIO/OpenAI 兼容接口的采样函数即可。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from protocol import (
    Action,
    Observation,
    ProtocolError,
    ToolSpec,
    build_system_prompt,
    clip_text,
    parse_action,
    protocol_feedback,
    sanitize_control_markers,
)
from tools import ToolError, ToolNotFoundError, ToolRegistry

# 终止原因：训练统计里按这个字段切分 bad case。
TERMINATED_ANSWER = "answer"
TERMINATED_MAX_TURNS = "max_turns"
TERMINATED_PROTOCOL_ERROR = "protocol_error"
TERMINATED_POLICY_ERROR = "policy_error"


class Policy(Protocol):
    """LLM 采样接口：输入对话历史与工具描述，返回这一轮 assistant 文本。"""

    def __call__(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> str: ...


@dataclass
class LoopConfig:
    """agent loop 的硬约束。

    :param max_turns: 单条轨迹最多允许的 assistant 轮数（含非法输出轮）；
    :param max_protocol_errors: 协议错误累计到多少轮就终止（``0`` 表示第一轮失败就终止）；
    :param max_context_chars: 上下文总字符预算，超出后裁剪最早的中间轮；
    :param max_observation_chars: 单条 observation 的字符上限，超出截断并标记；
    :param allow_implicit_answer: 是否接受不带 ``<answer>`` 标签的纯文本回答。
    """

    max_turns: int = 8
    max_protocol_errors: int = 2
    max_context_chars: int = 12_000
    max_observation_chars: int = 2_000
    allow_implicit_answer: bool = True

    def validate(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if self.max_observation_chars <= 0:
            raise ValueError("max_observation_chars must be positive")
        if self.max_protocol_errors < 0:
            raise ValueError("max_protocol_errors must be >= 0")


@dataclass(frozen=True)
class TurnRecord:
    """一轮 assistant 的完整记录。"""

    index: int
    assistant: str
    action_kind: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    observation: str | None = None
    observation_ok: bool | None = None
    error_code: str | None = None
    timed_out: bool = False
    truncated: bool = False
    latency_s: float = 0.0


@dataclass
class Trajectory:
    """一条完整 episode 的轨迹。

    这是 harness 最重要的输出：reward 打分、bad case 分析、SFT 数据导出、
    离线回放都从它出发。字段刻意保留「原始文本 + 结构化解析」两份，
    因为训练需要原始文本，分析需要结构化字段。
    """

    task_id: str
    prompt: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    terminated_by: str = TERMINATED_MAX_TURNS
    final_answer: str | None = None
    solved: bool = False
    protocol_errors: int = 0
    tool_errors: int = 0
    tool_calls: int = 0
    timeouts: int = 0
    truncated_observations: int = 0
    context_trims: int = 0
    wall_time_s: float = 0.0

    def record(self) -> dict[str, Any]:
        """导出 JSON 安全的轨迹字典（demo / 单测 / 落盘都用这个）。"""
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "terminated_by": self.terminated_by,
            "final_answer": self.final_answer,
            "solved": self.solved,
            "counts": {
                "turns": len(self.turns),
                "tool_calls": self.tool_calls,
                "protocol_errors": self.protocol_errors,
                "tool_errors": self.tool_errors,
                "timeouts": self.timeouts,
                "truncated_observations": self.truncated_observations,
                "context_trims": self.context_trims,
            },
            "wall_time_s": round(self.wall_time_s, 4),
            "messages": self.messages,
            "turns": [
                {
                    "index": turn.index,
                    "assistant": turn.assistant,
                    "action_kind": turn.action_kind,
                    "tool_name": turn.tool_name,
                    "tool_arguments": turn.tool_arguments,
                    "observation": turn.observation,
                    "observation_ok": turn.observation_ok,
                    "error_code": turn.error_code,
                    "timed_out": turn.timed_out,
                    "truncated": turn.truncated,
                    "latency_s": round(turn.latency_s, 4),
                }
                for turn in self.turns
            ],
        }


def _content_size(message: dict[str, Any]) -> int:
    """估算一条消息的字符数（MVP 用字符预算，真实训练按 token 算）。"""
    return len(str(message.get("content", "")))


def trim_messages(
    messages: Sequence[dict[str, Any]],
    max_chars: int,
) -> tuple[list[dict[str, Any]], bool]:
    """按字符预算裁剪上下文，保留 system prompt、任务描述和最近几轮。

    裁剪策略：从头丢中间轮，尾部优先保留；如果保留下来的第一条是 ``tool``
    （它对应的 assistant 调用已经被丢掉），就把这条孤儿 observation 也丢掉，
    否则会拼出非法的对话结构。

    :return: ``(新消息列表, 是否发生裁剪)``；函数不修改入参。
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    total = sum(_content_size(message) for message in messages)
    if total <= max_chars:
        return list(messages), False
    if len(messages) <= 2:
        return list(messages), False

    head = list(messages[:2])
    tail = list(messages[2:])
    used = sum(_content_size(message) for message in head)
    keep: list[dict[str, Any]] = []
    for message in reversed(tail):
        size = _content_size(message)
        if used + size > max_chars and keep:
            break
        keep.append(message)
        used += size
    keep.reverse()

    dropped = len(tail) - len(keep)
    while keep and keep[0].get("role") == "tool":
        keep.pop(0)
        dropped += 1
    if dropped == 0:
        return list(messages), False

    note = {
        "role": "system",
        "content": (
            f"[context-trimmed] dropped {dropped} earlier message(s) "
            f"to fit the {max_chars}-char budget."
        ),
    }
    return [*head, note, *keep], True


class AgentLoop:
    """把 policy、工具注册表和约束组装成一次 ReAct rollout。"""

    def __init__(self, registry: ToolRegistry, config: LoopConfig | None = None) -> None:
        self.registry = registry
        self.config = config or LoopConfig()
        self.config.validate()

    # ------------------------------------------------------------------ #
    # 单轮工具执行
    # ------------------------------------------------------------------ #

    def _execute(self, name: str, arguments: dict[str, Any]) -> Observation:
        """执行一次工具调用，把所有失败归一化成 Observation（绝不向 rollout 抛异常）。"""
        started = time.perf_counter()
        try:
            tool = self.registry.get(name)
        except ToolNotFoundError:
            return Observation(
                tool=name,
                ok=False,
                content=f"unknown tool {name!r}; available: {', '.join(self.registry.names())}",
                error_code="unknown_tool",
                latency_s=time.perf_counter() - started,
            )
        try:
            content = tool.invoke(arguments)
        except ProtocolError as exc:
            return Observation(
                tool=name,
                ok=False,
                content=exc.message,
                error_code=exc.code,
                latency_s=time.perf_counter() - started,
            )
        except ToolError as exc:
            return Observation(
                tool=name,
                ok=False,
                content=exc.message,
                error_code=exc.code,
                timed_out=exc.timed_out,
                latency_s=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001 - 工具实现有 bug 也不能崩 rollout
            return Observation(
                tool=name,
                ok=False,
                content=f"{type(exc).__name__}: {exc}",
                error_code="tool_internal_error",
                latency_s=time.perf_counter() - started,
            )

        cleaned = sanitize_control_markers(content)
        clipped, truncated = clip_text(cleaned, self.config.max_observation_chars)
        return Observation(
            tool=name,
            ok=True,
            content=clipped if clipped.strip() else "(empty output)",
            truncated=truncated,
            latency_s=time.perf_counter() - started,
        )

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #

    def run(self, task_id: str, prompt: str, policy: Policy) -> Trajectory:
        """跑一条完整轨迹。

        :param task_id: 任务 id，只用于轨迹标注；
        :param prompt: 任务描述，作为第一条 user 消息；
        :param policy: 采样函数，返回这一轮 assistant 文本。
        """
        started = time.perf_counter()
        config = self.config
        trajectory = Trajectory(task_id=task_id, prompt=prompt)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.registry.specs,
                    allow_implicit_answer=config.allow_implicit_answer,
                ),
            },
            {"role": "user", "content": prompt},
        ]

        terminated_by = TERMINATED_MAX_TURNS
        final_answer: str | None = None
        action: Action | None = None

        for turn_index in range(config.max_turns):
            messages, trimmed = trim_messages(messages, config.max_context_chars)
            if trimmed:
                trajectory.context_trims += 1

            turn_started = time.perf_counter()
            try:
                raw_assistant = policy(messages, self.registry.specs) or ""
            except Exception as exc:  # noqa: BLE001 - 采样失败（网络/额度）不崩 rollout
                trajectory.turns.append(
                    TurnRecord(
                        index=turn_index,
                        assistant="",
                        action_kind=TERMINATED_POLICY_ERROR,
                        error_code=TERMINATED_POLICY_ERROR,
                        observation=str(exc),
                        observation_ok=False,
                        latency_s=time.perf_counter() - turn_started,
                    )
                )
                terminated_by = TERMINATED_POLICY_ERROR
                break

            action = parse_action(
                raw_assistant,
                allow_implicit_answer=config.allow_implicit_answer,
            )
            messages.append({"role": "assistant", "content": raw_assistant})

            if action.kind == "answer":
                final_answer = action.content
                terminated_by = TERMINATED_ANSWER
                trajectory.turns.append(
                    TurnRecord(
                        index=turn_index,
                        assistant=raw_assistant,
                        action_kind=action.kind,
                        observation=action.content,
                        observation_ok=True,
                        latency_s=time.perf_counter() - turn_started,
                    )
                )
                break

            if action.kind == "invalid":
                trajectory.protocol_errors += 1
                feedback = protocol_feedback(action)
                messages.append({"role": "user", "content": feedback})
                trajectory.turns.append(
                    TurnRecord(
                        index=turn_index,
                        assistant=raw_assistant,
                        action_kind=action.kind,
                        observation=feedback,
                        observation_ok=False,
                        error_code=action.error_code,
                        latency_s=time.perf_counter() - turn_started,
                    )
                )
                if trajectory.protocol_errors >= config.max_protocol_errors:
                    terminated_by = TERMINATED_PROTOCOL_ERROR
                    break
                continue

            # action.kind == "tool"
            call = action.call
            if call is None:  # pragma: no cover - parse_action 的 tool 分支一定有 call
                raise AssertionError("tool action without a parsed call")
            trajectory.tool_calls += 1
            observation = self._execute(call.name, call.arguments)
            if not observation.ok:
                trajectory.tool_errors += 1
            if observation.timed_out:
                trajectory.timeouts += 1
            if observation.truncated:
                trajectory.truncated_observations += 1
            rendered = observation.render()
            messages.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "tool_call_id": f"call_{turn_index}",
                    "content": rendered,
                }
            )
            trajectory.turns.append(
                TurnRecord(
                    index=turn_index,
                    assistant=raw_assistant,
                    action_kind=action.kind,
                    tool_name=call.name,
                    tool_arguments=call.arguments,
                    observation=rendered,
                    observation_ok=observation.ok,
                    error_code=observation.error_code,
                    timed_out=observation.timed_out,
                    truncated=observation.truncated,
                    latency_s=time.perf_counter() - turn_started,
                )
            )

        trajectory.messages = messages
        trajectory.terminated_by = terminated_by
        trajectory.final_answer = final_answer
        trajectory.solved = terminated_by == TERMINATED_ANSWER and bool(
            final_answer and final_answer.strip()
        )
        trajectory.wall_time_s = time.perf_counter() - started
        return trajectory
