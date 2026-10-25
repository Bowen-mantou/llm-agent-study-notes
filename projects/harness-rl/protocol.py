"""09-harness 的工具调用协议层（protocol）。

这一层刻意做得很薄，只负责三件事：

1. 用 :class:`ToolSpec` 描述一个工具（名字、说明、JSON Schema 参数）；
2. 把 LLM 的一段文本解析成「工具调用 / 最终回答 / 非法输出」；
3. 定义 :class:`Observation` 与渲染规则，把工具结果安全地回填进上下文。

协议格式（文本级，不依赖 tokenizer / chat template，任何模型都能输出）：

.. code-block:: text

    <tool_call>
    {"name": "calculator", "arguments": {"expression": "2 + 3 * 4"}}
    </tool_call>

    ...最终回答...
    <answer>42</answer>

设计原则：**解析失败不抛异常**。解析失败会返回 ``kind="invalid"`` 加错误码，
由 agent loop 决定是重试还是终止——训练时这些失败样本本身就是要被惩罚的负轨迹，
所以它们必须能被完整记录下来，而不是让 rollout 直接崩掉。

真实项目里这一层对应 OpenAI function calling / Qwen 原生 ``<tool_call>`` /
Anthropic tool_use 的公共抽象。本文件用最简单的 JSON 文本协议把语义表达清楚，
换成任何一家模型的格式时，只需要改 :func:`parse_action`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# 协议常量与错误码
# --------------------------------------------------------------------------- #

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# 错误码集中定义：日志、统计、回喂给模型的反馈都用同一套字符串，
# 方便训练时按错误类型切分 bad case。
ERR_EMPTY_OUTPUT = "empty_output"
ERR_INVALID_JSON = "invalid_json"
ERR_MISSING_NAME = "missing_name"
ERR_ARGUMENTS_NOT_OBJECT = "arguments_not_object"
ERR_MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
ERR_UNCLOSED_TOOL_CALL = "unclosed_tool_call"
ERR_TRAILING_TEXT = "trailing_text"
ERR_MULTIPLE_ANSWERS = "multiple_answers"
ERR_UNCLOSED_ANSWER = "unclosed_answer"
ERR_EMPTY_ANSWER = "empty_answer"
ERR_MISSING_ANSWER_TAG = "missing_answer_tag"
ERR_MISSING_ARGUMENT = "missing_argument"
ERR_UNKNOWN_ARGUMENT = "unknown_argument"
ERR_BAD_ARGUMENT_TYPE = "bad_argument_type"

# 工具输出里出现模板特殊标记时要做无害化，否则会污染上下文结构。
_CONTROL_MARKERS = {
    "<|im_start|>": "< im_start >",
    "<|im_end|>": "< im_end >",
    "<|endoftext|>": "< endoftext >",
    "<tool_response>": "< tool_response >",
    "</tool_response>": "< /tool_response>",
    "<tool_call>": "< tool_call >",
    "</tool_call>": "< /tool_call >",
}


class ProtocolError(ValueError):
    """协议解析或参数校验失败。

    :param code: 机器可读的错误码，会写进统计和回喂给模型的反馈里；
    :param message: 人类可读的说明，原样回喂给模型，帮助它自我纠错。
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# 工具描述
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的静态描述，对应 OpenAI function calling 里的 ``function`` 字段。

    :param name: 工具名，模型在 ``<tool_call>`` 里用它发起调用；
    :param description: 给模型看的说明，写清楚「什么时候用、返回什么」；
    :param parameters: JSON Schema 对象，至少包含 ``type`` / ``properties`` / ``required``。
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def signature(self) -> str:
        """返回一行函数签名，用于拼系统提示词。

        例如 ``calculator(expression: string)``；可选参数标成 ``= optional``。
        """
        properties: dict[str, Any] = self.parameters.get("properties") or {}
        required = set(self.parameters.get("required") or [])
        parts: list[str] = []
        for key, schema in properties.items():
            kind = str(schema.get("type", "any"))
            parts.append(f"{key}: {kind}" if key in required else f"{key}: {kind} = optional")
        return f"{self.name}({', '.join(parts)})"

    def to_openai_schema(self) -> dict[str, Any]:
        """转换成 OpenAI 兼容接口可直接传的 tools 列表元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --------------------------------------------------------------------------- #
# 解析结果
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolCall:
    """一次已解析的工具调用。"""

    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""  # 原始 JSON 文本，调试轨迹时用


@dataclass(frozen=True)
class Action:
    """assistant 一轮输出的解析结果。

    :param kind: ``"tool"`` / ``"answer"`` / ``"invalid"``；
    :param content: kind 为 ``answer`` 时是最终答案，为 ``tool`` 时是调用前的推理文本，
        为 ``invalid`` 时是原始输出（便于复盘 bad case）；
    :param call: kind 为 ``tool`` 时的结构化调用；
    :param error_code: kind 为 ``invalid`` 时的错误码；
    :param error_message: kind 为 ``invalid`` 时的可读说明。
    """

    kind: Literal["tool", "answer", "invalid"]
    content: str = ""
    call: ToolCall | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class Observation:
    """一次工具执行的结果，是回填进上下文的 observation。

    :param tool: 工具名；
    :param ok: 是否成功（进程退出码为 0、无参数错误、无超时）；
    :param content: 成功时是工具输出，失败时是错误说明；
    :param error_code: 失败时的错误码（协议错误 / 工具错误 / unknown_tool / timeout ...）；
    :param timed_out: 是否因超时被杀；
    :param truncated: 内容是否被截断（工具层或 loop 层截断都算）；
    :param latency_s: 工具执行耗时（秒）。
    """

    tool: str
    ok: bool
    content: str
    error_code: str | None = None
    timed_out: bool = False
    truncated: bool = False
    latency_s: float = 0.0

    def render(self) -> str:
        """把 observation 渲染成回填给模型的文本。

        成功直接返回内容；失败返回 ``Error [code]: message``。
        报错原样回喂是 Agentic RL 的关键——模型从错误里自我纠错的行为就是这么长出来的。
        """
        if self.ok:
            return self.content
        tag = f"Error [{self.error_code or 'tool_error'}]"
        if self.timed_out:
            tag += " (timeout)"
        return f"{tag}: {self.content}"


# --------------------------------------------------------------------------- #
# 文本工具函数
# --------------------------------------------------------------------------- #


def sanitize_control_markers(text: str) -> str:
    """把工具输出里的模板特殊标记替换成无害写法，防止污染 observation 结构。"""
    for marker, replacement in _CONTROL_MARKERS.items():
        text = text.replace(marker, replacement)
    return text


def clip_text(text: str, max_chars: int, *, head_ratio: float = 0.6) -> tuple[str, bool]:
    """按「头部 + 尾部」截断长文本，中间标注省略了多少字符。

    保留尾部的原因是：报错 traceback、计算结果通常都在最后几行；
    保留头部的原因是：文件/检索结果的开头通常最有信息量。
    返回文本的长度不会超过 ``max_chars``（``max_chars`` 小于省略标记本身时除外）。

    :return: ``(截断后的文本, 是否发生截断)``。
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    marker = f"\n[... truncated {omitted} chars ...]\n"
    budget = max(1, max_chars - len(marker))
    head_chars = max(1, int(budget * head_ratio))
    tail_chars = max(0, budget - head_chars)
    clipped = f"{text[:head_chars]}{marker}{text[-tail_chars:] if tail_chars else ''}"
    return clipped, True


# --------------------------------------------------------------------------- #
# 解析与校验
# --------------------------------------------------------------------------- #


def _invalid(code: str, message: str, raw: str) -> Action:
    """构造一个非法输出的解析结果。"""
    return Action(kind="invalid", content=raw, error_code=code, error_message=message)


def _strip_code_fence(payload: str) -> str:
    """容忍模型把 JSON 包在 markdown 代码块里的常见习惯。"""
    match = _CODE_FENCE_PATTERN.fullmatch(payload.strip())
    if match:
        return match.group(1).strip()
    return payload.strip()


def parse_action(text: str, *, allow_implicit_answer: bool = True) -> Action:
    """把一段 assistant 文本解析成工具调用、最终回答或非法输出。

    判定顺序：

    1. 出现 ``<tool_call>``：必须恰好一个、闭合并能解析出 ``{"name": str, "arguments": dict}``，
       且 ``</tool_call>`` 之后不允许多余文本；
    2. 出现 ``<answer>``：恰好一个、非空，之后不允许多余文本；
    3. 其余非空文本：``allow_implicit_answer=True`` 时视为隐式最终答案，否则判非法。

    :param text: 模型这一轮生成的完整文本；
    :param allow_implicit_answer: 是否接受不带 ``<answer>`` 标签的纯文本最终回答。
    """
    raw = (text or "").strip()
    if not raw:
        return _invalid(ERR_EMPTY_OUTPUT, "assistant output is empty", raw)

    tool_matches = list(TOOL_CALL_PATTERN.finditer(raw))
    if raw.count(TOOL_CALL_OPEN) > 1 or len(tool_matches) > 1:
        return _invalid(
            ERR_MULTIPLE_TOOL_CALLS,
            "exactly one <tool_call> is allowed per turn",
            raw,
        )
    if TOOL_CALL_OPEN in raw and not tool_matches:
        return _invalid(
            ERR_UNCLOSED_TOOL_CALL,
            "found <tool_call> without a matching </tool_call>",
            raw,
        )

    if len(tool_matches) == 1:
        match = tool_matches[0]
        trailing = raw[match.end() :].strip()
        if trailing:
            return _invalid(
                ERR_TRAILING_TEXT,
                "no text is allowed after </tool_call>; put reasoning before the call",
                raw,
            )
        payload = _strip_code_fence(match.group(1))
        if not payload:
            return _invalid(ERR_INVALID_JSON, "tool call payload is empty", raw)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            return _invalid(ERR_INVALID_JSON, f"tool call is not valid JSON: {exc.msg}", raw)
        if not isinstance(data, dict):
            return _invalid(ERR_INVALID_JSON, "tool call must be a JSON object", raw)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _invalid(ERR_MISSING_NAME, "tool call JSON needs a non-empty string 'name'", raw)
        arguments = data.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _invalid(ERR_ARGUMENTS_NOT_OBJECT, "'arguments' must be a JSON object", raw)
        return Action(
            kind="tool",
            content=raw[: match.start()].strip(),
            call=ToolCall(name=name.strip(), arguments=arguments, raw_arguments=payload),
        )

    if ANSWER_OPEN in raw:
        answer_matches = list(ANSWER_PATTERN.finditer(raw))
        if raw.count(ANSWER_OPEN) > 1 or len(answer_matches) > 1:
            return _invalid(ERR_MULTIPLE_ANSWERS, "exactly one <answer> is allowed", raw)
        if not answer_matches:
            return _invalid(
                ERR_UNCLOSED_ANSWER,
                "found <answer> without a matching </answer>",
                raw,
            )
        match = answer_matches[0]
        trailing = raw[match.end() :].strip()
        if trailing:
            return _invalid(ERR_TRAILING_TEXT, "no text is allowed after </answer>", raw)
        answer = match.group(1).strip()
        if not answer:
            return _invalid(ERR_EMPTY_ANSWER, "<answer> must not be empty", raw)
        return Action(kind="answer", content=answer)

    if not allow_implicit_answer:
        return _invalid(
            ERR_MISSING_ANSWER_TAG,
            "finish with <answer>...</answer>",
            raw,
        )
    return Action(kind="answer", content=raw)


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _matches_type(expected: str, value: Any) -> bool:
    """按 JSON Schema 的基础类型检查 Python 值（bool 不算 number/integer）。"""
    if expected == "number" or expected == "integer":
        if isinstance(value, bool):
            return False
    if expected == "boolean":
        return isinstance(value, bool)
    allowed = _TYPE_MAP.get(expected)
    if allowed is None:
        return True  # 未知类型不拦截，交给工具自己校验
    return isinstance(value, allowed)


def validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """按工具的 JSON Schema 校验参数，返回归一化后的参数副本。

    只做 MVP 必要校验：required 是否齐全、未知参数、基础类型。
    校验失败抛 :class:`ProtocolError`，由 agent loop 转成可回喂的错误 observation。
    """
    if not isinstance(arguments, dict):
        raise ProtocolError(ERR_ARGUMENTS_NOT_OBJECT, f"arguments for {spec.name} must be a JSON object")
    properties: dict[str, Any] = spec.parameters.get("properties") or {}
    required = list(spec.parameters.get("required") or [])

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ProtocolError(
            ERR_MISSING_ARGUMENT,
            f"{spec.name} is missing required argument(s): {', '.join(missing)}",
        )
    unknown = [name for name in arguments if name not in properties]
    if unknown:
        expected = ", ".join(properties) or "(none)"
        raise ProtocolError(
            ERR_UNKNOWN_ARGUMENT,
            f"{spec.name} got unknown argument(s): {', '.join(unknown)}; expected: {expected}",
        )
    for name, value in arguments.items():
        expected_type = properties[name].get("type")
        if expected_type and not _matches_type(expected_type, value):
            raise ProtocolError(
                ERR_BAD_ARGUMENT_TYPE,
                f"{spec.name}.{name} expects {expected_type}, got {type(value).__name__}",
            )
    return dict(arguments)


# --------------------------------------------------------------------------- #
# 提示词与反馈文本
# --------------------------------------------------------------------------- #


def build_system_prompt(
    specs: list[ToolSpec],
    *,
    allow_implicit_answer: bool = True,
) -> str:
    """根据工具列表拼出系统提示词。

    真实项目里这段提示词会决定模型的工具调用格式遵循率，
    所以规则要短、要明确、要把「一个 turn 只能调一次工具」写死。
    """
    tool_lines = "\n".join(f"- {spec.signature()}: {spec.description}" for spec in specs)
    finish_rule = "Finish with exactly: <answer>your final answer</answer>\n"
    if allow_implicit_answer:
        finish_rule += "A plain final answer without tags is also accepted when no tool is needed.\n"
    return (
        "You are a tool-using agent. Solve the task by calling tools, then finish with an answer.\n\n"
        "## Available tools\n"
        f"{tool_lines}\n\n"
        "## Tool call format (exactly one call per turn)\n"
        '<tool_call>\n{"name": "<tool_name>", "arguments": {<json arguments>}}\n</tool_call>\n\n'
        "Put private reasoning as plain text BEFORE the tool call. "
        "Do not write anything after </tool_call>; wait for the observation instead.\n\n"
        "## Finishing\n"
        f"{finish_rule}\n"
        "## Rules\n"
        "- Never invent tool results; only the observation is truth.\n"
        "- If a tool call fails, read the error and retry with corrected arguments.\n"
        "- Keep the final answer short and concrete.\n"
    )


def protocol_feedback(action: Action) -> str:
    """把非法解析结果转成回喂给模型的纠错提示。"""
    return (
        f"[protocol-error:{action.error_code}] {action.error_message}\n"
        'Retry: emit exactly one <tool_call>{"name": ..., "arguments": {...}}</tool_call> '
        "with valid JSON, or finish with <answer>...</answer>."
    )
