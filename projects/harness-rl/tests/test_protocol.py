"""protocol.py 的单元测试：解析边界 + 参数校验 + 截断/无害化。"""

from __future__ import annotations

import json

import pytest

from protocol import (
    ERR_ARGUMENTS_NOT_OBJECT,
    ERR_BAD_ARGUMENT_TYPE,
    ERR_EMPTY_ANSWER,
    ERR_EMPTY_OUTPUT,
    ERR_INVALID_JSON,
    ERR_MISSING_ARGUMENT,
    ERR_MISSING_ANSWER_TAG,
    ERR_MISSING_NAME,
    ERR_MULTIPLE_ANSWERS,
    ERR_MULTIPLE_TOOL_CALLS,
    ERR_TRAILING_TEXT,
    ERR_UNCLOSED_ANSWER,
    ERR_UNCLOSED_TOOL_CALL,
    ERR_UNKNOWN_ARGUMENT,
    Action,
    ProtocolError,
    ToolSpec,
    build_system_prompt,
    clip_text,
    parse_action,
    protocol_feedback,
    sanitize_control_markers,
    validate_arguments,
)

CALC_SPEC = ToolSpec(
    name="calculator",
    description="Evaluate arithmetic.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "precision": {"type": "integer"},
        },
        "required": ["expression"],
    },
)


def _call(name: str, arguments: dict, *, thought: str = "") -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{thought}\n<tool_call>\n{payload}\n</tool_call>"


# --------------------------------------------------------------------------- #
# 工具调用解析
# --------------------------------------------------------------------------- #


def test_parse_tool_call_with_thought() -> None:
    action = parse_action(_call("calculator", {"expression": "2 + 3 * 4"}, thought="Let me compute."))
    assert action.kind == "tool"
    assert action.call is not None
    assert action.call.name == "calculator"
    assert action.call.arguments == {"expression": "2 + 3 * 4"}
    assert action.content == "Let me compute."
    assert action.error_code is None


def test_parse_tool_call_tolerates_json_code_fence() -> None:
    text = (
        '<tool_call>\n```json\n{"name": "echo", "arguments": {"text": "hi"}}\n```\n</tool_call>'
    )
    action = parse_action(text)
    assert action.kind == "tool"
    assert action.call is not None
    assert action.call.arguments == {"text": "hi"}


def test_parse_tool_call_with_missing_arguments_defaults_to_empty_object() -> None:
    text = '<tool_call>\n{"name": "now"}\n</tool_call>'
    action = parse_action(text)
    assert action.kind == "tool"
    assert action.call is not None
    assert action.call.arguments == {}


def test_parse_invalid_json() -> None:
    action = parse_action('<tool_call>\n{"name": "echo", "arguments": {oops}}\n</tool_call>')
    assert action.kind == "invalid"
    assert action.error_code == ERR_INVALID_JSON


def test_parse_missing_name() -> None:
    action = parse_action('<tool_call>\n{"arguments": {"text": "hi"}}\n</tool_call>')
    assert action.kind == "invalid"
    assert action.error_code == ERR_MISSING_NAME


def test_parse_arguments_not_object() -> None:
    action = parse_action('<tool_call>\n{"name": "echo", "arguments": ["hi"]}\n</tool_call>')
    assert action.kind == "invalid"
    assert action.error_code == ERR_ARGUMENTS_NOT_OBJECT


def test_parse_multiple_tool_calls() -> None:
    text = _call("echo", {"text": "a"}) + "\n" + _call("echo", {"text": "b"})
    action = parse_action(text)
    assert action.kind == "invalid"
    assert action.error_code == ERR_MULTIPLE_TOOL_CALLS


def test_parse_unclosed_tool_call() -> None:
    action = parse_action('<tool_call>\n{"name": "echo", "arguments": {"text": "hi"}}')
    assert action.kind == "invalid"
    assert action.error_code == ERR_UNCLOSED_TOOL_CALL


def test_parse_trailing_text_after_tool_call() -> None:
    action = parse_action(_call("echo", {"text": "hi"}) + "\nand then I will answer.")
    assert action.kind == "invalid"
    assert action.error_code == ERR_TRAILING_TEXT


def test_parse_empty_output() -> None:
    action = parse_action("   \n  ")
    assert action.kind == "invalid"
    assert action.error_code == ERR_EMPTY_OUTPUT


# --------------------------------------------------------------------------- #
# 最终回答解析
# --------------------------------------------------------------------------- #


def test_parse_answer_tag_with_reasoning() -> None:
    action = parse_action("I checked it twice.\n<answer> 42 </answer>")
    assert action.kind == "answer"
    assert action.content == "42"


def test_parse_answer_tag_not_last() -> None:
    action = parse_action("<answer>42</answer>\nthanks")
    assert action.kind == "invalid"
    assert action.error_code == ERR_TRAILING_TEXT


def test_parse_implicit_answer_allowed_and_rejected() -> None:
    implicit = parse_action("42")
    assert implicit.kind == "answer"
    assert implicit.content == "42"

    rejected = parse_action("42", allow_implicit_answer=False)
    assert rejected.kind == "invalid"
    assert rejected.error_code == ERR_MISSING_ANSWER_TAG


def test_parse_unclosed_answer() -> None:
    action = parse_action("<answer>42")
    assert action.kind == "invalid"
    assert action.error_code == ERR_UNCLOSED_ANSWER


def test_parse_multiple_answers() -> None:
    action = parse_action("<answer>1</answer>\n<answer>2</answer>")
    assert action.kind == "invalid"
    assert action.error_code == ERR_MULTIPLE_ANSWERS


def test_parse_empty_answer() -> None:
    action = parse_action("<answer>   </answer>")
    assert action.kind == "invalid"
    assert action.error_code == ERR_EMPTY_ANSWER


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #


def test_validate_arguments_ok() -> None:
    checked = validate_arguments(CALC_SPEC, {"expression": "1+1", "precision": 2})
    assert checked == {"expression": "1+1", "precision": 2}


def test_validate_arguments_missing_required() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        validate_arguments(CALC_SPEC, {"precision": 2})
    assert excinfo.value.code == ERR_MISSING_ARGUMENT


def test_validate_arguments_unknown_argument() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        validate_arguments(CALC_SPEC, {"expression": "1+1", "syntax": "weird"})
    assert excinfo.value.code == ERR_UNKNOWN_ARGUMENT


def test_validate_arguments_bad_type() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        validate_arguments(CALC_SPEC, {"expression": 123})
    assert excinfo.value.code == ERR_BAD_ARGUMENT_TYPE


def test_validate_arguments_bool_is_not_integer() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        validate_arguments(CALC_SPEC, {"expression": "1+1", "precision": True})
    assert excinfo.value.code == ERR_BAD_ARGUMENT_TYPE


# --------------------------------------------------------------------------- #
# 文本工具函数 / 提示词
# --------------------------------------------------------------------------- #


def test_clip_text_short_text_unchanged() -> None:
    text, truncated = clip_text("hello", 100)
    assert text == "hello"
    assert truncated is False


def test_clip_text_long_text_keeps_head_and_tail() -> None:
    text = "A" * 500 + "TAIL"
    clipped, truncated = clip_text(text, 100)
    assert truncated is True
    assert "truncated" in clipped
    assert clipped.startswith("A" * 10)
    assert clipped.endswith("TAIL")


def test_clip_text_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError):
        clip_text("hello", 0)


def test_sanitize_control_markers() -> None:
    dirty = "<|im_start|>hi<|im_end|><tool_call>bad</tool_call>"
    clean = sanitize_control_markers(dirty)
    for marker in ("<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"):
        assert marker not in clean
    assert "hi" in clean


def test_tool_spec_signature_and_openai_schema() -> None:
    assert CALC_SPEC.signature() == "calculator(expression: string, precision: integer = optional)"
    schema = CALC_SPEC.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert schema["function"]["parameters"] is CALC_SPEC.parameters


def test_build_system_prompt_mentions_tools_and_answer_format() -> None:
    prompt = build_system_prompt([CALC_SPEC], allow_implicit_answer=False)
    assert "calculator(" in prompt
    assert "<tool_call>" in prompt
    assert "<answer>" in prompt
    assert "plain final answer" not in prompt


def test_protocol_feedback_contains_error_code_and_retry_rule() -> None:
    action = Action(kind="invalid", error_code=ERR_INVALID_JSON, error_message="bad json")
    feedback = protocol_feedback(action)
    assert ERR_INVALID_JSON in feedback
    assert "bad json" in feedback
    assert "<answer>" in feedback
