"""tools.py 的单元测试：安全计算、检索、受限文件读取、注册表与错误归一化。"""

from __future__ import annotations

from pathlib import Path

import pytest

from protocol import ProtocolError, ToolSpec
from sandbox import SubprocessSandbox
from tools import (
    DOCS,
    Tool,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    build_default_registry,
    calculate,
    echo,
    now,
    read_file,
    search_docs,
)

# --------------------------------------------------------------------------- #
# calculator：AST 白名单
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("(12 + 30) * 2", "84"),
        ("84 / 2", "42"),
        ("2 ** 10", "1024"),
        ("-3 + 1", "-2"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("1 / 2", "0.5"),
    ],
)
def test_calculate_arithmetic(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


def test_calculate_rejects_code_injection() -> None:
    for expression in ('__import__("os").system("echo pwn")', "open('x')", "[1, 2][0]"):
        with pytest.raises(ToolError) as excinfo:
            calculate(expression)
        assert excinfo.value.code == "calc_unsupported"


def test_calculate_rejects_division_by_zero() -> None:
    with pytest.raises(ToolError) as excinfo:
        calculate("1 / 0")
    assert excinfo.value.code == "calc_division_by_zero"


def test_calculate_rejects_huge_exponent() -> None:
    with pytest.raises(ToolError) as excinfo:
        calculate("2 ** 999")
    assert excinfo.value.code == "calc_unsafe"


def test_calculate_rejects_overflowing_result() -> None:
    with pytest.raises(ToolError) as excinfo:
        calculate("(10 ** 18) * (10 ** 18)")
    assert excinfo.value.code == "calc_overflow"


def test_calculate_rejects_empty_and_too_long_expression() -> None:
    with pytest.raises(ToolError) as empty:
        calculate("   ")
    assert empty.value.code == "calc_empty"
    with pytest.raises(ToolError) as too_long:
        calculate("1 + " * 100 + "1")
    assert too_long.value.code == "calc_expression_too_long"


def test_calculate_rejects_syntax_error() -> None:
    with pytest.raises(ToolError) as excinfo:
        calculate("1 + * 2")
    assert excinfo.value.code == "calc_syntax_error"


# --------------------------------------------------------------------------- #
# echo / search_docs / now
# --------------------------------------------------------------------------- #


def test_echo_roundtrip() -> None:
    assert echo("hello 世界") == "hello 世界"


def test_search_docs_returns_relevant_doc() -> None:
    output = search_docs("subprocess timeout isolation", top_k=2)
    assert "[doc-1]" in output
    assert "sandbox" in output.lower()


def test_search_docs_no_match() -> None:
    output = search_docs("quantum chromodynamics")
    assert output.startswith("No documents matched")


def test_search_docs_respects_top_k_and_validates_query() -> None:
    output = search_docs("harness component", top_k=1)
    assert output.count("[doc-") == 1
    with pytest.raises(ToolError) as excinfo:
        search_docs("   ")
    assert excinfo.value.code == "search_empty_query"


def test_corpus_is_nonempty_and_ids_unique() -> None:
    ids = [doc.doc_id for doc in DOCS]
    assert ids and len(ids) == len(set(ids))


def test_now_local_and_utc() -> None:
    local = now("local")
    utc = now("utc")
    assert "T" in local and "T" in utc
    assert utc.endswith("+00:00")
    with pytest.raises(ToolError) as excinfo:
        now("mars")
    assert excinfo.value.code == "bad_timezone"


# --------------------------------------------------------------------------- #
# read_file：路径逃逸 / 类型 / 截断
# --------------------------------------------------------------------------- #


def test_read_file_ok(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello harness", encoding="utf-8")
    assert read_file(tmp_path, "notes.md") == "hello harness"


def test_read_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        read_file(tmp_path, "nope.md")
    assert excinfo.value.code == "file_not_found"


def test_read_file_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ToolError) as excinfo:
        read_file(tmp_path, "../secret.txt")
    assert excinfo.value.code == "path_escape"


def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ToolError) as excinfo:
        read_file(tmp_path, str(target))
    assert excinfo.value.code == "path_escape"


def test_read_file_rejects_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    with pytest.raises(ToolError) as excinfo:
        read_file(tmp_path, "sub")
    assert excinfo.value.code == "not_a_file"


def test_read_file_truncates(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("abcdefghij", encoding="utf-8")
    output = read_file(tmp_path, "long.txt", max_chars=4)
    assert output.startswith("abcd")
    assert "omitted" in output


# --------------------------------------------------------------------------- #
# Tool / ToolRegistry
# --------------------------------------------------------------------------- #


def _echo_tool() -> Tool:
    spec = ToolSpec(
        name="echo",
        description="echo",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    return Tool(spec, lambda args: echo(args["text"]))


def test_registry_register_and_lookup() -> None:
    registry = ToolRegistry([_echo_tool()])
    assert "echo" in registry
    assert len(registry) == 1
    assert registry.get("echo").name == "echo"
    assert [spec.name for spec in registry.specs] == ["echo"]
    with pytest.raises(ToolNotFoundError):
        registry.get("nope")
    with pytest.raises(ValueError):
        registry.register(_echo_tool())


def test_tool_invoke_validates_arguments() -> None:
    registry = ToolRegistry([_echo_tool()])
    with pytest.raises(ProtocolError) as excinfo:
        registry.get("echo").invoke({})
    assert excinfo.value.code == "missing_argument"


def test_tool_invoke_wraps_unexpected_exception() -> None:
    spec = ToolSpec(
        name="broken",
        description="broken",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    tool = Tool(spec, lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ToolError) as excinfo:
        tool.invoke({})
    assert excinfo.value.code == "tool_internal_error"
    assert "boom" in excinfo.value.message


def test_tool_invoke_passes_tool_error_through() -> None:
    spec = ToolSpec(
        name="fails",
        description="fails",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    def handler(args: dict) -> str:
        raise ToolError("custom_error", "expected failure")

    with pytest.raises(ToolError) as excinfo:
        Tool(spec, handler).invoke({})
    assert excinfo.value.code == "custom_error"


# --------------------------------------------------------------------------- #
# 默认注册表
# --------------------------------------------------------------------------- #


def test_default_registry_without_sandbox_has_no_run_python(tmp_path: Path) -> None:
    registry = build_default_registry(tmp_path)
    assert registry.names() == ["calculator", "echo", "search_docs", "read_file", "now"]


def test_default_registry_with_sandbox_registers_run_python(tmp_path: Path) -> None:
    sandbox = SubprocessSandbox(timeout_s=5.0, max_workers=1)
    registry = build_default_registry(tmp_path, sandbox)
    assert "run_python" in registry
    tool = registry.get("run_python")
    assert tool.sandboxed is True
    output = tool.invoke({"code": "print(6 * 7)"})
    assert output.strip() == "42"


def test_run_python_tool_reports_nonzero_exit(tmp_path: Path) -> None:
    sandbox = SubprocessSandbox(timeout_s=5.0, max_workers=1)
    registry = build_default_registry(tmp_path, sandbox)
    with pytest.raises(ToolError) as excinfo:
        registry.get("run_python").invoke({"code": "raise SystemExit(3)"})
    assert excinfo.value.code == "sandbox_error"


def test_run_python_tool_reports_timeout(tmp_path: Path) -> None:
    sandbox = SubprocessSandbox(timeout_s=0.5, max_workers=1)
    registry = build_default_registry(tmp_path, sandbox)
    with pytest.raises(ToolError) as excinfo:
        registry.get("run_python").invoke({"code": "import time; time.sleep(5)"})
    assert excinfo.value.code == "sandbox_timeout"
    assert excinfo.value.timed_out is True
