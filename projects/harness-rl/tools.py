"""09-harness 的零依赖工具集与工具注册表。

工具层要回答的问题只有两个：

1. **有哪些工具**：名字、说明、参数 schema（模型侧）；
2. **怎么执行**：参数校验、异常归一化、错误码（harness 侧）。

本文件提供 5 个工具，覆盖 agent 轨迹里最常见的几类交互：

====================  ==========================================  ============
工具                  作用                                        执行方式
====================  ==========================================  ============
``calculator``        安全的四则运算（AST 白名单，不 eval）        进程内
``echo``              原样返回，用来测协议/测截断                  进程内
``search_docs``       本地小语料检索（词面打分），模拟搜索后端      进程内
``read_file``         受限文件读取（只能读 workspace/ 内）          进程内
``run_python``        子进程沙箱里执行 Python，带超时与输出截断     子进程
====================  ==========================================  ============

约定：工具 handler 只接受 **已通过 schema 校验** 的参数字典，
返回非空字符串作为 observation；预期内的失败抛 :class:`ToolError`，
意外异常由 :meth:`Tool.invoke` 统一包成 ``tool_internal_error``。
这样 agent loop 永远拿得到结构化的 :class:`~protocol.Observation`。
"""

from __future__ import annotations

import ast
import datetime as dt
import operator
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from protocol import ProtocolError, ToolSpec, validate_arguments

if TYPE_CHECKING:  # 只为类型标注导入，运行时保持 tools 与 sandbox 解耦
    from sandbox import SubprocessSandbox

# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #


class ToolError(Exception):
    """工具执行失败。

    :param code: 错误码，会写进 ``Observation.error_code``，用于统计 bad case；
    :param message: 原样回喂给模型的错误说明；
    :param timed_out: 是否超时失败（沙箱工具用）。
    """

    def __init__(self, code: str, message: str, *, timed_out: bool = False) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.timed_out = timed_out


class ToolNotFoundError(KeyError):
    """模型调用了一个不存在的工具。"""


# --------------------------------------------------------------------------- #
# Tool / ToolRegistry
# --------------------------------------------------------------------------- #

ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    """工具 = 静态描述（给模型看） + 执行函数（给 harness 用）。"""

    spec: ToolSpec
    handler: ToolHandler
    sandboxed: bool = False  # 是否在子进程沙箱里执行（用于统计/README 说明）

    @property
    def name(self) -> str:
        return self.spec.name

    def invoke(self, arguments: dict[str, Any]) -> str:
        """校验参数并执行工具，把异常归一化成 :class:`ToolError`。

        :raises ProtocolError: 参数不合法（缺少必填项 / 类型错误 / 未知参数）；
        :raises ToolError: 工具执行失败。
        """
        checked = validate_arguments(self.spec, arguments)
        try:
            return self.handler(checked)
        except ToolError:
            raise
        except ProtocolError:
            raise
        except Exception as exc:  # noqa: BLE001 - 工具内部任何异常都要变成可回喂的错误
            raise ToolError(
                "tool_internal_error",
                f"{self.name} raised {type(exc).__name__}: {exc}",
            ) from exc


class ToolRegistry:
    """工具注册表：名字到 :class:`Tool` 的映射，负责暴露 schema 与分发调用。"""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """注册一个工具；重名直接报错，避免静默覆盖。"""
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """按名字取工具，不存在抛 :class:`ToolNotFoundError`。"""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterable[Tool]:
        return iter(self._tools.values())

    @property
    def specs(self) -> list[ToolSpec]:
        """按注册顺序返回全部工具描述，用于拼系统提示词。"""
        return [tool.spec for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


# --------------------------------------------------------------------------- #
# calculator：AST 白名单，绝不 eval
# --------------------------------------------------------------------------- #

_MAX_EXPRESSION_CHARS = 200
_MAX_ABS_RESULT = 1e18
_MAX_POW_EXPONENT = 64

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _guard_result(value: Any) -> int | float:
    """结果必须是有限数值且量级可控，防止 ``9**9**9`` 这类表达式打崩对象。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError("calc_unsupported", "expression result is not a number")
    if abs(value) > _MAX_ABS_RESULT:
        raise ToolError("calc_overflow", f"result magnitude exceeds {_MAX_ABS_RESULT:.0e}")
    return value


def _eval_node(node: ast.AST) -> int | float:
    """递归求值一个已经过白名单检查的 AST 节点。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("calc_unsupported", "only numeric literals are allowed")
        return _guard_result(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, (int, float)) or abs(right) > _MAX_POW_EXPONENT:
                raise ToolError(
                    "calc_unsafe",
                    f"exponent must have magnitude <= {_MAX_POW_EXPONENT}",
                )
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("calc_division_by_zero", "division by zero")
        try:
            value = _BINARY_OPS[type(node.op)](left, right)
        except OverflowError as exc:
            raise ToolError("calc_overflow", "numeric overflow") from exc
        return _guard_result(value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _guard_result(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    raise ToolError(
        "calc_unsupported",
        "only + - * / // % ** with parentheses and numeric literals are supported",
    )


def _format_number(value: int | float) -> str:
    """把计算结果格式化成模型好读的字符串（整数不显示 .0）。"""
    if isinstance(value, int):
        return str(value)
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))
    return f"{value:.10g}"


def calculate(expression: str) -> str:
    """计算一个算术表达式，返回字符串结果。

    使用 ``ast.parse(mode="eval")`` + 节点白名单，而不是 ``eval()``：
    模型生成的代码永远不能碰属性、函数调用、下标、推导式等语法。
    """
    expression = expression.strip()
    if not expression:
        raise ToolError("calc_empty", "expression is empty")
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ToolError(
            "calc_expression_too_long",
            f"expression exceeds {_MAX_EXPRESSION_CHARS} characters",
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError("calc_syntax_error", f"invalid expression: {exc.msg}") from exc
    return _format_number(_eval_node(tree))


# --------------------------------------------------------------------------- #
# echo：最小工具，用来测协议/截断
# --------------------------------------------------------------------------- #


def echo(text: str) -> str:
    """原样返回输入文本。"""
    return text


# --------------------------------------------------------------------------- #
# search_docs：本地小语料检索
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Doc:
    """一条本地语料。"""

    doc_id: str
    title: str
    text: str


# 语料故意用「harness 组件」本身来写，demo 里检索出来的就是本仓库在讲的东西。
DOCS: tuple[Doc, ...] = (
    Doc(
        "doc-1",
        "Sandbox",
        "sandbox isolates tool execution in a subprocess with timeout, output truncation "
        "and return-code handling. Never run model-generated code in the trainer process.",
    ),
    Doc(
        "doc-2",
        "Protocol",
        "protocol defines tool schemas, parses <tool_call> from model output, validates "
        "arguments and renders observations back into the context.",
    ),
    Doc(
        "doc-3",
        "Agent loop",
        "agent loop runs the ReAct cycle: generate, parse, execute, append observation. "
        "It enforces max turns, context trimming, truncation and protocol error budgets.",
    ),
    Doc(
        "doc-4",
        "Trajectory",
        "trajectory records every turn with assistant text, parsed action, observation, "
        "error code, latency and termination reason so a rollout can be replayed and scored.",
    ),
    Doc(
        "doc-5",
        "Harness",
        "harness wires protocol, tools, sandbox and loop into a reusable runner with "
        "run_episode, concurrent run_batch and aggregate metrics for training dashboards.",
    ),
    Doc(
        "doc-6",
        "Reward and evaluation",
        "reward maps a finished trajectory to a scalar. keep tool observations out of the "
        "loss mask, and only score the model-generated answer tokens.",
    ),
)

_MAX_SEARCH_TOP_K = 5


def _tokenize(text: str) -> list[str]:
    """极简分词：英文数字按 \\w+ 切，中文按字切（够用了，零依赖）。"""
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    tokens.extend(re.findall(r"[\u4e00-\u9fff]", lowered))
    return tokens


def search_docs(query: str, top_k: int = 3) -> str:
    """在本地语料里做词面检索，返回打分最高的文档片段。

    这是有意的「玩具检索后端」：真实项目里接的是 BM25 / 向量检索 / 在线搜索 API，
    但 harness 只关心「工具调用 → 文本 observation」这个接口。
    """
    query = query.strip()
    if not query:
        raise ToolError("search_empty_query", "query is empty")
    top_k = max(1, min(int(top_k), _MAX_SEARCH_TOP_K))
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        raise ToolError("search_empty_query", "query has no searchable tokens")

    scored: list[tuple[float, Doc]] = []
    for doc in DOCS:
        title_tokens = _tokenize(doc.title)
        text_tokens = _tokenize(doc.text)
        score = 3.0 * len(query_tokens & set(title_tokens)) + 1.0 * len(
            query_tokens & set(text_tokens)
        )
        scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1].doc_id))
    matched = [(score, doc) for score, doc in scored if score > 0][:top_k]
    if not matched:
        return f"No documents matched query: {query!r}"
    blocks = []
    for score, doc in matched:
        snippet = doc.text if len(doc.text) <= 240 else doc.text[:240] + "..."
        blocks.append(f"[{doc.doc_id}] {doc.title} (score={score:g})\n{snippet}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# read_file：受限文件读取
# --------------------------------------------------------------------------- #

_DEFAULT_READ_CHARS = 1200
_MAX_READ_CHARS = 4000


def read_file(root: Path, path: str, max_chars: int = _DEFAULT_READ_CHARS) -> str:
    """读取 ``root`` 目录内的文本文件。

    安全规则（缺一不可）：

    1. 拒绝绝对路径；
    2. 拒绝含 ``..`` 的路径；
    3. ``resolve()`` 之后再校验仍在 ``root`` 内（防止符号链接逃逸）；
    4. 只读普通文件，最多返回 ``max_chars`` 个字符。
    """
    candidate = Path(path)
    if candidate.is_absolute():
        raise ToolError("path_escape", "absolute paths are not allowed")
    if ".." in candidate.parts:
        raise ToolError("path_escape", "'..' is not allowed in path")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not _is_relative_to(resolved, resolved_root):
        raise ToolError("path_escape", "path escapes the workspace root")
    if not resolved.exists():
        raise ToolError("file_not_found", f"{path!r} does not exist")
    if not resolved.is_file():
        raise ToolError("not_a_file", f"{path!r} is not a regular file")
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError("read_error", f"failed to read {path!r}: {exc}") from exc
    max_chars = max(1, min(int(max_chars), _MAX_READ_CHARS))
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        return text[:max_chars] + f"\n[... {omitted} chars omitted ...]"
    return text


def _is_relative_to(path: Path, root: Path) -> bool:
    """``Path.is_relative_to`` 的兼容写法（仓库要求 3.13，这里保持显式）。"""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# now / run_python
# --------------------------------------------------------------------------- #


def now(zone: str = "local") -> str:
    """返回当前时间。``zone`` 支持 ``local`` / ``utc``。"""
    zone = zone.strip().lower()
    if zone == "local":
        current = dt.datetime.now().astimezone()
    elif zone == "utc":
        current = dt.datetime.now(dt.timezone.utc)
    else:
        raise ToolError("bad_timezone", "zone must be 'local' or 'utc'")
    return current.isoformat(timespec="seconds")


def _make_run_python_handler(sandbox: SubprocessSandbox) -> ToolHandler:
    """构造 ``run_python`` 的 handler：把模型代码交给子进程沙箱执行。"""

    def handler(arguments: dict[str, Any]) -> str:
        code = str(arguments["code"])
        # 不传 timeout_s 时沿用沙箱实例的默认超时（demo 里会把它调小方便展示）。
        timeout_s = float(arguments.get("timeout_s", sandbox.timeout_s))
        result = sandbox.run_python(code, timeout_s=timeout_s)
        if result.timed_out:
            raise ToolError(
                "sandbox_timeout",
                f"code exceeded {timeout_s:g}s and the process was killed",
                timed_out=True,
            )
        if not result.ok:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ToolError(
                "sandbox_error",
                detail or f"process exited with code {result.returncode}",
            )
        output = result.stdout
        if not output.strip():
            return "(no output)"
        if result.truncated:
            output += f"\n[... output truncated to {sandbox.max_output_chars} chars ...]"
        return output

    return handler


# --------------------------------------------------------------------------- #
# 工具描述与默认注册表
# --------------------------------------------------------------------------- #

CALCULATOR_SPEC = ToolSpec(
    name="calculator",
    description=(
        "Evaluate one arithmetic expression with + - * / // % ** and parentheses. "
        "Example: (12 + 30) * 2 / 2. Return the numeric result as text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. (12 + 30) * 2 / 2.",
            }
        },
        "required": ["expression"],
    },
)

ECHO_SPEC = ToolSpec(
    name="echo",
    description="Return the input text unchanged. Useful for protocol smoke tests.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Any text."}},
        "required": ["text"],
    },
)

SEARCH_DOCS_SPEC = ToolSpec(
    name="search_docs",
    description=(
        "Search a small local documentation corpus and return the top matching snippets. "
        "Use it to find the name or definition of a harness component."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords to search for."},
            "top_k": {
                "type": "integer",
                "description": "How many snippets to return, 1-5. Default 3.",
            },
        },
        "required": ["query"],
    },
)

READ_FILE_SPEC = ToolSpec(
    name="read_file",
    description=(
        "Read a UTF-8 text file inside the workspace directory (relative path only). "
        "Returns up to max_chars characters. Use it for local notes and docs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside the workspace, e.g. workspace/notes.md.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return, 1-4000. Default 1200.",
            },
        },
        "required": ["path"],
    },
)

NOW_SPEC = ToolSpec(
    name="now",
    description="Return the current timestamp as ISO-8601 text. zone is 'local' or 'utc'.",
    parameters={
        "type": "object",
        "properties": {
            "zone": {
                "type": "string",
                "description": "Time zone: 'local' or 'utc'. Default 'local'.",
            }
        },
        "required": [],
    },
)

RUN_PYTHON_SPEC = ToolSpec(
    name="run_python",
    description=(
        "Execute a short Python snippet in an isolated subprocess and return its stdout. "
        "The subprocess has a wall-clock timeout and output is truncated. "
        "Always use print() for the values you need."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute. Print what you need to see.",
            },
            "timeout_s": {
                "type": "number",
                "description": "Wall-clock timeout in seconds, 0.1-30. Default: the sandbox timeout.",
            },
        },
        "required": ["code"],
    },
)


def build_default_registry(
    root: str | Path,
    sandbox: SubprocessSandbox | None = None,
) -> ToolRegistry:
    """组装默认工具集。

    :param root: ``read_file`` 允许访问的工作区根目录（一般是 09-harness/）；
    :param sandbox: 传入沙箱时才会注册 ``run_python``（没有沙箱就不给模型执行代码的入口）。
    """
    root_path = Path(root).resolve()
    registry = ToolRegistry()
    registry.register(Tool(CALCULATOR_SPEC, lambda args: calculate(args["expression"])))
    registry.register(Tool(ECHO_SPEC, lambda args: echo(args["text"])))
    registry.register(
        Tool(
            SEARCH_DOCS_SPEC,
            lambda args: search_docs(args["query"], int(args.get("top_k", 3))),
        )
    )
    registry.register(
        Tool(
            READ_FILE_SPEC,
            lambda args: read_file(
                root_path,
                str(args["path"]),
                int(args.get("max_chars", _DEFAULT_READ_CHARS)),
            ),
        )
    )
    registry.register(Tool(NOW_SPEC, lambda args: now(str(args.get("zone", "local")))))
    if sandbox is not None:
        registry.register(
            Tool(
                RUN_PYTHON_SPEC,
                _make_run_python_handler(sandbox),
                sandboxed=True,
            )
        )
    return registry
