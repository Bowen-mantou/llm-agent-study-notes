# Harness-RL 快速启动

> 完整原理、8 个组件拆解与训练侧坑位见 [Harness-RL Blog](./readme.md)。

本篇**不需要 GPU、不需要 API key、不需要安装任何第三方运行时依赖**（只用 Python 标准库）。`.pytest` 跑测试时额外需要 `pytest`。

命令默认在 `09-harness/` 目录下执行；标了「仓库根目录」的除外。

## 0. 环境要求

```bash
python --version   # 建议 >= 3.10，本机验证用 3.13
python -m pytest --version   # 测试需要 pytest >= 8
```

如果环境里没有 pytest：

```bash
python -m pip install -r requirements.txt
```

## 1. 跑 demo（无 GPU / 无 API key）

```bash
cd 09-harness
python demo.py
```

或者从仓库根目录：

```bash
python 09-harness/demo.py
```

看到最后一行输出 `demo finished: all checks passed` 即通过，退出码为 0。demo 会依次展示：

1. 三个单条 episode 的完整 ReAct 轨迹（`calc-chain` / `docs-lookup` / `resilience`）；
2. 6 条任务的并发批跑与 `harness/*` 聚合指标；
3. 沙箱子进程的四项边界检查（超时 / 截断 / 非零退出码 / UTF-8 中文）。

## 2. 跑测试

```bash
cd 09-harness
python -m pytest tests -q
```

预期结果：

```text
108 passed
```

覆盖范围：协议解析失败、参数校验、工具注入、路径逃逸、子进程超时、输出截断、非零退出码、最大轮数、协议错误预算、policy 异常、上下文裁剪、轨迹 JSON 格式、批跑顺序与聚合指标、demo 端到端。

## 3. 接真模型（OpenAI 兼容接口）

harness 只依赖一个 policy 函数：`(messages, tools) -> str`。把脚本化 policy 换成一次 chat completion 即可：

```python
import json
from openai import OpenAI

client = OpenAI(base_url="https://api.example.com/v1", api_key="...")

def policy(messages, tools):
    response = client.chat.completions.create(
        model="your-model",
        messages=messages,
        tools=[spec.to_openai_schema() for spec in tools],
        temperature=1.0,
    )
    message = response.choices[0].message
    if message.tool_calls:
        call = message.tool_calls[0]
        payload = {"name": call.function.name, "arguments": json.loads(call.function.arguments)}
        return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
    return "<answer>" + (message.content or "") + "</answer>"
```

然后在 demo 里把 `ScriptedPolicy(...)` 换成 `policy` 即可；沙箱、轨迹、reward、指标都不需要改。

> 训练场景不要用这段：它每次都重渲染完整历史再采样，属于 text-in text-out；
> 真训练要 token-in token-out，见 readme 的「下一步：接 PyTRIO 做 rollout」。

## 4. 最小用法示例

```python
from harness import Harness, Task
from demo import ScriptedPolicy

harness = Harness()
task = Task("toy", "Compute (12 + 30) * 2 / 2.", expected_answer="42")
policy = ScriptedPolicy([
    '<tool_call>\n{"name": "calculator", "arguments": {"expression": "(12 + 30) * 2"}}\n</tool_call>',
    '<tool_call>\n{"name": "calculator", "arguments": {"expression": "84 / 2"}}\n</tool_call>',
    "<answer>42</answer>",
])
result = harness.run_episode(task, policy)
print(result.reward, result.trajectory.record()["counts"])
```

## 5. 目录速查

| 路径 | 说明 |
| --- | --- |
| `protocol.py` | 工具协议解析 / 校验 / observation 渲染 |
| `tools.py` | 工具注册表与 5 个零依赖工具 |
| `sandbox.py` | 子进程沙箱（超时 / 截断 / 退出码） |
| `agent_loop.py` | ReAct 循环与轨迹记录 |
| `harness.py` | `run_episode` / `run_batch` / reward / metrics |
| `demo.py` | 无外部依赖的 demo |
| `tests/` | 108 个 pytest 用例 |
| `workspace/notes.md` | `read_file` 工具的受限工作区示例 |
