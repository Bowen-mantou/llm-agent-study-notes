# Agent Harness from Scratch（原创实现）

一个最小但完整的 agent harness：工具协议、沙箱、ReAct 循环、轨迹采集、批量并发、评测指标。

```
python demo.py          # 无 API key 可跑
python -m pytest tests -q
```

包含：
- `protocol.py`：工具 schema / 调用解析 / 参数校验 / 错误码
- `tools.py`：calculator / echo / 本地检索 / 受限文件读取 / 时间 / Python 沙箱
- `sandbox.py`：子进程隔离、超时杀进程树、输出截断、并发限制
- `agent_loop.py`：ReAct 循环、最大轮数、协议错误预算、上下文裁剪、轨迹
- `harness.py`：run_episode / run_batch / reward / 指标聚合
- `tests/`：108 个单测

> 本实现为原创；设计灵感来自 veRL AgentLoop、OpenHands runtime、SWE-agent ACI 等公开工作（见 CREDITS）。
