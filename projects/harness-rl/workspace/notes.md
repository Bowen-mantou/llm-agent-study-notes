# Harness-RL 工作区笔记

这个文件是 `09-harness` demo 里 `read_file` 工具的读取目标。
工作区根目录固定为 `09-harness/`，工具只能读这个目录内的相对路径。

## Harness 的 8 个组件

1. **Protocol**：工具 schema、`<tool_call>` 解析、参数校验、observation 渲染。
2. **Tools**：calculator / echo / search_docs / read_file / now / run_python。
3. **Sandbox**：负责隔离执行（子进程）、超时杀进程、输出截断、退出码处理。
4. **Agent loop**：ReAct 循环、最大轮数、协议错误预算、上下文裁剪。
5. **Trajectory**：每轮的 assistant 原文、解析结果、observation、错误码、耗时。
6. **Harness**：把上面几层组装成 `run_episode` / `run_batch` 的 rollout 入口。
7. **Reward**：把轨迹映射成标量；outcome-only，只看最终答案。
8. **Metrics**：solved_rate、reward、turns、tool_calls、timeouts、truncated 等聚合指标。

## 一句话总结

> sandbox 把模型生成的不可信输入挡在训练进程之外，trajectory 把每一步都记下来，
> harness 让同一条协议可以同时服务训练、评测和复盘。
