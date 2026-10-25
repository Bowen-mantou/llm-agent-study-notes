# LLM 学习笔记与手写实现教程

> CMU 11-768: AI Agents 与 freeCodeCamp “LLMs from Scratch” 的中文**文字版讲义**，加上大模型/ML 手写实现教程，以及一个从零实现的 agent harness。
> 全部为个人学习整理；**原文出处、引用与版权说明见 [CREDITS.md](CREDITS.md)**。

[English summary](#english-summary) · [仓库结构](#仓库结构) · [不包含什么](#不包含什么版权原因) · [使用方式](#使用方式) · [许可](#许可)

---

## 这是什么

- **课程讲义（文字版，不含视频截图）**
  - CMU 11-768: AI Agents（2026 Fall，Daniel Fried & Graham Neubig）
  - LLMs from Scratch: Practical Engineering from Base Model to PPO RLHF（freeCodeCamp）
- **大模型/ML 手写实现教程**：MHA、GRPO、KV cache、RoPE、LoRA、交叉熵、KMeans 等。
- **Agent Harness 从零实现**：工具协议、沙箱、ReAct 循环、轨迹采集、批量并发与评测，含 108 个单测。

## 仓库结构

```
courses/
  cmu-11-768-ai-agents/        # 14 讲讲义（文字版 PDF + LaTeX 源）
  pytorch-llm-from-scratch/    # 6 小时课程讲义（文字版 PDF + LaTeX 源）
handwritten/                   # 大模型/ML 手写实现教程（HTML）
projects/harness-rl/           # 从零实现 agent harness（代码 + 测试 + 教程）
```

## 不包含什么（版权原因）

- ❌ 视频截图/关键帧、B 站字幕与逐字稿、视频封面
- ❌ 第三方开源仓库代码、论文 PDF、课程录屏
- ❌ 工程复盘、面试资料、个人简历/投递/面试记录、公司调研、课程处理脚本等（本仓库只保留课程讲义、手写教程与 harness）

## 使用方式

- **读课程**：从 `courses/.../README.md` 开始；PDF 为文字版，配合原视频食用效果最好。
- **做 harness**：`projects/harness-rl/` 可直接 `python demo.py` / `pytest`（无需 GPU/API key）。
- **手写教程**：`handwritten/` 里的 HTML 用浏览器直接打开。

## 许可

- **代码 / harness**：MIT License（见 [LICENSE](LICENSE)）
- **讲义 / 文档 / 笔记**：CC BY-NC-SA 4.0（见 [LICENSE-docs](LICENSE-docs)）
- 第三方引用内容版权归原作者；详见 [NOTICE](NOTICE) 与 [CREDITS](CREDITS.md)。

## 致谢

- CMU 11-768 课程组（Daniel Fried, Graham Neubig）与 B 站搬运/翻译 UP「常青藤名校课」
- freeCodeCamp.org 与 B 站中文配音 UP「AIDeepCoder」
- 灵茶山艾府（灵神）的算法公开课（仅作为学习参考，未转载其内容）
- Agentic-RL-Lab（KMnO4-zx，Apache-2.0）与 veRL/OpenHands/SWE-agent 等公开工作（harness 设计灵感）

完整出处、链接与许可证见 [CREDITS.md](CREDITS.md)。若你是版权方并认为本仓库内容侵权，请开 Issue 联系，我们会及时下架相关内容。

---

## English summary

An open-source collection of **Chinese study notes and hand-written implementation tutorials** for LLMs. It contains text-only lecture notes for CMU 11-768 (AI Agents) and freeCodeCamp's “LLMs from Scratch”, hand-written tutorials for MHA/GRPO/KV-cache/RoPE/LoRA etc., and a minimal agent harness implemented from scratch. **Video frames, subtitles, transcripts, third-party repos, engineering/interview documents and personal files are intentionally excluded for copyright/privacy reasons.** See [CREDITS.md](CREDITS.md) for full attribution. Code is MIT-licensed; notes are CC BY-NC-SA 4.0.
