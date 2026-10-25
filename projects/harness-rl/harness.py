"""09-harness 的可复用 Harness：把协议、工具、沙箱、循环组装成一个 rollout 入口。

对训练侧来说，harness 的职责可以概括成一句话：

> **喂进来 (task, policy)，吐出去 (trajectory, reward, metrics)，中间不要崩。**

于是这个文件只暴露三个东西：

- :meth:`Harness.run_episode`：跑一条轨迹并算 reward；
- :meth:`Harness.run_batch`：按 ``max_workers`` 并发跑一批任务（rollout worker 的雏形）；
- :func:`summarize` / :meth:`HarnessStats.metrics`：聚合成训练日志要的比例指标。

轨迹格式见 :meth:`agent_loop.Trajectory.record`。真实项目里 reward 函数由任务定义
（数学题用答案等价判定、检索题用 recall/F1、SWE 用单测通过率），
这里提供最朴素的 exact-match，保证 demo 无外部依赖也能跑。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_loop import AgentLoop, LoopConfig, Policy, Trajectory
from sandbox import SubprocessSandbox
from tools import ToolRegistry, build_default_registry

RewardFn = Callable[["Task", Trajectory], float]


@dataclass(frozen=True)
class Task:
    """一条待 rollout 的任务。

    :param task_id: 唯一 id，轨迹与统计都按它对齐；
    :param prompt: 任务描述（第一条 user 消息）；
    :param expected_answer: 期望答案；为 ``None`` 时 reward 只区分「是否正常给出答案」。
    """

    task_id: str
    prompt: str
    expected_answer: str | None = None


@dataclass
class EpisodeResult:
    """一条 episode 的 rollout 结果。"""

    task: Task
    trajectory: Trajectory
    reward: float
    wall_time_s: float

    def record(self) -> dict[str, Any]:
        """导出 JSON 安全的结果字典。"""
        return {
            "task_id": self.task.task_id,
            "reward": self.reward,
            "wall_time_s": round(self.wall_time_s, 4),
            "trajectory": self.trajectory.record(),
        }


def _normalize(text: str) -> str:
    """答案归一化：去首尾空白、小写、压缩空白。"""
    return " ".join(str(text).strip().lower().split())


def exact_match_reward(task: Task, trajectory: Trajectory) -> float:
    """最朴素的 outcome reward：答案完全一致得 1，否则 0。

    注意：reward 只看最终答案，**不看走了多少轮、调了什么工具**——
    这和 ReTool / Search-R1 的 outcome-only reward 一致，工具使用策略是学出来的。
    """
    if not trajectory.solved or trajectory.final_answer is None:
        return 0.0
    if task.expected_answer is None:
        return 1.0
    return 1.0 if _normalize(trajectory.final_answer) == _normalize(task.expected_answer) else 0.0


@dataclass
class HarnessStats:
    """一批 rollout 的聚合指标。"""

    episodes: int = 0
    solved: int = 0
    avg_reward: float = 0.0
    avg_turns: float = 0.0
    avg_wall_time_s: float = 0.0
    total_tool_calls: int = 0
    protocol_errors: int = 0
    tool_errors: int = 0
    timeouts: int = 0
    truncated_observations: int = 0
    context_trims: int = 0

    @classmethod
    def from_results(cls, results: Sequence[EpisodeResult]) -> "HarnessStats":
        """从一批 episode 结果计算聚合指标。"""
        if not results:
            return cls()
        count = len(results)
        return cls(
            episodes=count,
            solved=sum(1 for result in results if result.trajectory.solved),
            avg_reward=sum(result.reward for result in results) / count,
            avg_turns=sum(len(result.trajectory.turns) for result in results) / count,
            avg_wall_time_s=sum(result.wall_time_s for result in results) / count,
            total_tool_calls=sum(result.trajectory.tool_calls for result in results),
            protocol_errors=sum(result.trajectory.protocol_errors for result in results),
            tool_errors=sum(result.trajectory.tool_errors for result in results),
            timeouts=sum(result.trajectory.timeouts for result in results),
            truncated_observations=sum(
                result.trajectory.truncated_observations for result in results
            ),
            context_trims=sum(result.trajectory.context_trims for result in results),
        )

    @property
    def solved_rate(self) -> float:
        return self.solved / self.episodes if self.episodes else 0.0

    def metrics(self) -> dict[str, float]:
        """转换成 ``harness/*`` 前缀的标量指标，可直接写进 SwanLab。"""
        return {
            "harness/episodes": float(self.episodes),
            "harness/reward": self.avg_reward,
            "harness/solved_rate": self.solved_rate,
            "harness/avg_turns": self.avg_turns,
            "harness/avg_wall_time_s": self.avg_wall_time_s,
            "harness/tool_calls": float(self.total_tool_calls),
            "harness/protocol_errors": float(self.protocol_errors),
            "harness/tool_errors": float(self.tool_errors),
            "harness/timeouts": float(self.timeouts),
            "harness/truncated_observations": float(self.truncated_observations),
            "harness/context_trims": float(self.context_trims),
        }

    def format_table(self) -> str:
        """人类可读的汇总表（demo/测试直接打印这个）。"""
        rows = [
            ("episodes", str(self.episodes)),
            ("solved", str(self.solved)),
            ("solved_rate", f"{self.solved_rate:.1%}"),
            ("avg_reward", f"{self.avg_reward:.3f}"),
            ("avg_turns", f"{self.avg_turns:.2f}"),
            ("avg_wall_time_s", f"{self.avg_wall_time_s:.3f}"),
            ("tool_calls", str(self.total_tool_calls)),
            ("protocol_errors", str(self.protocol_errors)),
            ("tool_errors", str(self.tool_errors)),
            ("timeouts", str(self.timeouts)),
            ("truncated_observations", str(self.truncated_observations)),
            ("context_trims", str(self.context_trims)),
        ]
        width = max(len(name) for name, _ in rows) + 2
        lines = ["-" * (width + 12)]
        lines.extend(f"{name:<{width}}{value:>10}" for name, value in rows)
        lines.append("-" * (width + 12))
        return "\n".join(lines)


def summarize(results: Sequence[EpisodeResult]) -> HarnessStats:
    """便捷函数：``summarize(harness.run_batch(...))``。"""
    return HarnessStats.from_results(results)


class Harness:
    """可复用的 agent harness。

    典型用法：

    .. code-block:: python

        harness = Harness()                       # 默认工作区 = 本目录
        result = harness.run_episode(task, policy)  # 单条
        results = harness.run_batch(tasks, policy_factory)  # 并发一批
        print(summarize(results).format_table())
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        loop_config: LoopConfig | None = None,
        sandbox: SubprocessSandbox | None = None,
        registry: ToolRegistry | None = None,
        reward_fn: RewardFn | None = None,
    ) -> None:
        self.root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
        # 默认每个 Harness 自带一个沙箱；接真实训练时从外面传共享实例
        # （共享实例内部的信号量才能跨 episode 限住并发子进程数）。
        self.sandbox = sandbox if sandbox is not None else SubprocessSandbox()
        self.registry = (
            registry if registry is not None else build_default_registry(self.root, self.sandbox)
        )
        self.loop = AgentLoop(self.registry, loop_config or LoopConfig())
        self.reward_fn = reward_fn or exact_match_reward

    def run_episode(self, task: Task, policy: Policy) -> EpisodeResult:
        """跑一条 episode 并计算 reward。"""
        trajectory = self.loop.run(task.task_id, task.prompt, policy)
        reward = float(self.reward_fn(task, trajectory))
        return EpisodeResult(
            task=task,
            trajectory=trajectory,
            reward=reward,
            wall_time_s=trajectory.wall_time_s,
        )

    def run_batch(
        self,
        tasks: Sequence[Task],
        policy_factory: Callable[[Task], Policy],
        *,
        max_workers: int = 4,
    ) -> list[EpisodeResult]:
        """并发跑一批任务，返回顺序与 ``tasks`` 一致。

        :param policy_factory: 每个 task 构造一个独立 policy 的工厂。
            脚本化 policy / 采样器往往是「一个 task 一个状态」，不能跨 task 复用。
        :param max_workers: 并发 episode 数；真正的 LLM 采样还会受后端 QPS 限制，
            沙箱层面的并发由 :class:`~sandbox.SubprocessSandbox` 内部信号量兜底。
        """
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not tasks:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self.run_episode, task, policy_factory(task)) for task in tasks
            ]
            # 按提交顺序取结果，保证 batch 的指标和顺序可复现。
            return [future.result() for future in futures]

    def summarize(self, results: Sequence[EpisodeResult]) -> HarnessStats:
        """聚合一批结果。"""
        return summarize(results)
