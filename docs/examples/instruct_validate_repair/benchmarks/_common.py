"""Shared infrastructure for IVR benchmark evaluations.

Provides common dataclasses, a trial runner, and a report printer
used by all three benchmark scripts (infobench, multi_if, feedbackeval).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mellea import start_session
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import (
    AdaptiveRepairStrategy,
    MultiTurnStrategy,
    RejectionSamplingStrategy,
    RepairTemplateStrategy,
)


@dataclass
class BenchmarkTask:
    """A single benchmark task with a prompt and list of requirements."""

    name: str
    prompt: str
    requirements: list[Any]  # str | Requirement


@dataclass
class TrialResult:
    """Result from one (strategy, task) trial."""

    success: bool
    attempts: int
    reqs_passed_per_attempt: list[int] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Aggregated results for one (strategy, task) pair."""

    strategy_name: str
    task_name: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.success) / len(self.trials)

    @property
    def avg_attempts(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.attempts for t in self.trials) / len(self.trials)

    @property
    def avg_reqs_passed_attempt_1(self) -> float:
        scores = [
            t.reqs_passed_per_attempt[0]
            for t in self.trials
            if t.reqs_passed_per_attempt
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def efficiency_score(self) -> float:
        """Success rate divided by avg attempts — higher means faster convergence."""
        if self.avg_attempts == 0:
            return 0.0
        return self.success_rate / self.avg_attempts


def get_strategies(loop_budget: int) -> list[tuple[str, Any]]:
    """Return all four IVR strategies with the given loop budget."""
    return [
        ("RejectionSampling", RejectionSamplingStrategy(loop_budget=loop_budget)),
        ("RepairTemplate",    RepairTemplateStrategy(loop_budget=loop_budget)),
        ("MultiTurn",         MultiTurnStrategy(loop_budget=loop_budget)),
        ("AdaptiveRepair",    AdaptiveRepairStrategy(loop_budget=loop_budget)),
    ]


def run_trial(strategy: Any, task: BenchmarkTask, model_id: str) -> TrialResult:
    """Run a single (strategy, task) trial and return results."""
    m = start_session("ollama", model_id=model_id, ctx=ChatContext())
    result = m.instruct(
        task.prompt,
        requirements=task.requirements,
        strategy=strategy,
        return_sampling_results=True,
    )
    reqs_passed_per_attempt = [
        sum(1 for _, val in attempt_vals if val)
        for attempt_vals in result.sample_validations
    ]
    return TrialResult(
        success=result.success,
        attempts=len(result.sample_generations),
        reqs_passed_per_attempt=reqs_passed_per_attempt,
    )


def run_benchmark(
    tasks: list[BenchmarkTask],
    model_id: str,
    loop_budget: int,
    trials: int,
) -> list[BenchmarkResult]:
    """Run all four strategies against all tasks and return aggregated results."""
    strategies = get_strategies(loop_budget)
    results: list[BenchmarkResult] = []
    total = len(strategies) * len(tasks) * trials
    done = 0

    for strategy_name, strategy in strategies:
        for task in tasks:
            bench = BenchmarkResult(strategy_name=strategy_name, task_name=task.name)
            for trial_num in range(1, trials + 1):
                done += 1
                print(
                    f"  [{done}/{total}] {strategy_name} | {task.name} | trial {trial_num}",
                    flush=True,
                )
                trial = run_trial(strategy, task, model_id)
                bench.trials.append(trial)
                status = "✓" if trial.success else "✗"
                print(f"         {status} success={trial.success}  attempts={trial.attempts}")
            results.append(bench)

    return results


def print_benchmark_report(
    results: list[BenchmarkResult],
    tasks: list[BenchmarkTask],
    benchmark_name: str,
    model_id: str,
    loop_budget: int,
    trials: int,
) -> None:
    """Print a formatted comparison table for a benchmark run."""
    strategy_order = ["RejectionSampling", "RepairTemplate", "MultiTurn", "AdaptiveRepair"]
    strategies = [s for s in strategy_order if any(r.strategy_name == s for r in results)]
    col_w = 18

    print("\n" + "=" * 80)
    print(f"{benchmark_name} RESULTS")
    print(f"Model: {model_id}  |  loop_budget={loop_budget}  |  trials={trials}")
    print("=" * 80)

    for task in tasks:
        n_reqs = len(task.requirements)
        print(f"\nTask: {task.name} ({n_reqs} requirements)")
        print("-" * 70)
        print(
            f"{'Strategy':<{col_w}} {'Success Rate':>14} {'Avg Attempts':>14} {'Reqs @ Attempt 1':>18}"
        )
        print("-" * 70)
        for strategy_name in strategies:
            bench = next(
                (r for r in results if r.strategy_name == strategy_name and r.task_name == task.name),
                None,
            )
            if bench is None:
                continue
            print(
                f"{strategy_name:<{col_w}} "
                f"{bench.success_rate * 100:>13.0f}% "
                f"{bench.avg_attempts:>14.1f} "
                f"{bench.avg_reqs_passed_attempt_1:>14.1f}/{n_reqs}"
            )

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (averaged across all tasks)")
    print("-" * 80)
    print(f"{'Strategy':<{col_w}} {'Success Rate':>14} {'Avg Attempts':>14} {'Efficiency':>12}")
    print("-" * 80)
    for strategy_name in strategies:
        strategy_results = [r for r in results if r.strategy_name == strategy_name]
        if not strategy_results:
            continue
        avg_success    = sum(r.success_rate for r in strategy_results) / len(strategy_results)
        avg_attempts   = sum(r.avg_attempts for r in strategy_results) / len(strategy_results)
        avg_efficiency = sum(r.efficiency_score for r in strategy_results) / len(strategy_results)
        print(
            f"{strategy_name:<{col_w}} "
            f"{avg_success * 100:>13.0f}% "
            f"{avg_attempts:>14.1f} "
            f"{avg_efficiency:>12.2f}"
        )
    print("=" * 80 + "\n")
