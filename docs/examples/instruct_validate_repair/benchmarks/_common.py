"""Shared infrastructure for IVR benchmark evaluations.

Provides common dataclasses, a trial runner, and a report printer
used by all three benchmark scripts (infobench, multi_if, feedbackeval).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from mellea import start_session
from mellea.core import Requirement
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import (
    AdaptiveRepairStrategy,
    MultiTurnStrategy,
    RejectionSamplingStrategy,
    RepairTemplateStrategy,
)


# ── Shared LLM judge infrastructure ───────────────────────────────────────────

def ollama_judge(prompt: str, model_id: str) -> str:
    """Synchronous Ollama API call for judge validation. Returns raw response text."""
    body = json.dumps({"model": model_id, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["response"]


def _parse_judge_response(response: str) -> bool:
    """Strip DeepSeek-R1 <think> block and return True if response starts with YES."""
    if "</think>" in response:
        response = response.split("</think>")[-1]
    return response.strip().upper().startswith("YES")


def create_joint_validator(
    all_constraints: list[str],
    judge_model_id: str,
    original_instruction: str | None = None,
) -> Any:
    """Return a simple_validate-compatible validator that evaluates ALL constraints
    simultaneously. Used as the authoritative signal for compositional scoring
    (HSR/SSR/CSL in FollowBench, task-level pass/fail in ComplexBench).

    Per-constraint validators drive the repair loop; this joint validator drives
    final scoring metrics only.

    Args:
        all_constraints: All constraint texts that must be satisfied together.
        judge_model_id: Ollama model ID used as judge (e.g. "deepseek-r1:8b").
        original_instruction: Full original prompt, included as context for the
            judge so it can evaluate constraints in the right framing.
    """
    def _check(out: str) -> tuple[bool, str]:
        instr_block = (
            f"Original instruction:\n{original_instruction}\n\n"
            if original_instruction else ""
        )
        constraints_block = "\n".join(
            f"{i + 1}. {c}" for i, c in enumerate(all_constraints)
        )
        prompt = (
            f"{instr_block}"
            f"The response must satisfy ALL of these constraints simultaneously:\n"
            f"{constraints_block}\n\n"
            f'Response:\n"""\n{out}\n"""\n\n'
            f"Does the response satisfy ALL of the above constraints simultaneously?\n"
            f"Answer with only YES or NO."
        )
        try:
            raw = ollama_judge(prompt, judge_model_id)
        except Exception as e:
            return (False, f"Joint judge error: {e}")
        passed = _parse_judge_response(raw)
        return (passed, "" if passed else "Not all constraints satisfied simultaneously")

    return simple_validate(_check)


@dataclass
class BenchmarkTask:
    """A single benchmark task with a prompt and list of requirements."""

    name: str
    prompt: str
    requirements: list[Any]  # str | Requirement
    example_id: int | None = None      # FollowBench: source example identifier
    level: int | None = None           # FollowBench: constraint difficulty level (1–5)
    joint_req_index: int | None = None # index of joint compositional Requirement in requirements


@dataclass
class TrialResult:
    """Result from one (strategy, task) trial."""

    success: bool
    attempts: int
    reqs_passed_per_attempt: list[int] = field(default_factory=list)
    final_req_results: list[bool] = field(default_factory=list)
    # ^ per-requirement pass/fail for the final selected attempt


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
    # Per-requirement pass/fail for the final selected attempt
    final_vals = result.sample_validations[result.result_index]
    final_req_results = [bool(val) for _, val in final_vals]
    return TrialResult(
        success=result.success,
        attempts=len(result.sample_generations),
        reqs_passed_per_attempt=reqs_passed_per_attempt,
        final_req_results=final_req_results,
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
