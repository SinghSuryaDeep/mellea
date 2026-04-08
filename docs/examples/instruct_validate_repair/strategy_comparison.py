# pytest: ollama, qualitative, llm

"""Strategy Comparison: RejectionSampling vs RepairTemplate vs MultiTurn vs AdaptiveRepair.

This script benchmarks the four built-in IVR sampling strategies on a fixed set
of tasks with the same loop_budget, measuring:

  - Success rate     : fraction of trials where all requirements were satisfied
  - Avg attempts     : average number of LLM calls made (lower = more efficient)
  - Avg reqs passed  : average requirements satisfied per attempt (improvement signal)

Usage:
    python docs/examples/instruct_validate_repair/strategy_comparison.py

Requirements:
    - Ollama running locally (default port 11434)
    - Model pulled: ollama pull llama3.2:1b
"""

from dataclasses import dataclass, field
from mellea import start_session
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import (
    AdaptiveRepairStrategy,
    MultiTurnStrategy,
    RejectionSamplingStrategy,
    RepairTemplateStrategy,
)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "gpt-oss:20b"   # Change to any model available in your Ollama
LOOP_BUDGET = 4            # Max attempts per trial
TRIALS = 5                 # Number of times each (strategy, task) pair is run

# ── Benchmark tasks ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    """A single benchmark task with a prompt and requirements."""

    name: str
    prompt: str
    requirements: list[str]


TASKS = [
    BenchmarkTask(
        name="Exact Word Count",
        prompt="Write a short description of what a smartphone is.",
        requirements=[
            "The response must be exactly 25 words — count carefully, not 24 not 26",
            "Do not use the words 'device', 'phone', 'mobile', or 'smart'",
            "The response must end with a question mark",
            "Do not start any sentence with the word 'It'",
        ],
    ),
    BenchmarkTask(
        name="Tight Multi-Constraint",
        prompt="Explain three benefits of exercise.",
        requirements=[
            "Use exactly 3 numbered points (1. 2. 3.) — no introduction or conclusion text",
            "Each numbered point must be a single sentence of no more than 10 words",
            "Do not use the words 'health', 'healthy', 'body', 'mind', or 'physical'",
            "Each sentence must end with a period",
            "Do not use the words 'also', 'additionally', or 'furthermore'",
        ],
    ),
    BenchmarkTask(
        name="Constrained Story",
        prompt="Write a 2-sentence story about a robot.",
        requirements=[
            "The story must be exactly 2 sentences — no more, no less",
            "The first sentence must contain exactly 8 words",
            "Do not use the words 'metal', 'machine', 'programmed', or 'built'",
            "The second sentence must end with an exclamation mark",
            "The word 'robot' must appear exactly once across both sentences",
        ],
    ),
]

# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Result from a single strategy + task trial."""

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
        """Fraction of trials that succeeded."""
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.success) / len(self.trials)

    @property
    def avg_attempts(self) -> float:
        """Average number of attempts across all trials."""
        if not self.trials:
            return 0.0
        return sum(t.attempts for t in self.trials) / len(self.trials)

    @property
    def avg_reqs_passed_attempt_1(self) -> float:
        """Average requirements passed on the very first attempt (baseline)."""
        scores = [
            t.reqs_passed_per_attempt[0]
            for t in self.trials
            if t.reqs_passed_per_attempt
        ]
        return sum(scores) / len(scores) if scores else 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_trial(strategy, task: BenchmarkTask) -> TrialResult:
    """Run a single trial and return the result."""
    m = start_session("ollama", model_id=MODEL_ID, ctx=ChatContext())

    result = m.instruct(
        task.prompt,
        requirements=task.requirements,
        strategy=strategy,
        return_sampling_results=True,
    )

    # Count requirements passed per attempt
    reqs_passed_per_attempt = [
        sum(1 for _, val in attempt_vals if val)
        for attempt_vals in result.sample_validations
    ]

    return TrialResult(
        success=result.success,
        attempts=len(result.sample_generations),
        reqs_passed_per_attempt=reqs_passed_per_attempt,
    )


def run_benchmark() -> list[BenchmarkResult]:
    """Run all strategies on all tasks and return aggregated results."""
    strategies = [
        ("RejectionSampling", RejectionSamplingStrategy(loop_budget=LOOP_BUDGET)),
        ("RepairTemplate",    RepairTemplateStrategy(loop_budget=LOOP_BUDGET)),
        ("MultiTurn",         MultiTurnStrategy(loop_budget=LOOP_BUDGET)),
        ("AdaptiveRepair",    AdaptiveRepairStrategy(loop_budget=LOOP_BUDGET)),
    ]

    results: list[BenchmarkResult] = []

    total = len(strategies) * len(TASKS) * TRIALS
    done = 0

    for strategy_name, strategy in strategies:
        for task in TASKS:
            bench = BenchmarkResult(strategy_name=strategy_name, task_name=task.name)

            for trial_num in range(1, TRIALS + 1):
                done += 1
                print(f"  [{done}/{total}] {strategy_name} | {task.name} | trial {trial_num}", flush=True)
                trial = run_trial(strategy, task)
                bench.trials.append(trial)
                status = "✓" if trial.success else "✗"
                print(f"         {status} success={trial.success}  attempts={trial.attempts}")

            results.append(bench)

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: list[BenchmarkResult]) -> None:
    """Print a formatted comparison table."""
    tasks = list({r.task_name for r in results})
    strategies = list({r.strategy_name for r in results})

    # Preserve insertion order
    strategy_order = ["RejectionSampling", "RepairTemplate", "MultiTurn", "AdaptiveRepair"]
    strategies = [s for s in strategy_order if s in strategies]

    col_w = 18

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON RESULTS")
    print(f"Model: {MODEL_ID}  |  loop_budget={LOOP_BUDGET}  |  trials={TRIALS}")
    print("=" * 80)

    for task_name in tasks:
        print(f"\nTask: {task_name}")
        print("-" * 70)
        header = f"{'Strategy':<{col_w}} {'Success Rate':>14} {'Avg Attempts':>14} {'Reqs @ Attempt 1':>18}"
        print(header)
        print("-" * 70)

        for strategy_name in strategies:
            bench = next(
                (r for r in results if r.strategy_name == strategy_name and r.task_name == task_name),
                None,
            )
            if bench is None:
                continue

            total_reqs = len(next(t for t in TASKS if t.name == task_name).requirements)
            print(
                f"{strategy_name:<{col_w}} "
                f"{bench.success_rate * 100:>13.0f}% "
                f"{bench.avg_attempts:>14.1f} "
                f"{bench.avg_reqs_passed_attempt_1:>14.1f}/{total_reqs}"
            )

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY (averaged across all tasks)")
    print("-" * 70)
    header = f"{'Strategy':<{col_w}} {'Success Rate':>14} {'Avg Attempts':>14}"
    print(header)
    print("-" * 70)

    for strategy_name in strategies:
        strategy_results = [r for r in results if r.strategy_name == strategy_name]
        avg_success = sum(r.success_rate for r in strategy_results) / len(strategy_results)
        avg_attempts = sum(r.avg_attempts for r in strategy_results) / len(strategy_results)
        print(
            f"{strategy_name:<{col_w}} "
            f"{avg_success * 100:>13.0f}% "
            f"{avg_attempts:>14.1f}"
        )

    print("=" * 80 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nRunning strategy comparison benchmark...")
    print(f"Model: {MODEL_ID} | loop_budget={LOOP_BUDGET} | {TRIALS} trials per (strategy, task)\n")

    results = run_benchmark()
    print_report(results)
