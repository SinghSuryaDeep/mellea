"""Adaptive Repair Strategy for IVR (Instruct-Validate-Repair) loops.

This module provides an advanced sampling strategy that adapts repair feedback
based on failure history across iterations. Unlike simpler strategies that only
look at the most recent failure, AdaptiveRepairStrategy tracks patterns across
all attempts and provides increasingly specific feedback.

Key Features:
    - Tracks failure patterns across all iterations (not just the last one)
    - Prioritizes requirements by failure frequency and validation score
    - Escalates language for repeatedly failing requirements
    - Includes the failed output in feedback for better context
    - Selects the best failed attempt (most requirements passed) when budget exhausts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ...core import (
    Context,
    FancyLogger,
    ModelOutputThunk,
    Requirement,
    ValidationResult,
)
from ..components import Instruction, Message
from ..context import ChatContext
from .base import BaseSamplingStrategy

if TYPE_CHECKING:
    from ...core import Component


class EscalationLevel(Enum):
    """Escalation levels for requirement failures.

    Used to determine how strongly to emphasize a failure in the repair message.
    """

    NORMAL = 1       # First failure
    IMPORTANT = 2    # Failed twice
    CRITICAL = 3     # Failed three or more times


@dataclass
class RequirementFailureStats:
    """Statistics about a requirement's failure history.

    Tracks how many times a requirement has failed across iterations,
    the most recent validation result, and computes priority for repair.

    Attributes:
        requirement: The Requirement object that failed.
        failure_count: Number of times this requirement has failed.
        latest_validation: The most recent ValidationResult for this requirement.
        total_score: Cumulative score across all validations (for averaging).
        description_key: Unique identifier for grouping (requirement description).
    """

    requirement: Requirement
    failure_count: int = 0
    latest_validation: ValidationResult | None = None
    total_score: float = 0.0
    description_key: str = ""

    @property
    def average_score(self) -> float:
        """Calculate average validation score across all failures."""
        if self.failure_count == 0:
            return 0.5  # Default middle score
        return self.total_score / self.failure_count

    @property
    def escalation_level(self) -> EscalationLevel:
        """Determine escalation level based on failure count."""
        if self.failure_count >= 3:
            return EscalationLevel.CRITICAL
        elif self.failure_count == 2:
            return EscalationLevel.IMPORTANT
        return EscalationLevel.NORMAL

    @property
    def priority_score(self) -> tuple[int, float]:
        """Compute priority score for sorting.

        Returns a tuple of (negative_failure_count, average_score) so that:
        - Requirements with more failures come first (higher priority)
        - Among equal failure counts, lower scores come first
        """
        return (-self.failure_count, self.average_score)


@dataclass
class RepairContext:
    """Context information used to generate repair feedback.

    Aggregates all information needed to construct an effective repair message.

    Attributes:
        failure_stats: Dict mapping requirement descriptions to their failure stats.
        current_failures: List of (Requirement, ValidationResult) for the latest attempt.
        last_output: The model's most recent output that failed validation.
        iteration_count: Total number of attempts made so far.
    """

    failure_stats: dict[str, RequirementFailureStats] = field(default_factory=dict)
    current_failures: list[tuple[Requirement, ValidationResult]] = field(default_factory=list)
    last_output: str | None = None
    iteration_count: int = 0

    def get_sorted_failures(self) -> list[RequirementFailureStats]:
        """Get current failures sorted by priority (most critical first)."""
        current_keys = {
            req.description or "[unnamed]"
            for req, _ in self.current_failures
        }
        relevant_stats = [
            stats for key, stats in self.failure_stats.items()
            if key in current_keys
        ]
        return sorted(relevant_stats, key=lambda s: s.priority_score)


class AdaptiveRepairStrategy(BaseSamplingStrategy):
    """A sampling strategy that adapts repair feedback based on failure history.

    This strategy improves upon RepairTemplateStrategy by:

    1. **Tracking failure history**: Maintains statistics across all iterations,
       not just the most recent one. This allows identifying persistent problems.

    2. **Prioritizing failures**: Requirements that fail repeatedly are listed
       first and emphasized more strongly in the repair message.

    3. **Escalating language**: Uses progressively stronger language for
       requirements that fail multiple times ("Issue" -> "Important" -> "CRITICAL").

    4. **Including context**: Shows a snippet of the failed output so the model
       can see exactly what was wrong.

    5. **Smart failure selection**: When all attempts fail, returns the attempt
       that passed the most requirements (not just the first one).

    Args:
        loop_budget: Maximum number of attempts before giving up. Must be >= 1.
        requirements: Optional list of requirements to override instruction requirements.
        max_output_snippet_length: Maximum characters of failed output to include
            in feedback. Set to 0 to disable. Default is 500.
        include_improvement_hints: If True, includes hints about what improved
            between iterations. Default is True.
        context_mode: How to handle context between attempts.
            - "reset": Use the original context (default, like RepairTemplateStrategy)
            - "continue": Continue with the current context (like MultiTurnStrategy)
    """

    def __init__(
        self,
        *,
        loop_budget: int = 3,
        requirements: list[Requirement] | None = None,
        max_output_snippet_length: int = 500,
        include_improvement_hints: bool = True,
        context_mode: str = "reset",
    ):
        """Initialize the AdaptiveRepairStrategy.

        Args:
            loop_budget: Maximum number of attempts. Must be >= 1.
            requirements: Optional requirements to use instead of instruction requirements.
            max_output_snippet_length: Max chars of output to show in feedback (0 to disable).
            include_improvement_hints: Whether to mention improvements between iterations.
            context_mode: Either "reset" or "continue" for context handling.

        Raises:
            ValueError: If loop_budget < 1 or context_mode is invalid.
        """
        if loop_budget < 1:
            raise ValueError(f"loop_budget must be >= 1, got {loop_budget}")
        if context_mode not in ("reset", "continue"):
            raise ValueError(
                f"context_mode must be 'reset' or 'continue', got {context_mode!r}"
            )

        super().__init__(loop_budget=loop_budget, requirements=requirements)
        self.max_output_snippet_length = max_output_snippet_length
        self.include_improvement_hints = include_improvement_hints
        self.context_mode = context_mode

    @staticmethod
    def _build_repair_context(
        past_val: list[list[tuple[Requirement, ValidationResult]]],
        past_results: list[ModelOutputThunk],
    ) -> RepairContext:
        """Build a RepairContext from the validation history.

        Analyzes all past validation results to compute failure statistics
        and identify patterns.

        Args:
            past_val: List of validation results for each iteration.
                Each element is a list of (Requirement, ValidationResult) tuples.
            past_results: List of model outputs for each iteration.

        Returns:
            A RepairContext containing aggregated failure statistics.
        """
        context = RepairContext()
        context.iteration_count = len(past_val)

        # Analyze all iterations to build failure statistics
        for iteration_results in past_val:
            for req, val in iteration_results:
                if not val.as_bool():
                    key = req.description or "[unnamed requirement]"

                    if key not in context.failure_stats:
                        context.failure_stats[key] = RequirementFailureStats(
                            requirement=req,
                            description_key=key,
                        )

                    stats = context.failure_stats[key]
                    stats.failure_count += 1
                    stats.latest_validation = val

                    if val.score is not None:
                        stats.total_score += val.score
                    else:
                        stats.total_score += 0.0

        # Set current failures (from the most recent iteration)
        if past_val:
            context.current_failures = [
                (req, val) for req, val in past_val[-1] if not val.as_bool()
            ]

        # Get the last output
        if past_results:
            context.last_output = past_results[-1].value

        return context

    @staticmethod
    def _format_escalation_prefix(level: EscalationLevel, count: int) -> str:
        """Format the prefix for a failure based on escalation level.

        Args:
            level: The escalation level.
            count: The number of times this requirement has failed.

        Returns:
            A formatted prefix string.
        """
        if level == EscalationLevel.CRITICAL:
            return f"🚨 CRITICAL (failed {count}x)"
        elif level == EscalationLevel.IMPORTANT:
            return f"⚠️ Important (failed {count}x)"
        else:
            return "• Issue"

    @staticmethod
    def _truncate_output(output: str, max_length: int) -> str:
        """Truncate output to a maximum length, adding ellipsis if needed.

        Args:
            output: The output string to truncate.
            max_length: Maximum allowed length.

        Returns:
            The truncated string with ellipsis if it was truncated.
        """
        if not output or max_length <= 0:
            return ""

        if len(output) <= max_length:
            return output

        truncation_point = max_length - 3  # Leave room for "..."

        # Try to break at a space
        last_space = output.rfind(" ", 0, truncation_point)
        if last_space > truncation_point * 0.7:  # Only use space if it's not too far back
            truncation_point = last_space

        return output[:truncation_point] + "..."

    @staticmethod
    def _build_repair_message(
        repair_context: RepairContext,
        max_output_snippet_length: int,
        include_improvement_hints: bool,
    ) -> str:
        """Build the repair message from the repair context.

        Constructs a structured feedback message that:
        - Shows the failed output (truncated)
        - Lists failures in priority order
        - Uses escalated language for repeated failures
        - Includes improvement hints if enabled

        Args:
            repair_context: The aggregated repair context.
            max_output_snippet_length: Max length for output snippet.
            include_improvement_hints: Whether to include improvement hints.

        Returns:
            A formatted repair message string.
        """
        lines: list[str] = []

        lines.append(
            f"Your output (attempt {repair_context.iteration_count}) "
            "did not meet all requirements."
        )
        lines.append("")

        # Show the failed output if available and enabled
        if repair_context.last_output and max_output_snippet_length > 0:
            snippet = AdaptiveRepairStrategy._truncate_output(
                repair_context.last_output,
                max_output_snippet_length,
            )
            lines.append("Your previous attempt:")
            lines.append(f'"""{snippet}"""')
            lines.append("")

        # Get sorted failures (most critical first)
        sorted_stats = repair_context.get_sorted_failures()

        if not sorted_stats:
            lines.append("No specific failure information available.")
            return "\n".join(lines)

        lines.append(f"Issues to fix ({len(sorted_stats)} remaining):")
        lines.append("")

        for stats in sorted_stats:
            prefix = AdaptiveRepairStrategy._format_escalation_prefix(
                stats.escalation_level,
                stats.failure_count,
            )

            # Use the validation reason if available, otherwise use description
            if stats.latest_validation is not None and stats.latest_validation.reason:
                detail = stats.latest_validation.reason
            else:
                detail = stats.description_key

            # Add score information if available and meaningful
            score_info = ""
            if stats.latest_validation is not None and stats.latest_validation.score is not None:
                score = stats.latest_validation.score
                if score < 0.3:
                    score_info = " [far from passing]"
                elif score < 0.7:
                    score_info = " [partially met]"

            lines.append(f"  {prefix}: {detail}{score_info}")

        # Add improvement hints if enabled and there's history
        if include_improvement_hints and repair_context.iteration_count > 1:
            all_failed_keys = set(repair_context.failure_stats.keys())
            current_failed_keys = {
                req.description or "[unnamed]"
                for req, _ in repair_context.current_failures
            }
            improved_keys = all_failed_keys - current_failed_keys

            if improved_keys:
                lines.append("")
                lines.append(f"✓ Good progress: {len(improved_keys)} issue(s) now resolved!")

        return "\n".join(lines)

    def repair(
        self,
        old_ctx: Context,
        new_ctx: Context,
        past_actions: list[Component],
        past_results: list[ModelOutputThunk],
        past_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> tuple[Component, Context]:
        """Generate a repair action based on failure history.

        Analyzes the full history of failures and generates contextually
        appropriate feedback using the instance's configuration.

        Args:
            old_ctx: The context WITHOUT the last action + output.
            new_ctx: The context including the last action + output.
            past_actions: List of actions that have been executed (without success).
            past_results: List of (unsuccessful) generation results for these actions.
            past_val: List of validation results for the results.

        Returns:
            A tuple of (next_action, context) for the next generation attempt.

        Raises:
            ValueError: If past_actions or past_val is empty.
        """
        flog = FancyLogger.get_logger()

        if not past_actions:
            flog.warning("repair() called with empty past_actions")
            raise ValueError("past_actions cannot be empty")

        if not past_val:
            flog.warning("repair() called with empty past_val")
            raise ValueError("past_val cannot be empty")

        repair_context = AdaptiveRepairStrategy._build_repair_context(
            past_val, past_results
        )

        last_action = past_actions[-1]
        ctx_to_use = old_ctx if self.context_mode == "reset" else new_ctx

        # Handle Instruction components (most common case)
        if isinstance(last_action, Instruction):
            repair_message = AdaptiveRepairStrategy._build_repair_message(
                repair_context,
                self.max_output_snippet_length,
                self.include_improvement_hints,
            )
            repaired_instruction = last_action.copy_and_repair(
                repair_string=repair_message
            )
            flog.info(
                f"Generated repair message for attempt {repair_context.iteration_count + 1}: "
                f"{len(repair_context.current_failures)} failures"
            )
            return repaired_instruction, ctx_to_use

        # Handle ChatContext with Message-based repair (like MultiTurnStrategy)
        elif isinstance(new_ctx, ChatContext):
            repair_message = AdaptiveRepairStrategy._build_repair_message(
                repair_context,
                self.max_output_snippet_length,
                self.include_improvement_hints,
            )
            next_action = Message(
                role="user",
                content=repair_message + "\n\nPlease try again.",
            )
            flog.info(
                f"Generated chat repair message for attempt {repair_context.iteration_count + 1}"
            )
            return next_action, new_ctx

        # Fallback: return the action unchanged (like RejectionSamplingStrategy)
        else:
            flog.warning(
                f"Unsupported action type {type(last_action).__name__}, "
                "falling back to unchanged action"
            )
            return last_action, ctx_to_use

    @staticmethod
    def select_from_failure(
        sampled_actions: list[Component],
        sampled_results: list[ModelOutputThunk],
        sampled_val: list[list[tuple[Requirement, ValidationResult]]],
    ) -> int:
        """Select the best failed attempt when budget is exhausted.

        Unlike simpler strategies that always return index 0 or -1, this method
        analyzes all attempts and returns the one that:
        1. Passed the most requirements
        2. On tie, has the highest average validation score
        3. On tie, prefers later attempts (more refined)

        Args:
            sampled_actions: List of actions that have been executed.
            sampled_results: List of generation results.
            sampled_val: List of validation results for each attempt.

        Returns:
            The index of the best failed attempt.
        """
        if not sampled_val:
            return 0

        if len(sampled_val) == 1:
            return 0

        best_index = 0
        best_score = (-1, -1.0, -1)  # (pass_count, avg_score, index)

        flog = FancyLogger.get_logger()

        for idx, validation_results in enumerate(sampled_val):
            if not validation_results:
                continue

            pass_count = sum(1 for _, val in validation_results if val.as_bool())

            failed_scores = [
                val.score for _, val in validation_results
                if not val.as_bool() and val.score is not None
            ]
            avg_score = sum(failed_scores) / len(failed_scores) if failed_scores else 0.0

            current_score = (pass_count, avg_score, idx)
            if current_score > best_score:
                best_score = current_score
                best_index = idx

        flog.info(
            f"select_from_failure: chose attempt {best_index} "
            f"with {best_score[0]} passed requirements "
            f"(avg_score: {best_score[1]:.2f})"
        )

        return best_index

    def __repr__(self) -> str:
        """Return a string representation of the strategy."""
        return (
            f"AdaptiveRepairStrategy("
            f"loop_budget={self.loop_budget}, "
            f"max_output_snippet_length={self.max_output_snippet_length}, "
            f"include_improvement_hints={self.include_improvement_hints}, "
            f"context_mode={self.context_mode!r})"
        )
