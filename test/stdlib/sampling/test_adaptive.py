"""Unit tests for AdaptiveRepairStrategy."""

from unittest.mock import MagicMock

import pytest

from mellea.core import ModelOutputThunk, Requirement, ValidationResult
from mellea.stdlib.components import Instruction, Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling.adaptive import (
    AdaptiveRepairStrategy,
    EscalationLevel,
    RepairContext,
    RequirementFailureStats,
)


class TestAdaptiveRepairStrategyInit:
    """Test AdaptiveRepairStrategy initialization."""

    def test_default_init(self):
        strategy = AdaptiveRepairStrategy()
        assert strategy.loop_budget == 3
        assert strategy.max_output_snippet_length == 500
        assert strategy.include_improvement_hints is True
        assert strategy.context_mode == "reset"

    def test_custom_init(self):
        strategy = AdaptiveRepairStrategy(
            loop_budget=5,
            max_output_snippet_length=200,
            include_improvement_hints=False,
            context_mode="continue",
        )
        assert strategy.loop_budget == 5
        assert strategy.max_output_snippet_length == 200
        assert strategy.include_improvement_hints is False
        assert strategy.context_mode == "continue"

    def test_invalid_loop_budget_raises(self):
        with pytest.raises(ValueError, match="loop_budget must be >= 1"):
            AdaptiveRepairStrategy(loop_budget=0)

    def test_invalid_context_mode_raises(self):
        with pytest.raises(ValueError, match="context_mode must be"):
            AdaptiveRepairStrategy(context_mode="invalid")

    def test_repr(self):
        strategy = AdaptiveRepairStrategy(loop_budget=2)
        r = repr(strategy)
        assert "AdaptiveRepairStrategy" in r
        assert "loop_budget=2" in r


class TestRequirementFailureStats:
    """Test RequirementFailureStats dataclass properties."""

    def test_average_score_no_failures(self):
        req = Requirement(description="test")
        stats = RequirementFailureStats(requirement=req, failure_count=0)
        assert stats.average_score == 0.5

    def test_average_score_with_failures(self):
        req = Requirement(description="test")
        stats = RequirementFailureStats(
            requirement=req, failure_count=2, total_score=0.6
        )
        assert stats.average_score == pytest.approx(0.3)

    def test_escalation_normal(self):
        req = Requirement(description="test")
        stats = RequirementFailureStats(requirement=req, failure_count=1)
        assert stats.escalation_level == EscalationLevel.NORMAL

    def test_escalation_important(self):
        req = Requirement(description="test")
        stats = RequirementFailureStats(requirement=req, failure_count=2)
        assert stats.escalation_level == EscalationLevel.IMPORTANT

    def test_escalation_critical(self):
        req = Requirement(description="test")
        stats = RequirementFailureStats(requirement=req, failure_count=3)
        assert stats.escalation_level == EscalationLevel.CRITICAL

    def test_priority_score_ordering(self):
        req = Requirement(description="test")
        stats_rare = RequirementFailureStats(requirement=req, failure_count=1)
        stats_frequent = RequirementFailureStats(requirement=req, failure_count=3)
        # More failures = higher priority (lower tuple value due to negation)
        assert stats_frequent.priority_score < stats_rare.priority_score


class TestBuildRepairContext:
    """Test AdaptiveRepairStrategy._build_repair_context."""

    def test_empty_history(self):
        ctx = AdaptiveRepairStrategy._build_repair_context([], [])
        assert ctx.iteration_count == 0
        assert ctx.failure_stats == {}
        assert ctx.current_failures == []
        assert ctx.last_output is None

    def test_single_iteration_one_failure(self):
        req = Requirement(description="Be concise")
        val = ValidationResult(False, reason="Too long")

        result = MagicMock(spec=ModelOutputThunk)
        result.value = "some long output"

        ctx = AdaptiveRepairStrategy._build_repair_context(
            past_val=[[(req, val)]],
            past_results=[result],
        )

        assert ctx.iteration_count == 1
        assert "Be concise" in ctx.failure_stats
        assert ctx.failure_stats["Be concise"].failure_count == 1
        assert ctx.last_output == "some long output"
        assert len(ctx.current_failures) == 1

    def test_passed_requirements_not_tracked(self):
        req = Requirement(description="Be formal")
        val = ValidationResult(True)  # Passed

        ctx = AdaptiveRepairStrategy._build_repair_context(
            past_val=[[(req, val)]],
            past_results=[],
        )

        assert "Be formal" not in ctx.failure_stats

    def test_repeated_failures_accumulate(self):
        req = Requirement(description="Use headers")
        val = ValidationResult(False)

        ctx = AdaptiveRepairStrategy._build_repair_context(
            past_val=[[(req, val)], [(req, val)], [(req, val)]],
            past_results=[],
        )

        assert ctx.failure_stats["Use headers"].failure_count == 3
        assert ctx.failure_stats["Use headers"].escalation_level == EscalationLevel.CRITICAL

    def test_score_accumulated(self):
        req = Requirement(description="Short")
        val1 = ValidationResult(False, score=0.4)
        val2 = ValidationResult(False, score=0.6)

        ctx = AdaptiveRepairStrategy._build_repair_context(
            past_val=[[(req, val1)], [(req, val2)]],
            past_results=[],
        )

        stats = ctx.failure_stats["Short"]
        assert stats.total_score == pytest.approx(1.0)
        assert stats.average_score == pytest.approx(0.5)


class TestTruncateOutput:
    """Test AdaptiveRepairStrategy._truncate_output."""

    def test_short_output_unchanged(self):
        result = AdaptiveRepairStrategy._truncate_output("hello", 100)
        assert result == "hello"

    def test_long_output_truncated(self):
        long_text = "a" * 600
        result = AdaptiveRepairStrategy._truncate_output(long_text, 100)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_zero_max_length_returns_empty(self):
        result = AdaptiveRepairStrategy._truncate_output("hello world", 0)
        assert result == ""

    def test_empty_input_returns_empty(self):
        result = AdaptiveRepairStrategy._truncate_output("", 100)
        assert result == ""

    def test_word_boundary_preferred(self):
        text = "hello world this is a test string for truncation"
        result = AdaptiveRepairStrategy._truncate_output(text, 20)
        # Should not cut in middle of a word (unless space is too far back)
        assert result.endswith("...")


class TestFormatEscalationPrefix:
    """Test AdaptiveRepairStrategy._format_escalation_prefix."""

    def test_normal_prefix(self):
        prefix = AdaptiveRepairStrategy._format_escalation_prefix(EscalationLevel.NORMAL, 1)
        assert prefix == "• Issue"

    def test_important_prefix(self):
        prefix = AdaptiveRepairStrategy._format_escalation_prefix(EscalationLevel.IMPORTANT, 2)
        assert "Important" in prefix
        assert "2x" in prefix

    def test_critical_prefix(self):
        prefix = AdaptiveRepairStrategy._format_escalation_prefix(EscalationLevel.CRITICAL, 4)
        assert "CRITICAL" in prefix
        assert "4x" in prefix


class TestBuildRepairMessage:
    """Test AdaptiveRepairStrategy._build_repair_message."""

    def _make_repair_context(self, failures, last_output=None, iteration_count=1):
        ctx = RepairContext()
        ctx.iteration_count = iteration_count
        ctx.last_output = last_output
        ctx.current_failures = failures
        for req, val in failures:
            key = req.description or "[unnamed]"
            stats = RequirementFailureStats(
                requirement=req,
                failure_count=1,
                latest_validation=val,
                description_key=key,
            )
            ctx.failure_stats[key] = stats
        return ctx

    def test_message_contains_iteration_count(self):
        repair_ctx = self._make_repair_context([], iteration_count=3)
        msg = AdaptiveRepairStrategy._build_repair_message(repair_ctx, 500, True)
        assert "attempt 3" in msg

    def test_message_includes_output_snippet(self):
        req = Requirement(description="Be short")
        val = ValidationResult(False)
        repair_ctx = self._make_repair_context([(req, val)], last_output="some bad output")
        msg = AdaptiveRepairStrategy._build_repair_message(repair_ctx, 500, True)
        assert "some bad output" in msg

    def test_message_no_snippet_when_disabled(self):
        req = Requirement(description="Be short")
        val = ValidationResult(False)
        repair_ctx = self._make_repair_context([(req, val)], last_output="some bad output")
        msg = AdaptiveRepairStrategy._build_repair_message(repair_ctx, 0, True)
        assert "some bad output" not in msg

    def test_message_uses_validation_reason(self):
        req = Requirement(description="Be formal")
        val = ValidationResult(False, reason="Used slang: 'gonna'")
        repair_ctx = self._make_repair_context([(req, val)])
        msg = AdaptiveRepairStrategy._build_repair_message(repair_ctx, 0, False)
        assert "Used slang" in msg

    def test_message_falls_back_to_description(self):
        req = Requirement(description="Use headers")
        val = ValidationResult(False)  # No reason
        repair_ctx = self._make_repair_context([(req, val)])
        msg = AdaptiveRepairStrategy._build_repair_message(repair_ctx, 0, False)
        assert "Use headers" in msg

    def test_improvement_hint_shown_on_second_iteration(self):
        # Simulate: req1 failed before but passes now, req2 still failing
        req1 = Requirement(description="Use headers")
        req2 = Requirement(description="Be concise")
        val_fail = ValidationResult(False)

        ctx = RepairContext()
        ctx.iteration_count = 2
        ctx.current_failures = [(req2, val_fail)]  # only req2 failing now

        # Both failed in history
        ctx.failure_stats = {
            "Use headers": RequirementFailureStats(
                requirement=req1, failure_count=1, description_key="Use headers"
            ),
            "Be concise": RequirementFailureStats(
                requirement=req2, failure_count=1, description_key="Be concise",
                latest_validation=val_fail,
            ),
        }

        msg = AdaptiveRepairStrategy._build_repair_message(ctx, 0, include_improvement_hints=True)
        assert "Good progress" in msg

    def test_no_improvement_hint_on_first_iteration(self):
        req = Requirement(description="Be short")
        val = ValidationResult(False)
        repair_ctx = self._make_repair_context([(req, val)], iteration_count=1)
        msg = AdaptiveRepairStrategy._build_repair_message(repair_ctx, 0, True)
        assert "Good progress" not in msg


class TestAdaptiveRepair:
    """Test AdaptiveRepairStrategy.repair instance method."""

    def test_repair_with_instruction_uses_reset_context(self):
        old_ctx = MagicMock()
        new_ctx = MagicMock()

        instruction = MagicMock(spec=Instruction)
        repaired = MagicMock(spec=Instruction)
        instruction.copy_and_repair.return_value = repaired

        result_mock = MagicMock(spec=ModelOutputThunk)
        result_mock.value = "bad output"

        req = Requirement(description="Be formal")
        val = ValidationResult(False, reason="Too casual")

        strategy = AdaptiveRepairStrategy()  # default context_mode="reset"
        next_action, returned_ctx = strategy.repair(
            old_ctx=old_ctx,
            new_ctx=new_ctx,
            past_actions=[instruction],
            past_results=[result_mock],
            past_val=[[(req, val)]],
        )

        assert next_action is repaired
        assert returned_ctx is old_ctx  # "reset" mode uses old_ctx

    def test_repair_with_chat_context_returns_message(self):
        old_ctx = MagicMock()
        new_ctx = MagicMock(spec=ChatContext)

        # Use a non-Instruction action to trigger the ChatContext branch
        non_instruction_action = MagicMock()
        non_instruction_action.__class__ = Message

        result_mock = MagicMock(spec=ModelOutputThunk)
        result_mock.value = "output"

        req = Requirement(description="Use headers")
        val = ValidationResult(False, reason="No headers found")

        strategy = AdaptiveRepairStrategy()
        next_action, returned_ctx = strategy.repair(
            old_ctx=old_ctx,
            new_ctx=new_ctx,
            past_actions=[non_instruction_action],
            past_results=[result_mock],
            past_val=[[(req, val)]],
        )

        assert isinstance(next_action, Message)
        assert next_action.role == "user"
        assert "Please try again" in next_action.content
        assert returned_ctx is new_ctx

    def test_repair_raises_on_empty_past_actions(self):
        old_ctx = MagicMock()
        new_ctx = MagicMock()

        with pytest.raises(ValueError, match="past_actions cannot be empty"):
            AdaptiveRepairStrategy().repair(
                old_ctx=old_ctx,
                new_ctx=new_ctx,
                past_actions=[],
                past_results=[],
                past_val=[[]],
            )

    def test_repair_raises_on_empty_past_val(self):
        old_ctx = MagicMock()
        new_ctx = MagicMock()
        action = MagicMock(spec=Instruction)

        with pytest.raises(ValueError, match="past_val cannot be empty"):
            AdaptiveRepairStrategy().repair(
                old_ctx=old_ctx,
                new_ctx=new_ctx,
                past_actions=[action],
                past_results=[],
                past_val=[],
            )

    def test_repair_uses_continue_context_when_configured(self):
        old_ctx = MagicMock()
        new_ctx = MagicMock()

        instruction = MagicMock(spec=Instruction)
        instruction.copy_and_repair.return_value = MagicMock(spec=Instruction)

        result_mock = MagicMock(spec=ModelOutputThunk)
        result_mock.value = "output"

        req = Requirement(description="Be formal")
        val = ValidationResult(False)

        strategy = AdaptiveRepairStrategy(context_mode="continue")
        _, returned_ctx = strategy.repair(
            old_ctx=old_ctx,
            new_ctx=new_ctx,
            past_actions=[instruction],
            past_results=[result_mock],
            past_val=[[(req, val)]],
        )

        assert returned_ctx is new_ctx  # "continue" mode uses new_ctx


class TestSelectFromFailure:
    """Test AdaptiveRepairStrategy.select_from_failure."""

    def test_empty_returns_zero(self):
        idx = AdaptiveRepairStrategy.select_from_failure([], [], [])
        assert idx == 0

    def test_single_attempt_returns_zero(self):
        req = Requirement(description="R1")
        val = ValidationResult(False)
        idx = AdaptiveRepairStrategy.select_from_failure(
            [], [], [[(req, val)]]
        )
        assert idx == 0

    def test_prefers_attempt_with_more_passes(self):
        req1 = Requirement(description="R1")
        req2 = Requirement(description="R2")

        # Attempt 0: 0 passes
        val_fail = ValidationResult(False)
        # Attempt 1: 1 pass, 1 fail
        val_pass = ValidationResult(True)

        sampled_val = [
            [(req1, val_fail), (req2, val_fail)],
            [(req1, val_pass), (req2, val_fail)],
        ]

        idx = AdaptiveRepairStrategy.select_from_failure([], [], sampled_val)
        assert idx == 1

    def test_prefers_later_attempt_on_tie(self):
        req = Requirement(description="R1")
        val_fail = ValidationResult(False)

        sampled_val = [
            [(req, val_fail)],
            [(req, val_fail)],
            [(req, val_fail)],
        ]

        idx = AdaptiveRepairStrategy.select_from_failure([], [], sampled_val)
        assert idx == 2  # Later attempt preferred on tie

    def test_prefers_higher_score_on_equal_passes(self):
        req = Requirement(description="R1")
        val_low = ValidationResult(False, score=0.1)
        val_high = ValidationResult(False, score=0.8)

        sampled_val = [
            [(req, val_low)],
            [(req, val_high)],
        ]

        idx = AdaptiveRepairStrategy.select_from_failure([], [], sampled_val)
        assert idx == 1  # Higher score preferred


@pytest.mark.qualitative
@pytest.mark.ollama
@pytest.mark.llm
class TestAdaptiveIntegration:
    """Integration test — requires Ollama running locally."""

    def test_adaptive_with_ollama(self):
        from mellea import start_session
        from mellea.stdlib.context import ChatContext

        strategy = AdaptiveRepairStrategy(loop_budget=3)
        m = start_session("ollama", model_id="llama3.2:1b", ctx=ChatContext())

        result = m.instruct(
            "Write a formal email",
            requirements=["Be formal", "Include greeting"],
            strategy=strategy,
            return_sampling_results=True,
        )

        assert result is not None
        assert hasattr(result, "success")
        assert len(result.sample_generations) >= 1
        assert result.value is not None

        print(f"\n--- Result ---")
        print(f"Success:  {result.success}")
        print(f"Attempts: {len(result.sample_generations)}")
        print(f"Output:\n{result.value}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
