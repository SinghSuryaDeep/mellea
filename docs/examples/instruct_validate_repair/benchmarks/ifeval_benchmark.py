"""IFEval benchmark integration for IVR strategy comparison.

IFEval (ICLR 2024) evaluates instruction following using 541 prompts with
programmatic verifiers — no LLM judge needed. Each prompt has 1–3 verifiable
constraints (word count, keyword presence, format, punctuation, etc.).

This runner loads the google/IFEval dataset from HuggingFace, converts each
constraint into a mellea Requirement with a programmatic validator, and
compares all four IVR strategies.

Reference: https://arxiv.org/abs/2311.07911

Requirements:
    pip install datasets

Usage:
    cd docs/examples/instruct_validate_repair/benchmarks
    python ifeval_benchmark.py                 # 50 prompts (default)
    python ifeval_benchmark.py --sample 100    # 100 prompts
    python ifeval_benchmark.py --all           # all 541 prompts (slow)
"""

from __future__ import annotations

import json
import re
import sys

from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate

from _common import BenchmarkTask, print_benchmark_report, run_benchmark

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID    = "llama3.2:3b"
LOOP_BUDGET = 4
TRIALS      = 10
SAMPLE_SIZE = 10   # number of IFEval prompts to test (None = all 541)

# ── Instruction verifiers ──────────────────────────────────────────────────────
# Each function takes (response: str, kwargs: dict) and returns (bool, str).
# Returns (None, None) for unsupported instruction types — those are skipped.

def _num_words(text: str) -> int:
    return len(text.split())

def _num_sentences(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])

def _num_paragraphs(text: str) -> int:
    return len([p for p in text.split("\n\n") if p.strip()])

def _num_bullets(text: str) -> int:
    return len([l for l in text.split("\n") if re.match(r"^\s*[-•*]\s", l)])

def _num_highlights(text: str) -> int:
    return len(re.findall(r"\*[^*]+\*", text))

def _num_placeholders(text: str) -> int:
    return len(re.findall(r"\[.*?\]", text))

def _verify(instruction_id: str, kwargs: dict, response: str) -> tuple[bool | None, str | None]:
    """Verify one IFEval instruction. Returns (None, None) if unsupported."""

    # ── Keywords ──────────────────────────────────────────────────────────────

    if instruction_id == "keywords:existence":
        keywords = kwargs.get("keywords", [])
        lower = response.lower()
        missing = [k for k in keywords if k.lower() not in lower]
        return len(missing) == 0, f"Missing keywords: {missing}"

    if instruction_id == "keywords:forbidden_words":
        forbidden = kwargs.get("forbidden_words", [])
        lower = response.lower()
        found = [w for w in forbidden if w.lower() in lower]
        return len(found) == 0, f"Contains forbidden words: {found}"

    if instruction_id == "keywords:frequency":
        keyword  = kwargs.get("keyword", "")
        expected = kwargs.get("frequency", 1)
        relation = kwargs.get("relation", "at least")
        count = len(re.findall(rf"\b{re.escape(keyword)}\b", response, re.IGNORECASE))
        if relation == "at least":
            ok = count >= expected
        elif relation == "at most":
            ok = count <= expected
        else:
            ok = count == expected
        return ok, f"'{keyword}' appears {count}x (need {relation} {expected}x)"

    if instruction_id == "keywords:letter_frequency":
        letter   = kwargs.get("letter", "")
        expected = kwargs.get("let_frequency", 1)
        relation = kwargs.get("let_relation", "at least")
        count = response.lower().count(letter.lower())
        if relation == "at least":
            ok = count >= expected
        elif relation == "at most":
            ok = count <= expected
        else:
            ok = count == expected
        return ok, f"Letter '{letter}' appears {count}x (need {relation} {expected}x)"

    # ── Length constraints ─────────────────────────────────────────────────────

    if instruction_id == "length_constraint:number_words":
        num      = kwargs.get("num_words", 0)
        relation = kwargs.get("relation", "at least")
        count    = _num_words(response)
        if relation == "at least":
            ok = count >= num
        elif relation == "at most":
            ok = count <= num
        else:
            ok = count == num
        return ok, f"Word count: {count} (need {relation} {num})"

    if instruction_id == "length_constraint:number_sentences":
        num      = kwargs.get("num_sentences", 1)
        relation = kwargs.get("relation", "at least")
        count    = _num_sentences(response)
        if relation == "at least":
            ok = count >= num
        elif relation == "at most":
            ok = count <= num
        else:
            ok = count == num
        return ok, f"Sentence count: {count} (need {relation} {num})"

    if instruction_id == "length_constraint:number_paragraphs":
        num   = kwargs.get("num_paragraphs", 1)
        count = _num_paragraphs(response)
        return count == num, f"Paragraph count: {count} (need {num})"

    if instruction_id == "length_constraint:nth_paragraph_first_word":
        n          = kwargs.get("nth_paragraph", 1)
        first_word = kwargs.get("first_word", "")
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        if len(paragraphs) < n:
            return False, f"Response has only {len(paragraphs)} paragraph(s), need {n}"
        actual = paragraphs[n - 1].split()[0].strip(".,!?\"'").lower() if paragraphs[n - 1].split() else ""
        return actual == first_word.lower(), (
            f"Paragraph {n} starts with '{actual}', need '{first_word}'"
        )

    # ── Punctuation ────────────────────────────────────────────────────────────

    if instruction_id == "punctuation:no_comma":
        has = "," in response
        return not has, "Response must not contain commas"

    # ── Start / End ────────────────────────────────────────────────────────────

    if instruction_id == "startend:end_checker":
        end_phrase = kwargs.get("end_phrase", "")
        ok = response.strip().lower().endswith(end_phrase.lower())
        return ok, f"Must end with '{end_phrase}'"

    if instruction_id == "startend:quotation":
        stripped = response.strip()
        ok = stripped.startswith('"') and stripped.endswith('"')
        return ok, "Response must be wrapped in double quotes"

    # ── Case constraints ───────────────────────────────────────────────────────

    if instruction_id == "change_case:english_capital":
        alpha = [c for c in response if c.isalpha()]
        ok = all(c.isupper() for c in alpha) if alpha else True
        return ok, "Response must be entirely in UPPERCASE"

    if instruction_id == "change_case:english_lowercase":
        alpha = [c for c in response if c.isalpha()]
        ok = all(c.islower() for c in alpha) if alpha else True
        return ok, "Response must be entirely in lowercase"

    if instruction_id == "change_case:first_word_answer":
        words = response.strip().split()
        if not words:
            return False, "Empty response"
        ok = words[0][0].isupper() if words[0] else False
        return ok, f"First word '{words[0]}' must be capitalised"

    # ── Detectable format ──────────────────────────────────────────────────────

    if instruction_id == "detectable_format:json_format":
        try:
            json.loads(response.strip())
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Response is not valid JSON: {e}"

    if instruction_id == "detectable_format:number_bullet_lists":
        num   = kwargs.get("num_bullets", 1)
        count = _num_bullets(response)
        return count >= num, f"Bullet count: {count} (need at least {num})"

    if instruction_id == "detectable_format:number_highlighted_sections":
        num   = kwargs.get("num_highlights", 1)
        count = _num_highlights(response)
        return count >= num, f"Highlighted sections: {count} (need at least {num})"

    if instruction_id == "detectable_format:multiple_sections":
        section_splitter = kwargs.get("section_splitter", "###")
        num_sections     = kwargs.get("num_sections", 2)
        sections = [s for s in response.split("\n") if s.strip().startswith(section_splitter)]
        ok = len(sections) >= num_sections
        return ok, f"Found {len(sections)} '{section_splitter}' sections (need {num_sections})"

    if instruction_id == "detectable_format:title":
        # Title must be wrapped in << >> or appear as a markdown header
        has_title = bool(
            re.search(r"<<[^>]+>>", response) or
            re.search(r"^#{1,2}\s+\S", response, re.MULTILINE)
        )
        return has_title, "Response must contain a title (<<title>> or # Title)"

    # ── Detectable content ─────────────────────────────────────────────────────

    if instruction_id == "detectable_content:number_placeholders":
        num   = kwargs.get("num_placeholders", 1)
        count = _num_placeholders(response)
        return count >= num, f"Placeholder count: {count} (need at least {num})"

    if instruction_id == "detectable_content:postscript":
        marker = kwargs.get("postscript_marker", "P.S.")
        ok = marker in response
        return ok, f"Response must contain postscript marker '{marker}'"

    # ── Combination ────────────────────────────────────────────────────────────

    if instruction_id == "combination:two_responses":
        ok = "****" in response
        return ok, "Response must contain two sections separated by '****'"

    if instruction_id == "combination:repeat_prompt":
        prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
        ok = prompt_to_repeat.strip() in response
        return ok, f"Response must repeat the prompt: '{prompt_to_repeat[:50]}...'"

    # ── Unsupported ────────────────────────────────────────────────────────────
    # language:response_language, detectable_format:constrained_response,
    # follow_up:given_prompt — these require LLM judge or context we don't have.
    return None, None


def _make_requirement(instruction_id: str, kwargs: dict) -> Requirement | None:
    """Build a mellea Requirement from an IFEval instruction. Returns None if unsupported."""
    # Quick check — can we verify this instruction type?
    test_ok, _ = _verify(instruction_id, kwargs, "test")
    if test_ok is None:
        return None

    description = f"{instruction_id} {json.dumps(kwargs, ensure_ascii=False)}"

    def _validate(response: str, _id=instruction_id, _kw=kwargs) -> tuple[bool, str]:
        ok, reason = _verify(_id, _kw, response)
        if ok is None:
            return True, "unsupported — skipped"
        return ok, reason or ""

    return Requirement(description, validation_fn=simple_validate(_validate))


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_ifeval_tasks(sample: int | None = SAMPLE_SIZE) -> list[BenchmarkTask]:
    """Load IFEval dataset from HuggingFace and convert to BenchmarkTask list."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading google/IFEval from HuggingFace...")
    ds = load_dataset("google/IFEval", split="train")

    tasks: list[BenchmarkTask] = []
    skipped = 0

    rows = list(ds)
    if sample:
        rows = rows[:sample]

    for row in rows:
        prompt           = row["prompt"]
        instruction_ids  = row["instruction_id_list"]
        kwarg_list       = row["kwargs"]

        requirements = []
        for instr_id, kw in zip(instruction_ids, kwarg_list):
            req = _make_requirement(instr_id, kw)
            if req is not None:
                requirements.append(req)

        if not requirements:
            skipped += 1
            continue

        # Use first 60 chars of prompt as task name
        name = prompt[:60].replace("\n", " ").strip() + ("..." if len(prompt) > 60 else "")
        tasks.append(BenchmarkTask(name=name, prompt=prompt, requirements=requirements))

    print(f"Loaded {len(tasks)} tasks ({skipped} skipped — all constraints unsupported)")
    return tasks


# ── Official IFEval metrics ────────────────────────────────────────────────────

def compute_ifeval_official_metrics(results: list, tasks: list) -> None:
    """Compute and print the four official IFEval metrics for each strategy.

    Prompt-level strict  : fraction of prompts where ALL requirements pass
                           on the final selected attempt (= our success_rate).
    Instruction-level strict: fraction of individual (prompt, requirement) pairs
                           that pass on the final selected attempt.
    Prompt-level loose   : same as strict — our validators already normalise
                           text (lowercase, whitespace), so strict ≈ loose.
    Instruction-level loose: same as instruction-level strict for same reason.

    Note: For exact numbers comparable to the Google IFEval leaderboard, run
    the official evaluation script from github.com/google-research/google-research/
    tree/master/instruction_following_eval on the collected model outputs.
    """
    strategy_order = ["RejectionSampling", "RepairTemplate", "MultiTurn", "AdaptiveRepair"]
    strategies = [s for s in strategy_order if any(r.strategy_name == s for r in results)]

    print("\n" + "=" * 80)
    print("OFFICIAL IFEval METRICS")
    print("Note: loose reported = strict here. Official loose applies response")
    print("transformations (strip first line / headers / bullets) before re-checking.")
    print("For exact loose numbers run the official Google IFEval eval script.")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Prompt-Strict':>14} {'Instr-Strict':>14} {'Prompt-Loose':>14} {'Instr-Loose':>14}")
    print("-" * 80)

    for strategy_name in strategies:
        strategy_results = [r for r in results if r.strategy_name == strategy_name]

        # Prompt-level strict = success_rate per task, averaged across tasks
        prompt_strict_vals = [r.success_rate for r in strategy_results]
        prompt_strict = sum(prompt_strict_vals) / len(prompt_strict_vals) if prompt_strict_vals else 0.0

        # Instruction-level strict = fraction of individual requirements that pass
        # Computed from final_req_results across all trials and tasks
        all_req_results: list[bool] = []
        for bench in strategy_results:
            for trial in bench.trials:
                all_req_results.extend(trial.final_req_results)
        instr_strict = sum(all_req_results) / len(all_req_results) if all_req_results else 0.0

        # Loose ≈ strict for our validators (they already normalise comparisons)
        prompt_loose = prompt_strict
        instr_loose  = instr_strict

        print(
            f"{strategy_name:<20} "
            f"{prompt_strict*100:>13.1f}% "
            f"{instr_strict*100:>13.1f}% "
            f"{prompt_loose*100:>13.1f}% "
            f"{instr_loose*100:>13.1f}%"
        )
    print("=" * 80 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = None if "--all" in sys.argv else SAMPLE_SIZE
    if "--sample" in sys.argv:
        idx = sys.argv.index("--sample")
        sample = int(sys.argv[idx + 1])

    tasks = load_ifeval_tasks(sample)
    if not tasks:
        print("No tasks loaded. Exiting.")
        sys.exit(1)

    total_reqs = sum(len(t.requirements) for t in tasks)
    print(f"\nIFEval benchmark: {len(tasks)} prompts, {total_reqs} total requirements")
    print(f"Model: {MODEL_ID}  |  loop_budget={LOOP_BUDGET}  |  trials={TRIALS}\n")

    results = run_benchmark(tasks, MODEL_ID, LOOP_BUDGET, TRIALS)
    print_benchmark_report(results, tasks, "IFEval", MODEL_ID, LOOP_BUDGET, TRIALS)
    compute_ifeval_official_metrics(results, tasks)
