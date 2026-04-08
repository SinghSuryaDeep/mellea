"""FollowBench benchmark integration for IVR strategy comparison.

FollowBench (ACL 2024) evaluates instruction following at 5 difficulty levels
by progressively adding constraints to a base prompt. Level 1 = 1 constraint,
Level 5 = 5 accumulated constraints. This tests whether strategies can satisfy
increasingly complex, layered requirements.

Constraint types in FollowBench:
  - Content   : what the response should discuss
  - Situation : the scenario or persona to adopt
  - Style     : writing style, tone, register
  - Format    : structure, length, punctuation (programmatically verifiable)
  - Example   : follow a given example pattern

This runner focuses on format constraints (programmatically verifiable) and
uses heuristic validators for style/content constraints where possible.

Reference: https://arxiv.org/abs/2310.20410
Dataset  : YuxinJiang/FollowBench on HuggingFace

Requirements:
    pip install datasets

Usage:
    cd docs/examples/instruct_validate_repair/benchmarks
    python followbench_benchmark.py                # level 3 (default)
    python followbench_benchmark.py --level 5      # hardest level
    python followbench_benchmark.py --sample 30    # 30 prompts
"""

from __future__ import annotations

import re
import sys

from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate

from _common import BenchmarkTask, print_benchmark_report, run_benchmark

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID    = "gpt-oss:20b"
LOOP_BUDGET = 4
TRIALS      = 10
LEVEL       = 5     # constraint difficulty level (1–5)
SAMPLE_SIZE = 10    # number of FollowBench prompts to test

# ── Heuristic validators ───────────────────────────────────────────────────────

def _wc(text: str) -> int:
    return len(text.split())

def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])

def _paragraph_count(text: str) -> int:
    return len([p for p in text.split("\n\n") if p.strip()])

def _bullet_count(text: str) -> int:
    return len([l for l in text.split("\n") if re.match(r"^\s*[-•*]\s", l)])

def _numbered_count(text: str) -> int:
    return len(re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))

def _has_words(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return any(w.lower() in lower for w in words)

def _lacks_words(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return not any(w.lower() in lower for w in words)

# ── Constraint text → Requirement ─────────────────────────────────────────────

def _constraint_to_requirement(constraint: str) -> Requirement:
    """Convert a FollowBench constraint string into a programmatic Requirement.

    Attempts to parse common patterns. Falls back to a keyword-presence check
    for constraints that can't be verified structurally.
    """
    c = constraint.strip().lower()

    # Word count: "your response should be X words" / "at least X words" / "no more than X words"
    m = re.search(r"(?:exactly|at least|at most|no more than|fewer than|more than)\s+(\d+)\s+words?", c)
    if m:
        num = int(m.group(1))
        if "at least" in c or "more than" in c:
            return Requirement(
                constraint,
                validation_fn=simple_validate(
                    lambda out, n=num: (_wc(out) >= n, f"Word count: {_wc(out)} (need >= {n})")
                ),
            )
        elif "at most" in c or "no more than" in c or "fewer than" in c:
            return Requirement(
                constraint,
                validation_fn=simple_validate(
                    lambda out, n=num: (_wc(out) <= n, f"Word count: {_wc(out)} (need <= {n})")
                ),
            )
        else:  # exactly
            return Requirement(
                constraint,
                validation_fn=simple_validate(
                    lambda out, n=num: (_wc(out) == n, f"Word count: {_wc(out)} (need exactly {n})")
                ),
            )

    # Sentence count: "use X sentences" / "in X sentences"
    m = re.search(r"(?:use|write|in|exactly|at least)\s+(\d+)\s+sentences?", c)
    if m:
        num = int(m.group(1))
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, n=num: (
                    _sentence_count(out) == n,
                    f"Sentence count: {_sentence_count(out)} (need {n})"
                )
            ),
        )

    # Paragraph count: "X paragraphs"
    m = re.search(r"(\d+)\s+paragraphs?", c)
    if m:
        num = int(m.group(1))
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, n=num: (
                    _paragraph_count(out) == n,
                    f"Paragraph count: {_paragraph_count(out)} (need {n})"
                )
            ),
        )

    # Bullet / numbered list
    if any(kw in c for kw in ["bullet", "bulleted", "bullet point"]):
        m = re.search(r"(\d+)\s+bullet", c)
        num = int(m.group(1)) if m else 1
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, n=num: (
                    _bullet_count(out) >= n,
                    f"Bullet count: {_bullet_count(out)} (need >= {n})"
                )
            ),
        )

    if any(kw in c for kw in ["numbered list", "numbered points"]):
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out: (
                    _numbered_count(out) >= 1,
                    "Response must contain a numbered list"
                )
            ),
        )

    # No comma
    if "no comma" in c or "without comma" in c:
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out: ("," not in out, "Response must not contain commas")
            ),
        )

    # Uppercase / lowercase
    if "all caps" in c or "uppercase" in c or "capital letters" in c:
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out: (
                    all(ch.isupper() for ch in out if ch.isalpha()),
                    "Response must be in ALL CAPS"
                )
            ),
        )

    if "lowercase" in c or "lower case" in c:
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out: (
                    all(ch.islower() for ch in out if ch.isalpha()),
                    "Response must be in all lowercase"
                )
            ),
        )

    # Ends with / starts with
    m = re.search(r"end(?:s|ing)? with ['\"](.+?)['\"]", c)
    if m:
        phrase = m.group(1)
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, p=phrase: (
                    out.strip().lower().endswith(p.lower()),
                    f"Must end with '{p}'"
                )
            ),
        )

    m = re.search(r"start(?:s|ing)? with ['\"](.+?)['\"]", c)
    if m:
        phrase = m.group(1)
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, p=phrase: (
                    out.strip().lower().startswith(p.lower()),
                    f"Must start with '{p}'"
                )
            ),
        )

    # Include / mention specific keywords
    m = re.search(r"(?:include|mention|contain|use the (?:word|phrase))\s+['\"](.+?)['\"]", c)
    if m:
        keyword = m.group(1)
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, k=keyword: (
                    k.lower() in out.lower(),
                    f"Must include '{k}'"
                )
            ),
        )

    # Avoid / do not use
    m = re.search(r"(?:avoid|do not use|don't use|without using)\s+['\"](.+?)['\"]", c)
    if m:
        keyword = m.group(1)
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, k=keyword: (
                    k.lower() not in out.lower(),
                    f"Must not contain '{k}'"
                )
            ),
        )

    # Fallback: treat the constraint as a keyword-inclusion check
    # (checks that key noun/verb words from the constraint appear in the response)
    keywords = [w for w in re.findall(r"\b[a-z]{4,}\b", c)
                if w not in {"your", "response", "should", "must", "make", "sure",
                             "that", "with", "have", "this", "text", "write", "about"}]
    if keywords:
        sample_kw = keywords[:3]
        return Requirement(
            constraint,
            validation_fn=simple_validate(
                lambda out, kw=sample_kw: (
                    _has_words(out, kw),
                    f"Response should address: {kw}"
                )
            ),
        )

    # Last resort: always-pass (constraint is unverifiable)
    return Requirement(
        constraint,
        validation_fn=simple_validate(lambda out: (True, "unverifiable — skipped")),
    )


# ── Dataset loading ────────────────────────────────────────────────────────────
#
# FollowBench schema (actual):
#   example_id : int   — 40 unique base prompts
#   category   : str   — "format", "content", "style", "situation", "example", "mixed"
#   source     : str
#   instruction: str   — full prompt with constraints embedded as appended sentences
#   level      : int   — 0 (base, no constraints) through 5 (5 accumulated constraints)
#   target     : str   — empty
#
# Strategy: load level-N rows where category contains "format" (most verifiable),
# then extract added constraints by diffing against the level-0 base instruction
# for the same example_id. Parse each added sentence as a verifiable requirement.

def _extract_added_sentences(base: str, full: str) -> list[str]:
    """Return sentences in `full` that are not in `base`."""
    # Split on sentence boundaries
    def sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    base_sents = set(sentences(base))
    return [s for s in sentences(full) if s not in base_sents]


def load_followbench_tasks(
    level: int = LEVEL,
    sample: int | None = SAMPLE_SIZE,
) -> list[BenchmarkTask]:
    """Load FollowBench from HuggingFace and return BenchmarkTask list.

    Filters to format-category rows at the given level (most verifiable).
    Extracts added constraint sentences by diffing against the level-0 base
    instruction, then converts each to a programmatic Requirement.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading YuxinJiang/FollowBench (level={level}) from HuggingFace...")
    try:
        ds = load_dataset("YuxinJiang/FollowBench", split="train")
    except Exception as e:
        print(f"ERROR loading FollowBench: {e}")
        print("Check: https://huggingface.co/datasets/YuxinJiang/FollowBench")
        sys.exit(1)

    rows = list(ds)

    # Build level-0 base instructions per (example_id, category)
    # Use the shortest level-0 instruction as the base for each example_id
    base_by_id: dict[int, str] = {}
    for r in rows:
        if r["level"] == 0:
            eid = r["example_id"]
            instr = r["instruction"]
            if eid not in base_by_id or len(instr) < len(base_by_id[eid]):
                base_by_id[eid] = instr

    # Filter to target level, prefer format-containing categories
    target_rows = [
        r for r in rows
        if r["level"] == level and "format" in r["category"]
    ]

    # Fallback: any category at target level if no format rows
    if not target_rows:
        target_rows = [r for r in rows if r["level"] == level]

    if sample:
        target_rows = target_rows[:sample]

    tasks: list[BenchmarkTask] = []
    skipped = 0

    for row in target_rows:
        prompt = row["instruction"]
        eid    = row["example_id"]

        if not prompt:
            continue

        # Extract constraint sentences added on top of the base instruction
        base = base_by_id.get(eid, "")
        added = _extract_added_sentences(base, prompt) if base else []

        # Convert added sentences to Requirements
        requirements = []
        for sentence in added:
            req = _constraint_to_requirement(sentence)
            if req is not None:
                requirements.append(req)

        # If no verifiable constraints extracted, try parsing the full prompt
        if not requirements:
            req = _constraint_to_requirement(prompt)
            if req is not None:
                requirements.append(req)

        if not requirements:
            skipped += 1
            continue

        name = prompt[:60].replace("\n", " ").strip() + ("..." if len(prompt) > 60 else "")
        tasks.append(BenchmarkTask(name=name, prompt=prompt, requirements=requirements))

    print(f"Loaded {len(tasks)} tasks at level {level} ({skipped} skipped — no verifiable constraints)")
    return tasks


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    level  = LEVEL
    sample = SAMPLE_SIZE

    if "--level" in sys.argv:
        idx   = sys.argv.index("--level")
        level = int(sys.argv[idx + 1])
    if "--sample" in sys.argv:
        idx    = sys.argv.index("--sample")
        sample = int(sys.argv[idx + 1])

    tasks = load_followbench_tasks(level=level, sample=sample)
    if not tasks:
        print("No tasks loaded. Exiting.")
        sys.exit(1)

    total_reqs = sum(len(t.requirements) for t in tasks)
    print(f"\nFollowBench Level {level}: {len(tasks)} prompts, {total_reqs} total requirements")
    print(f"Model: {MODEL_ID}  |  loop_budget={LOOP_BUDGET}  |  trials={TRIALS}\n")

    results = run_benchmark(tasks, MODEL_ID, LOOP_BUDGET, TRIALS)
    print_benchmark_report(
        results, tasks,
        f"FollowBench (Level {level})",
        MODEL_ID, LOOP_BUDGET, TRIALS,
    )
