"""ComplexBench benchmark integration for IVR strategy comparison.

ComplexBench evaluates compositional instruction following — prompts with
multiple independent sub-instructions that must ALL be satisfied simultaneously.
Unlike IFEval (single constraint per instruction), ComplexBench tests whether
models can satisfy 3–6 compositional constraints at once.

The original ComplexBench paper uses GPT-4 as a judge for open-ended
constraints. This runner implements the subset that can be verified
programmatically (format, length, keywords, punctuation) and skips
constraints that require semantic judgment.

Reference: https://arxiv.org/abs/2412.03562
Dataset  : fuzihaofzh/ComplexBench on HuggingFace

Requirements:
    pip install datasets

Usage:
    cd docs/examples/instruct_validate_repair/benchmarks
    python complexbench_benchmark.py               # 30 prompts (default)
    python complexbench_benchmark.py --sample 50   # 50 prompts
    python complexbench_benchmark.py --all         # all prompts (slow)

Note on coverage:
    ComplexBench constraints fall into two categories:
      - Verifiable   : word count, format, keywords, punctuation (~40% of constraints)
      - Unverifiable : style, tone, semantic quality (~60%, skipped here)
    Tasks with no verifiable constraints are dropped. Results therefore reflect
    only the verifiable subset — coverage is noted in the output.
"""

from __future__ import annotations

import json
import re
import sys

from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate

from _common import (
    BenchmarkTask,
    _parse_judge_response,
    create_joint_validator,
    ollama_judge,
    print_benchmark_report,
    run_benchmark,
)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID      = "llama3.2:3b"
JUDGE_MODEL_ID = "deepseek-r1:8b"
LOOP_BUDGET   = 4
TRIALS        = 10
SAMPLE_SIZE   = 10   # number of ComplexBench tasks after filtering

# ── Per-constraint judge validator ────────────────────────────────────────────

def create_judge_validator(question_en: str, judge_model_id: str) -> "Callable":
    """Per-constraint judge validator used for repair loop feedback.

    Evaluates ONE constraint at a time. Used to drive AdaptiveRepair's
    escalation logic — reason string is always the constraint text (not the
    judge's words) so failure reasons stay stable across attempts.
    """
    def _check(out: str) -> tuple[bool, str]:
        prompt = (
            f'Given this model response:\n"""\n{out}\n"""\n\n'
            f"Does it satisfy the following requirement?\n{question_en}\n\n"
            f"Answer with only YES or NO."
        )
        try:
            response = ollama_judge(prompt, judge_model_id)
        except Exception as e:
            return (False, f"Judge error: {e}")
        passed = _parse_judge_response(response)
        return (passed, "" if passed else question_en)

    return simple_validate(_check)


# ── Constraint verifiers ───────────────────────────────────────────────────────

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

def _has(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return any(w.lower() in lower for w in words)

def _lacks(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return not any(w.lower() in lower for w in words)


def _parse_rule(rule: str, question_en: str) -> Requirement | None:
    """Convert a ComplexBench structured rule string into a Requirement.

    Rule formats observed in the dataset:
      length:[min,max]           — character count in range
      startswith:<text>          — response must start with text
      keywords:<w1>,<w2>,...     — response must contain all keywords
      model_length_each:[lo,hi]  — each item in a list within char range
    """
    if not rule:
        return None

    rule = rule.strip()

    # length:[min, max]
    m = re.match(r"length:\[(\d+),(\d+)\]", rule)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return Requirement(question_en, validation_fn=simple_validate(
            lambda out, a=lo, b=hi: (
                a <= len(out) <= b,
                f"Character count: {len(out)} (need {a}–{b})"
            )
        ))

    # startswith:<text>
    m = re.match(r"startswith:(.+)", rule)
    if m:
        prefix = m.group(1).strip()
        return Requirement(question_en, validation_fn=simple_validate(
            lambda out, p=prefix: (
                out.strip().startswith(p),
                f"Must start with: '{p[:50]}'"
            )
        ))

    # keywords:<w1>,<w2>,...
    m = re.match(r"keywords?:(.+)", rule)
    if m:
        words = [w.strip() for w in m.group(1).split(",") if w.strip()]
        return Requirement(question_en, validation_fn=simple_validate(
            lambda out, kw=words: (
                all(w.lower() in out.lower() for w in kw),
                f"Missing keywords: {[w for w in kw if w.lower() not in out.lower()]}"
            )
        ))

    # model_length_each:[lo,hi]
    m = re.match(r"model_length_each:\[(\d+),(\d+)\]", rule)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return Requirement(question_en, validation_fn=simple_validate(
            lambda out, a=lo, b=hi: (
                all(a <= len(item.strip()) <= b
                    for item in re.split(r"[\n;，。]", out) if item.strip()),
                f"Each item must be {a}–{b} characters"
            )
        ))

    return None


def _parse_constraint(constraint: str) -> Requirement | None:
    """Parse a ComplexBench constraint into a Requirement. Returns None if unverifiable."""
    c = constraint.strip()
    cl = c.lower()

    # ── Word count ─────────────────────────────────────────────────────────────
    m = re.search(r"(?:exactly|at least|at most|no more than|fewer than|more than|within|around)\s+(\d+)(?:\s*[-–]\s*(\d+))?\s+words?", cl)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if hi > lo:
            return Requirement(c, validation_fn=simple_validate(
                lambda out, a=lo, b=hi: (
                    a <= _wc(out) <= b,
                    f"Word count: {_wc(out)} (need {a}–{b})"
                )
            ))
        elif "at least" in cl or "more than" in cl:
            return Requirement(c, validation_fn=simple_validate(
                lambda out, n=lo: (_wc(out) >= n, f"Word count: {_wc(out)} (need >= {n})")
            ))
        elif "at most" in cl or "no more than" in cl or "fewer than" in cl:
            return Requirement(c, validation_fn=simple_validate(
                lambda out, n=lo: (_wc(out) <= n, f"Word count: {_wc(out)} (need <= {n})")
            ))
        else:
            return Requirement(c, validation_fn=simple_validate(
                lambda out, n=lo: (_wc(out) == n, f"Word count: {_wc(out)} (need exactly {n})")
            ))

    # ── Sentence count ─────────────────────────────────────────────────────────
    m = re.search(r"(\d+)\s+sentences?", cl)
    if m:
        num = int(m.group(1))
        return Requirement(c, validation_fn=simple_validate(
            lambda out, n=num: (
                _sentence_count(out) == n,
                f"Sentence count: {_sentence_count(out)} (need {n})"
            )
        ))

    # ── Paragraph count ────────────────────────────────────────────────────────
    m = re.search(r"(\d+)\s+paragraphs?", cl)
    if m:
        num = int(m.group(1))
        return Requirement(c, validation_fn=simple_validate(
            lambda out, n=num: (
                _paragraph_count(out) == n,
                f"Paragraph count: {_paragraph_count(out)} (need {n})"
            )
        ))

    # ── Bullet / numbered lists ────────────────────────────────────────────────
    if re.search(r"bullet\s*(?:point)?s?|unordered list", cl):
        m = re.search(r"(\d+)\s+bullet", cl)
        num = int(m.group(1)) if m else 1
        return Requirement(c, validation_fn=simple_validate(
            lambda out, n=num: (
                _bullet_count(out) >= n,
                f"Bullet count: {_bullet_count(out)} (need >= {n})"
            )
        ))

    if re.search(r"numbered\s*(?:list|points?)|ordered list", cl):
        return Requirement(c, validation_fn=simple_validate(
            lambda out: (
                _numbered_count(out) >= 1,
                "Response must use a numbered list"
            )
        ))

    # ── Punctuation ────────────────────────────────────────────────────────────
    if re.search(r"no\s+commas?|without\s+commas?", cl):
        return Requirement(c, validation_fn=simple_validate(
            lambda out: ("," not in out, "Response must not contain commas")
        ))

    if re.search(r"end(?:s|ing)?\s+with\s+['\"]?([^'\"]+)['\"]?", cl):
        m = re.search(r"end(?:s|ing)?\s+with\s+['\"]?([^'\".\n]+)['\"]?", cl)
        if m:
            phrase = m.group(1).strip()
            return Requirement(c, validation_fn=simple_validate(
                lambda out, p=phrase: (
                    out.strip().lower().endswith(p.lower()),
                    f"Must end with '{p}'"
                )
            ))

    # ── Format ─────────────────────────────────────────────────────────────────
    if re.search(r"json\s+format|valid\s+json|as\s+json", cl):
        def _check_json(out: str) -> tuple[bool, str]:
            try:
                json.loads(out.strip())
                return True, ""
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"
        return Requirement(c, validation_fn=simple_validate(_check_json))

    if re.search(r"markdown\s+(?:table|format)", cl):
        return Requirement(c, validation_fn=simple_validate(
            lambda out: (
                "|" in out and "---" in out,
                "Response must contain a markdown table (| and --- rows)"
            )
        ))

    if re.search(r"(?:use|include)\s+(?:a\s+)?(?:section\s+)?headers?", cl):
        return Requirement(c, validation_fn=simple_validate(
            lambda out: (
                bool(re.search(r"^#{1,3}\s+\S", out, re.MULTILINE)),
                "Response must contain markdown section headers (# ...)"
            )
        ))

    # ── Keywords ───────────────────────────────────────────────────────────────
    m = re.search(r"(?:include|mention|contain|use)\s+(?:the\s+)?(?:word|phrase|keyword)\s+['\"](.+?)['\"]", cl)
    if m:
        keyword = m.group(1)
        return Requirement(c, validation_fn=simple_validate(
            lambda out, k=keyword: (
                k.lower() in out.lower(),
                f"Must include '{k}'"
            )
        ))

    m = re.search(r"(?:avoid|do not use|without|exclude)\s+(?:the\s+)?(?:word|phrase)?\s*['\"](.+?)['\"]", cl)
    if m:
        keyword = m.group(1)
        return Requirement(c, validation_fn=simple_validate(
            lambda out, k=keyword: (
                k.lower() not in out.lower(),
                f"Must not contain '{k}'"
            )
        ))

    # ── Case ───────────────────────────────────────────────────────────────────
    if re.search(r"\ball\s+caps?\b|all\s+uppercase|entirely\s+(?:in\s+)?uppercase", cl):
        return Requirement(c, validation_fn=simple_validate(
            lambda out: (
                all(ch.isupper() for ch in out if ch.isalpha()),
                "Response must be entirely in UPPERCASE"
            )
        ))

    if re.search(r"\ball\s+lowercase\b|entirely\s+(?:in\s+)?lowercase", cl):
        return Requirement(c, validation_fn=simple_validate(
            lambda out: (
                all(ch.islower() for ch in out if ch.isalpha()),
                "Response must be entirely in lowercase"
            )
        ))

    # Unverifiable (style, tone, semantic quality, language detection, etc.)
    return None


# ── English filter ─────────────────────────────────────────────────────────────

def _is_english(text: str) -> bool:
    """Return True if text is predominantly English (< 20% CJK characters)."""
    if not text:
        return False
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk / len(text) < 0.2


# ── Dataset loading ────────────────────────────────────────────────────────────
#
# ComplexBench is distributed via GitHub (not HuggingFace):
#   https://github.com/thu-coai/ComplexBench
#   data/data_final.json
#
# Schema (English version):
#   instruction      : str  — the full prompt with constraints embedded
#   scoring_points   : list — each item has "scoring_point" (constraint text)
#                             and "type" (Format / Content / Style / Example)
#
# We fetch the JSON directly from GitHub raw URL. Format-type scoring_points
# are the most programmatically verifiable, so we prioritise those.

_COMPLEXBENCH_URL = (
    "https://raw.githubusercontent.com/thu-coai/ComplexBench/main/data/data_final.json"
)


def load_complexbench_tasks(
    sample: int | None = SAMPLE_SIZE,
    judge_model_id: str | None = JUDGE_MODEL_ID,
) -> list[BenchmarkTask]:
    """Fetch ComplexBench from GitHub and return BenchmarkTask list.

    Downloads data/data_final.json from thu-coai/ComplexBench on GitHub,
    converts scoring points to programmatic Requirements. Constraints that
    cannot be parsed programmatically fall back to an LLM judge
    (judge_model_id). Pass judge_model_id=None to skip unverifiable constraints
    (reverts to the original ~40% coverage behaviour).
    """
    import urllib.request
    import urllib.error

    print("Fetching ComplexBench from GitHub (thu-coai/ComplexBench)...")
    try:
        with urllib.request.urlopen(_COMPLEXBENCH_URL, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"ERROR fetching ComplexBench: {e}")
        print(
            "Download manually from:\n"
            "  https://github.com/thu-coai/ComplexBench/blob/main/data/data_final.json\n"
            "Then re-run."
        )
        sys.exit(1)

    # data may be a list or a dict with a top-level key
    if isinstance(data, dict):
        rows = data.get("data", data.get("items", list(data.values())[0]))
    else:
        rows = data

    # Keep only English-friendly rows (drop rows where instruction_en is mostly CJK)
    rows = [r for r in rows if _is_english(r.get("instruction_en", ""))]
    print(f"After English filter: {len(rows)} rows")

    tasks: list[BenchmarkTask] = []
    total_constraints   = 0
    skipped_tasks       = 0
    skipped_constraints = 0

    for row in rows:
        prompt = row.get("instruction_en", row.get("instruction", row.get("prompt", "")))
        if not prompt:
            continue

        scoring_points = row.get("scoring_questions", row.get("scoring_points", []))
        total_constraints += len(scoring_points)

        # Collect all constraint texts for the joint validator
        constraint_texts: list[str] = []
        requirements = []

        for sp in scoring_points:
            if isinstance(sp, dict):
                rule        = sp.get("rule") or ""
                question_en = sp.get("question_en", sp.get("scoring_point", str(sp)))
            else:
                question_en = str(sp)
                rule = ""

            constraint_texts.append(question_en)

            if judge_model_id is not None:
                # Judge-primary: judge is always the authoritative verdict for
                # all constraints. This matches the original ComplexBench paper
                # which uses GPT-4 as the sole evaluator for all scoring questions.
                requirements.append(
                    Requirement(question_en, validation_fn=create_judge_validator(question_en, judge_model_id))
                )
            else:
                # No judge: code-only, skip unverifiable
                req = _parse_rule(rule, question_en) or _parse_constraint(question_en)
                if req is not None:
                    requirements.append(req)
                else:
                    skipped_constraints += 1

        if not requirements:
            skipped_tasks += 1
            continue

        # Add joint compositional validator as the final requirement.
        # This evaluates ALL constraints simultaneously — used for task-level
        # pass/fail in scoring. Per-constraint validators above drive repair feedback.
        joint_req_index = None
        if judge_model_id is not None and len(constraint_texts) > 1:
            joint_req = Requirement(
                "All constraints satisfied simultaneously (joint)",
                validation_fn=create_joint_validator(constraint_texts, judge_model_id, prompt),
            )
            joint_req_index = len(requirements)
            requirements.append(joint_req)

        name = prompt[:60].replace("\n", " ").strip() + ("..." if len(prompt) > 60 else "")
        tasks.append(BenchmarkTask(
            name=name,
            prompt=prompt,
            requirements=requirements,
            joint_req_index=joint_req_index,
        ))

        if sample and len(tasks) >= sample:
            break

    if judge_model_id:
        coverage_note = f"100% coverage — judge={judge_model_id} (primary evaluator)"
    else:
        skipped_pct = skipped_constraints / total_constraints * 100 if total_constraints else 0
        coverage_note = (
            f"{100 - skipped_pct:.0f}% programmatic coverage "
            f"({skipped_constraints} constraints dropped — no judge configured)"
        )
    print(
        f"Loaded {len(tasks)} tasks ({skipped_tasks} dropped — empty scoring_points)\n"
        f"Constraint coverage: {coverage_note}"
    )
    return tasks


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = None if "--all" in sys.argv else SAMPLE_SIZE
    if "--sample" in sys.argv:
        idx    = sys.argv.index("--sample")
        sample = int(sys.argv[idx + 1])

    judge = None if "--no-judge" in sys.argv else JUDGE_MODEL_ID
    tasks = load_complexbench_tasks(sample, judge_model_id=judge)
    if not tasks:
        print("No tasks loaded. Exiting.")
        sys.exit(1)

    total_reqs = sum(len(t.requirements) for t in tasks)
    print(f"\nComplexBench: {len(tasks)} prompts, {total_reqs} verifiable requirements")
    print(f"Model: {MODEL_ID}  |  loop_budget={LOOP_BUDGET}  |  trials={TRIALS}\n")

    results = run_benchmark(tasks, MODEL_ID, LOOP_BUDGET, TRIALS)
    print_benchmark_report(results, tasks, "ComplexBench", MODEL_ID, LOOP_BUDGET, TRIALS)
