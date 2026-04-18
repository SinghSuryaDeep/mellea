# """FollowBench benchmark integration for IVR strategy comparison.

# FollowBench (ACL 2024) evaluates instruction following at 5 difficulty levels
# by progressively adding constraints to a base prompt. Level 1 = 1 constraint,
# Level 5 = 5 accumulated constraints. This tests whether strategies can satisfy
# increasingly complex, layered requirements.

# Constraint types in FollowBench:
#   - Content   : what the response should discuss
#   - Situation : the scenario or persona to adopt
#   - Style     : writing style, tone, register
#   - Format    : structure, length, punctuation (programmatically verifiable)
#   - Example   : follow a given example pattern

# This runner focuses on format constraints (programmatically verifiable) and
# uses heuristic validators for style/content constraints where possible.

# Reference: https://arxiv.org/abs/2310.20410
# Dataset  : YuxinJiang/FollowBench on HuggingFace

# Requirements:
#     pip install datasets

# Usage:
#     cd docs/examples/instruct_validate_repair/benchmarks
#     python followbench_benchmark.py                # level 3 (default)
#     python followbench_benchmark.py --level 5      # hardest level
#     python followbench_benchmark.py --sample 30    # 30 prompts
# """

# from __future__ import annotations

# import re
# import sys

# from mellea.core import Requirement
# from mellea.stdlib.requirements import simple_validate

# from _common import (
#     BenchmarkTask,
#     _parse_judge_response,
#     create_joint_validator,
#     ollama_judge,
#     print_benchmark_report,
#     run_benchmark,
# )

# # ── Configuration ─────────────────────────────────────────────────────────────

# MODEL_ID       = "llama3.2:3b"
# JUDGE_MODEL_ID = "deepseek-r1:8b"
# LOOP_BUDGET    = 4
# TRIALS         = 10
# SAMPLE_SIZE    = 10    # number of FollowBench example_ids to test (all 5 levels each)

# # ── Heuristic validators ───────────────────────────────────────────────────────

# def _wc(text: str) -> int:
#     return len(text.split())

# def _sentence_count(text: str) -> int:
#     return len([s for s in re.split(r"[.!?]+", text) if s.strip()])

# def _paragraph_count(text: str) -> int:
#     return len([p for p in text.split("\n\n") if p.strip()])

# def _bullet_count(text: str) -> int:
#     return len([l for l in text.split("\n") if re.match(r"^\s*[-•*]\s", l)])

# def _numbered_count(text: str) -> int:
#     return len(re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))

# def _has_words(text: str, words: list[str]) -> bool:
#     lower = text.lower()
#     return any(w.lower() in lower for w in words)

# def _lacks_words(text: str, words: list[str]) -> bool:
#     lower = text.lower()
#     return not any(w.lower() in lower for w in words)

# # ── Per-constraint judge validator ────────────────────────────────────────────

# def create_judge_validator(
#     constraint_text: str,
#     judge_model_id: str,
#     evolution_path: list[str] | None = None,
# ) -> "Callable":
#     """Per-constraint judge validator for the repair loop.

#     evolution_path: all accumulated constraint sentences up to this level,
#     shown as context so the judge evaluates each constraint in the right framing.
#     Reason string is always constraint_text so AdaptiveRepair sees stable strings.
#     """
#     def _check(out: str) -> tuple[bool, str]:
#         if evolution_path:
#             context_block = (
#                 "The response was given the following accumulated constraints:\n"
#                 + "\n".join(f"- {c}" for c in evolution_path)
#                 + "\n\n"
#             )
#         else:
#             context_block = ""

#         prompt = (
#             f"{context_block}"
#             f'Given this model response:\n"""\n{out}\n"""\n\n'
#             f"Does it satisfy the following requirement?\n{constraint_text}\n\n"
#             f"Answer with only YES or NO."
#         )
#         try:
#             response = ollama_judge(prompt, judge_model_id)
#         except Exception as e:
#             return (False, f"Judge error: {e}")

#         passed = _parse_judge_response(response)
#         return (passed, "" if passed else constraint_text)

#     return simple_validate(_check)


# # ── Constraint text → Requirement ─────────────────────────────────────────────

# def _constraint_to_requirement(constraint: str) -> Requirement | None:
#     """Convert a FollowBench constraint string into a deterministic Requirement.

#     Only handles strictly programmatic constraints (word count, bullets, case,
#     punctuation, quoted keywords). Returns None for everything else — those
#     constraints are handled by the per-level joint validator in the task builder,
#     which passes the full Level N instruction to the judge rather than extracted
#     sentence fragments.
#     """
#     c = constraint.strip().lower()

#     # Word count: "your response should be X words" / "at least X words" / "no more than X words"
#     m = re.search(r"(?:exactly|at least|at most|no more than|fewer than|more than)\s+(\d+)\s+words?", c)
#     if m:
#         num = int(m.group(1))
#         if "at least" in c or "more than" in c:
#             return Requirement(
#                 constraint,
#                 validation_fn=simple_validate(
#                     lambda out, n=num: (_wc(out) >= n, f"Word count: {_wc(out)} (need >= {n})")
#                 ),
#             )
#         elif "at most" in c or "no more than" in c or "fewer than" in c:
#             return Requirement(
#                 constraint,
#                 validation_fn=simple_validate(
#                     lambda out, n=num: (_wc(out) <= n, f"Word count: {_wc(out)} (need <= {n})")
#                 ),
#             )
#         else:  # exactly
#             return Requirement(
#                 constraint,
#                 validation_fn=simple_validate(
#                     lambda out, n=num: (_wc(out) == n, f"Word count: {_wc(out)} (need exactly {n})")
#                 ),
#             )

#     # Sentence count: "use X sentences" / "in X sentences"
#     m = re.search(r"(?:use|write|in|exactly|at least)\s+(\d+)\s+sentences?", c)
#     if m:
#         num = int(m.group(1))
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out, n=num: (
#                     _sentence_count(out) == n,
#                     f"Sentence count: {_sentence_count(out)} (need {n})"
#                 )
#             ),
#         )

#     # Paragraph count: "X paragraphs"
#     m = re.search(r"(\d+)\s+paragraphs?", c)
#     if m:
#         num = int(m.group(1))
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out, n=num: (
#                     _paragraph_count(out) == n,
#                     f"Paragraph count: {_paragraph_count(out)} (need {n})"
#                 )
#             ),
#         )

#     # Bullet / numbered list
#     if any(kw in c for kw in ["bullet", "bulleted", "bullet point"]):
#         m = re.search(r"(\d+)\s+bullet", c)
#         num = int(m.group(1)) if m else 1
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out, n=num: (
#                     _bullet_count(out) >= n,
#                     f"Bullet count: {_bullet_count(out)} (need >= {n})"
#                 )
#             ),
#         )

#     if any(kw in c for kw in ["numbered list", "numbered points"]):
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out: (
#                     _numbered_count(out) >= 1,
#                     "Response must contain a numbered list"
#                 )
#             ),
#         )

#     # No comma
#     if "no comma" in c or "without comma" in c:
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out: ("," not in out, "Response must not contain commas")
#             ),
#         )

#     # Uppercase / lowercase
#     if "all caps" in c or "uppercase" in c or "capital letters" in c:
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out: (
#                     all(ch.isupper() for ch in out if ch.isalpha()),
#                     "Response must be in ALL CAPS"
#                 )
#             ),
#         )

#     if "lowercase" in c or "lower case" in c:
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out: (
#                     all(ch.islower() for ch in out if ch.isalpha()),
#                     "Response must be in all lowercase"
#                 )
#             ),
#         )

#     # Ends with / starts with
#     m = re.search(r"end(?:s|ing)? with ['\"](.+?)['\"]", c)
#     if m:
#         phrase = m.group(1)
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out, p=phrase: (
#                     out.strip().lower().endswith(p.lower()),
#                     f"Must end with '{p}'"
#                 )
#             ),
#         )

#     m = re.search(r"start(?:s|ing)? with ['\"](.+?)['\"]", c)
#     if m:
#         phrase = m.group(1)
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out, p=phrase: (
#                     out.strip().lower().startswith(p.lower()),
#                     f"Must start with '{p}'"
#                 )
#             ),
#         )

#     # Include / mention specific keywords
#     m = re.search(r"(?:include|mention|contain|use the (?:word|phrase))\s+['\"](.+?)['\"]", c)
#     if m:
#         keyword = m.group(1)
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out, k=keyword: (
#                     k.lower() in out.lower(),
#                     f"Must include '{k}'"
#                 )
#             ),
#         )

#     # Avoid / do not use
#     m = re.search(r"(?:avoid|do not use|don't use|without using)\s+['\"](.+?)['\"]", c)
#     if m:
#         keyword = m.group(1)
#         return Requirement(
#             constraint,
#             validation_fn=simple_validate(
#                 lambda out, k=keyword: (
#                     k.lower() not in out.lower(),
#                     f"Must not contain '{k}'"
#                 )
#             ),
#         )

#     # Non-deterministic constraints (style, tone, content, situation) return None.
#     # They are covered by the per-level joint validator in load_followbench_tasks.
#     return None


# # ── Dataset loading ────────────────────────────────────────────────────────────
# #
# # FollowBench schema (actual):
# #   example_id : int   — 40 unique base prompts
# #   category   : str   — "format", "content", "style", "situation", "example", "mixed"
# #   source     : str
# #   instruction: str   — full prompt with constraints embedded as appended sentences
# #   level      : int   — 0 (base, no constraints) through 5 (5 accumulated constraints)
# #   target     : str   — empty
# #
# # Design: treat the Level N instruction as the full authoritative prompt.
# # Constraints are extracted by diffing Level N against Level 0 (base). Each
# # extracted sentence gets a per-constraint validator for repair loop feedback.
# # A joint validator covers all accumulated constraints simultaneously for
# # authoritative HSR/SSR/CSL scoring.

# # def _extract_added_sentences(base: str, full: str) -> list[str]:
# #     """Return sentences in `full` that are not in `base`."""
# #     def sentences(text: str) -> list[str]:
# #         return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
# #     base_sents = set(sentences(base))
# #     return [s for s in sentences(full) if s not in base_sents]


# def load_followbench_tasks(
#     sample: int | None = SAMPLE_SIZE,
#     judge_model_id: str | None = JUDGE_MODEL_ID,
#     level: int | None = None,
# ) -> list[BenchmarkTask]:
#     """Load FollowBench from HuggingFace and return BenchmarkTask list.

#     Loads all five difficulty levels (1–5) for up to ``sample`` example_ids.
#     For each level:
#       - Per-constraint validators (code or judge) drive the repair loop feedback.
#       - A joint validator evaluates ALL accumulated constraints simultaneously
#         and is used as the authoritative signal for HSR/SSR/CSL scoring.

#     BenchmarkTask.example_id, .level, and .joint_req_index are populated so
#     that post-loop metrics can separate repair signals from joint scoring.
#     """
#     try:
#         from datasets import load_dataset
#     except ImportError:
#         print("ERROR: 'datasets' package not installed. Run: pip install datasets")
#         sys.exit(1)

#     print("Loading YuxinJiang/FollowBench (all levels 1–5) from HuggingFace...")
#     try:
#         ds = load_dataset("YuxinJiang/FollowBench", split="train")
#     except Exception as e:
#         print(f"ERROR loading FollowBench: {e}")
#         print("Check: https://huggingface.co/datasets/YuxinJiang/FollowBench")
#         sys.exit(1)

#     rows = list(ds)

#     # Build level-0 base instructions per example_id (shortest wins)
#     base_by_id: dict[int, str] = {}
#     for r in rows:
#         if r["level"] == 0:
#             eid   = r["example_id"]
#             instr = r["instruction"]
#             if eid not in base_by_id or len(instr) < len(base_by_id[eid]):
#                 base_by_id[eid] = instr

#     # Build level-N instructions per (example_id, level)
#     instr_by_id_level: dict[tuple[int, int], str] = {}
#     for r in rows:
#         lv = r["level"]
#         if 1 <= lv <= 5:
#             instr_by_id_level[(r["example_id"], lv)] = r["instruction"]

#     all_eids = sorted(base_by_id.keys())
#     if sample:
#         all_eids = all_eids[:sample]

#     tasks: list[BenchmarkTask] = []
#     skipped = 0

#     for eid in all_eids:
#         # 🔥 NEW: use full instructions instead of diffed sentences
#         evolution_instructions: list[str] = []

#         for lv in range(1, 6):
#             if level is not None and lv != level:
#                 continue
#             prompt = instr_by_id_level.get((eid, lv))
#             if not prompt:
#                 continue

#             # Accumulate FULL instructions (authoritative constraints)
#             evolution_instructions.append(prompt)

#             requirements: list[Requirement] = []

#             # OPTIONAL: keep deterministic checks (weak, for repair signals)
#             # det_req = _constraint_to_requirement(prompt)
#             det_req = None  # disable for correctness
#             if det_req is not None:
#                 requirements.append(det_req)

#             # Always require at least joint validator (authoritative)
#             joint_req_index = None
#             if judge_model_id is not None:
#                 joint_req = Requirement(
#                     f"All constraints satisfied up to level {lv}",
#                     validation_fn=create_joint_validator(
#                         all_constraints=evolution_instructions,   # 🔥 KEY CHANGE
#                         judge_model_id=judge_model_id,
#                         original_instruction=prompt
#                     ),
#                 )
#                 joint_req_index = len(requirements)
#                 requirements.append(joint_req)

#             if not requirements:
#                 skipped += 1
#                 continue

#             name = f"[L{lv}] " + prompt[:55].replace("\n", " ").strip() + (
#                 "..." if len(prompt) > 55 else ""
#             )

#             tasks.append(BenchmarkTask(
#                 name=name,
#                 prompt=prompt,
#                 requirements=requirements,
#                 example_id=eid,
#                 level=lv,
#                 joint_req_index=joint_req_index,
#             ))
#         # base = base_by_id[eid]
#         # # Accumulated evolution path across levels 1..lv
#         # evolution_path: list[str] = []

#         # for lv in range(1, 6):
#         #     prompt = instr_by_id_level.get((eid, lv))
#         #     if not prompt:
#         #         continue

#         #     # Extract sentences added at this level via diff against base
#         #     added = _extract_added_sentences(base, prompt) if base else []
#         #     evolution_path = evolution_path + added  # accumulate

#         #     requirements: list[Requirement] = []
#         #     for sentence in added:
#         #         req = _constraint_to_requirement(sentence)
#         #         if req is not None:
#         #             requirements.append(req)

#         #     if not requirements:
#         #         skipped += 1
#         #         continue

#         #     # Add joint validator as the final requirement — evaluates all
#         #     # accumulated constraints simultaneously against the full instruction.
#         #     # This is the authoritative signal for HSR/SSR/CSL.
#         #     joint_req_index = None
#         #     if judge_model_id is not None and len(evolution_path) > 1:
#         #         joint_req = Requirement(
#         #             f"All {len(evolution_path)} accumulated constraints satisfied (joint)",
#         #             validation_fn=create_joint_validator(
#         #                 evolution_path, judge_model_id, prompt
#         #             ),
#         #         )
#         #         joint_req_index = len(requirements)
#         #         requirements.append(joint_req)

#         #     name = f"[L{lv}] " + prompt[:55].replace("\n", " ").strip() + (
#         #         "..." if len(prompt) > 55 else ""
#         #     )
#         #     tasks.append(BenchmarkTask(
#         #         name=name,
#         #         prompt=prompt,
#         #         requirements=requirements,
#         #         example_id=eid,
#         #         level=lv,
#         #         joint_req_index=joint_req_index,
#         #     ))

#     judge_note = f"judge={judge_model_id}" if judge_model_id else "no judge"
#     print(
#         f"Loaded {len(tasks)} tasks across levels 1–5 for {len(all_eids)} example_ids "
#         f"({skipped} skipped, {judge_note})"
#     )
#     return tasks


# # ── Official FollowBench metrics (HSR / SSR / CSL) ────────────────────────────

# def compute_followbench_metrics(
#     results: list,
#     tasks: list[BenchmarkTask],
#     strategy_name: str,
# ) -> None:
#     """Compute and print HSR, SSR, CSL for one strategy's results.

#     HSR (Hard Satisfaction Rate): fraction of example_ids where ALL constraints
#         at level N pass, for each N. Requires all requirements to be satisfied.
#     SSR (Soft Satisfaction Rate): average fraction of requirements that pass
#         across all (example_id, level) pairs — partial credit per requirement.
#     CSL (Consistent Satisfaction Level): longest prefix of consecutive levels
#         (starting at 1) where all constraints are satisfied, averaged across
#         example_ids.
#     """
#     from _common import BenchmarkResult

#     # Collect strategy results keyed by task name
#     strategy_results = {r.task_name: r for r in results if r.strategy_name == strategy_name}

#     # Map (example_id, level) → (success_rate, avg_reqs_passed_fraction)
#     example_ids = sorted({t.example_id for t in tasks if t.example_id is not None})
#     levels = sorted({t.level for t in tasks if t.level is not None})

#     # success_rate[eid][lv] = fraction of trials that fully passed
#     # req_fraction[eid][lv] = avg fraction of requirements that passed on final attempt
#     success_by_el: dict[tuple[int, int], float] = {}
#     req_frac_by_el: dict[tuple[int, int], float] = {}

#     for task in tasks:
#         if task.example_id is None or task.level is None:
#             continue
#         bench = strategy_results.get(task.name)
#         if bench is None:
#             continue
#         key = (task.example_id, task.level)

#         # HSR uses joint validator result if available; falls back to success_rate
#         if task.joint_req_index is not None:
#             jidx = task.joint_req_index
#             joint_passes = [
#                 trial.final_req_results[jidx]
#                 for trial in bench.trials
#                 if jidx < len(trial.final_req_results)
#             ]
#             success_by_el[key] = sum(joint_passes) / len(joint_passes) if joint_passes else 0.0
#         else:
#             success_by_el[key] = bench.success_rate

#         # SSR uses per-constraint results only (exclude joint validator)
#         fracs = []
#         for trial in bench.trials:
#             per_constraint = (
#                 trial.final_req_results[:task.joint_req_index]
#                 if task.joint_req_index is not None
#                 else trial.final_req_results
#             )
#             if per_constraint:
#                 fracs.append(sum(per_constraint) / len(per_constraint))
#         req_frac_by_el[key] = sum(fracs) / len(fracs) if fracs else 0.0

#     # HSR per level
#     hsr: dict[int, float] = {}
#     for lv in levels:
#         vals = [success_by_el.get((eid, lv), 0.0) for eid in example_ids]
#         hsr[lv] = sum(vals) / len(vals) if vals else 0.0

#     # SSR: average req fraction across all (eid, level) pairs
#     all_fracs = list(req_frac_by_el.values())
#     ssr = sum(all_fracs) / len(all_fracs) if all_fracs else 0.0

#     # CSL: longest consecutive run of passing levels starting from L1 per example_id
#     csl_scores = []
#     for eid in example_ids:
#         csl = 0
#         for lv in range(1, max(levels) + 1):
#             if success_by_el.get((eid, lv), 0.0) >= 0.5:  # majority of trials pass
#                 csl += 1
#             else:
#                 break
#         csl_scores.append(csl)
#     avg_csl = sum(csl_scores) / len(csl_scores) if csl_scores else 0.0

#     print(f"\n  {strategy_name} — Official FollowBench Metrics:")
#     hsr_str = "  ".join(f"L{lv}:{hsr[lv]*100:.0f}%" for lv in levels)
#     print(f"    HSR per level : {hsr_str}")
#     print(f"    SSR (soft)    : {ssr*100:.1f}%")
#     print(f"    Avg CSL       : {avg_csl:.2f} / {max(levels)}")


# # ── Entry point ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     sample = SAMPLE_SIZE
#     judge  = JUDGE_MODEL_ID

#     if "--sample" in sys.argv:
#         idx    = sys.argv.index("--sample")
#         sample = int(sys.argv[idx + 1])
#     if "--no-judge" in sys.argv:
#         judge = None

#     tasks = load_followbench_tasks(sample=sample, judge_model_id=judge)
#     if not tasks:
#         print("No tasks loaded. Exiting.")
#         sys.exit(1)

#     total_reqs = sum(len(t.requirements) for t in tasks)
#     levels_present = sorted({t.level for t in tasks if t.level})
#     print(f"\nFollowBench levels {levels_present}: {len(tasks)} tasks, {total_reqs} total requirements")
#     print(f"Model: {MODEL_ID}  |  loop_budget={LOOP_BUDGET}  |  trials={TRIALS}\n")

#     results = run_benchmark(tasks, MODEL_ID, LOOP_BUDGET, TRIALS)
#     print_benchmark_report(results, tasks, "FollowBench (all levels)", MODEL_ID, LOOP_BUDGET, TRIALS)

#     print("\n" + "=" * 80)
#     print("OFFICIAL FOLLOWBENCH METRICS (HSR / SSR / CSL)")
#     print("=" * 80)
#     for strategy_name in ["RejectionSampling", "RepairTemplate", "MultiTurn", "AdaptiveRepair"]:
#         compute_followbench_metrics(results, tasks, strategy_name)




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

from _common import (
    BenchmarkTask,
    _parse_judge_response,
    create_joint_validator,
    ollama_judge,
    print_benchmark_report,
    run_benchmark,
)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID       = "llama3.2:3b"
JUDGE_MODEL_ID = "deepseek-r1:8b"
LOOP_BUDGET    = 4
TRIALS         = 10
SAMPLE_SIZE    = 10    # number of FollowBench example_ids to test (all 5 levels each)

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

# ── Per-constraint judge validator ────────────────────────────────────────────

def create_judge_validator(
    constraint_text: str,
    judge_model_id: str,
    evolution_path: list[str] | None = None,
) -> "Callable":
    """Per-constraint judge validator for the repair loop.

    evolution_path: all accumulated constraint sentences up to this level,
    shown as context so the judge evaluates each constraint in the right framing.
    Reason string is always constraint_text so AdaptiveRepair sees stable strings.
    """
    def _check(out: str) -> tuple[bool, str]:
        if evolution_path:
            context_block = (
                "The response was given the following accumulated constraints:\n"
                + "\n".join(f"- {c}" for c in evolution_path)
                + "\n\n"
            )
        else:
            context_block = ""

        prompt = (
            f"{context_block}"
            f'Given this model response:\n"""\n{out}\n"""\n\n'
            f"Does it satisfy the following requirement?\n{constraint_text}\n\n"
            f"Answer with only YES or NO."
        )
        try:
            response = ollama_judge(prompt, judge_model_id)
        except Exception as e:
            return (False, f"Judge error: {e}")

        passed = _parse_judge_response(response)
        return (passed, "" if passed else constraint_text)

    return simple_validate(_check)


# ── Constraint text → Requirement ─────────────────────────────────────────────

def _constraint_to_requirement(constraint: str) -> Requirement | None:
    """Convert a FollowBench constraint string into a deterministic Requirement.

    Only handles strictly programmatic constraints (word count, bullets, case,
    punctuation, quoted keywords). Returns None for everything else — those
    constraints are handled by the per-level joint validator in the task builder,
    which passes the full Level N instruction to the judge rather than extracted
    sentence fragments.
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

    # Non-deterministic constraints (style, tone, content, situation) return None.
    # They are covered by the per-level joint validator in load_followbench_tasks.
    return None


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
# Design: treat the Level N instruction as the full authoritative prompt.
# Constraints are extracted by diffing Level N against Level 0 (base). Each
# extracted sentence gets a per-constraint validator for repair loop feedback.
# A joint validator covers all accumulated constraints simultaneously for
# authoritative HSR/SSR/CSL scoring.

# def _extract_added_sentences(base: str, full: str) -> list[str]:
#     """Return sentences in `full` that are not in `base`."""
#     def sentences(text: str) -> list[str]:
#         return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
#     base_sents = set(sentences(base))
#     return [s for s in sentences(full) if s not in base_sents]


def load_followbench_tasks(
    sample: int | None = SAMPLE_SIZE,
    judge_model_id: str | None = JUDGE_MODEL_ID,
    level: int | None = None,
) -> list[BenchmarkTask]:
    """Load FollowBench from HuggingFace and return BenchmarkTask list.

    Loads all five difficulty levels (1–5) for up to ``sample`` example_ids.
    For each level:
      - Per-constraint validators (code or judge) drive the repair loop feedback.
      - A joint validator evaluates ALL accumulated constraints simultaneously
        and is used as the authoritative signal for HSR/SSR/CSL scoring.

    BenchmarkTask.example_id, .level, and .joint_req_index are populated so
    that post-loop metrics can separate repair signals from joint scoring.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading YuxinJiang/FollowBench (all levels 1–5) from HuggingFace...")
    try:
        ds = load_dataset("YuxinJiang/FollowBench", split="train")
    except Exception as e:
        print(f"ERROR loading FollowBench: {e}")
        print("Check: https://huggingface.co/datasets/YuxinJiang/FollowBench")
        sys.exit(1)

    rows = list(ds)

    # Build level-0 base instructions per example_id (shortest wins)
    base_by_id: dict[int, str] = {}
    for r in rows:
        if r["level"] == 0:
            eid   = r["example_id"]
            instr = r["instruction"]
            if eid not in base_by_id or len(instr) < len(base_by_id[eid]):
                base_by_id[eid] = instr

    # Build level-N instructions per (example_id, level)
    instr_by_id_level: dict[tuple[int, int], str] = {}
    for r in rows:
        lv = r["level"]
        if 1 <= lv <= 5:
            instr_by_id_level[(r["example_id"], lv)] = r["instruction"]

    all_eids = sorted(base_by_id.keys())
    if sample:
        all_eids = all_eids[:sample]

    tasks: list[BenchmarkTask] = []
    skipped = 0

    for eid in all_eids:
        # 🔥 NEW: use full instructions instead of diffed sentences
        evolution_instructions: list[str] = []

        for lv in range(1, 6):
            if level is not None and lv != level:
                continue
            prompt = instr_by_id_level.get((eid, lv))
            if not prompt:
                continue

            # Accumulate FULL instructions (authoritative constraints)
            evolution_instructions.append(prompt)

            requirements: list[Requirement] = []

            # OPTIONAL: keep deterministic checks (weak, for repair signals)
            det_req = _constraint_to_requirement(prompt)
            #det_req = None  # disable for correctness
            if det_req is not None:
                requirements.append(det_req)

            # Always require at least joint validator (authoritative)
            joint_req_index = None
            if judge_model_id is not None:
                joint_req = Requirement(
                    f"All constraints satisfied up to level {lv}",
                    validation_fn=create_joint_validator(
                        all_constraints=evolution_instructions,   # 🔥 KEY CHANGE
                        judge_model_id=judge_model_id,
                        original_instruction=prompt
                    ),
                )
                joint_req_index = len(requirements)
                requirements.append(joint_req)

            if not requirements:
                skipped += 1
                continue

            name = f"[L{lv}] " + prompt[:55].replace("\n", " ").strip() + (
                "..." if len(prompt) > 55 else ""
            )

            tasks.append(BenchmarkTask(
                name=name,
                prompt=prompt,
                requirements=requirements,
                example_id=eid,
                level=lv,
                joint_req_index=joint_req_index,
            ))
        # base = base_by_id[eid]
        # # Accumulated evolution path across levels 1..lv
        # evolution_path: list[str] = []

        # for lv in range(1, 6):
        #     prompt = instr_by_id_level.get((eid, lv))
        #     if not prompt:
        #         continue

        #     # Extract sentences added at this level via diff against base
        #     added = _extract_added_sentences(base, prompt) if base else []
        #     evolution_path = evolution_path + added  # accumulate

        #     requirements: list[Requirement] = []
        #     for sentence in added:
        #         req = _constraint_to_requirement(sentence)
        #         if req is not None:
        #             requirements.append(req)

        #     if not requirements:
        #         skipped += 1
        #         continue

        #     # Add joint validator as the final requirement — evaluates all
        #     # accumulated constraints simultaneously against the full instruction.
        #     # This is the authoritative signal for HSR/SSR/CSL.
        #     joint_req_index = None
        #     if judge_model_id is not None and len(evolution_path) > 1:
        #         joint_req = Requirement(
        #             f"All {len(evolution_path)} accumulated constraints satisfied (joint)",
        #             validation_fn=create_joint_validator(
        #                 evolution_path, judge_model_id, prompt
        #             ),
        #         )
        #         joint_req_index = len(requirements)
        #         requirements.append(joint_req)

        #     name = f"[L{lv}] " + prompt[:55].replace("\n", " ").strip() + (
        #         "..." if len(prompt) > 55 else ""
        #     )
        #     tasks.append(BenchmarkTask(
        #         name=name,
        #         prompt=prompt,
        #         requirements=requirements,
        #         example_id=eid,
        #         level=lv,
        #         joint_req_index=joint_req_index,
        #     ))

    judge_note = f"judge={judge_model_id}" if judge_model_id else "no judge"
    level_str = f"[{level}]" if level else "[1–5]"
    print(
        f"Loaded {len(tasks)} tasks across levels {level_str} for {len(all_eids)} example_ids "
        f"({skipped} skipped, {judge_note})"
    )
    return tasks


# ── Official FollowBench metrics (HSR / SSR / CSL) ────────────────────────────

def compute_followbench_metrics(
    results: list,
    tasks: list[BenchmarkTask],
    strategy_name: str,
) -> None:
    """Compute and print HSR, SSR, CSL for one strategy's results.

    HSR (Hard Satisfaction Rate): fraction of example_ids where ALL constraints
        at level N pass, for each N. Requires all requirements to be satisfied.
    SSR (Soft Satisfaction Rate): average fraction of requirements that pass
        across all (example_id, level) pairs — partial credit per requirement.
    CSL (Consistent Satisfaction Level): longest prefix of consecutive levels
        (starting at 1) where all constraints are satisfied, averaged across
        example_ids.
    """
    from _common import BenchmarkResult

    # Collect strategy results keyed by task name
    strategy_results = {r.task_name: r for r in results if r.strategy_name == strategy_name}

    # Map (example_id, level) → (success_rate, avg_reqs_passed_fraction)
    example_ids = sorted({t.example_id for t in tasks if t.example_id is not None})
    levels = sorted({t.level for t in tasks if t.level is not None})

    # success_rate[eid][lv] = fraction of trials that fully passed
    # req_fraction[eid][lv] = avg fraction of requirements that passed on final attempt
    success_by_el: dict[tuple[int, int], float] = {}
    req_frac_by_el: dict[tuple[int, int], float] = {}

    for task in tasks:
        if task.example_id is None or task.level is None:
            continue
        bench = strategy_results.get(task.name)
        if bench is None:
            continue
        key = (task.example_id, task.level)

        # HSR uses joint validator result if available; falls back to success_rate
        if task.joint_req_index is not None:
            jidx = task.joint_req_index
            joint_passes = [
                trial.final_req_results[jidx]
                for trial in bench.trials
                if jidx < len(trial.final_req_results)
            ]
            success_by_el[key] = sum(joint_passes) / len(joint_passes) if joint_passes else 0.0
        else:
            success_by_el[key] = bench.success_rate

        # SSR uses per-constraint results only (exclude joint validator)
        fracs = []
        for trial in bench.trials:
            per_constraint = (
                trial.final_req_results[:task.joint_req_index]
                if task.joint_req_index is not None
                else trial.final_req_results
            )
            if per_constraint:
                fracs.append(sum(per_constraint) / len(per_constraint))
        req_frac_by_el[key] = sum(fracs) / len(fracs) if fracs else 0.0

    # HSR per level
    hsr: dict[int, float] = {}
    for lv in levels:
        vals = [success_by_el.get((eid, lv), 0.0) for eid in example_ids]
        hsr[lv] = sum(vals) / len(vals) if vals else 0.0

    # SSR: average req fraction across all (eid, level) pairs
    all_fracs = list(req_frac_by_el.values())
    ssr = sum(all_fracs) / len(all_fracs) if all_fracs else 0.0

    # CSL: longest consecutive run of passing levels starting from L1 per example_id
    csl_scores = []
    for eid in example_ids:
        csl = 0
        for lv in range(1, max(levels) + 1):
            if success_by_el.get((eid, lv), 0.0) >= 0.5:  # majority of trials pass
                csl += 1
            else:
                break
        csl_scores.append(csl)
    avg_csl = sum(csl_scores) / len(csl_scores) if csl_scores else 0.0

    print(f"\n  {strategy_name} — Official FollowBench Metrics:")
    hsr_str = "  ".join(f"L{lv}:{hsr[lv]*100:.0f}%" for lv in levels)
    print(f"    HSR per level : {hsr_str}")
    print(f"    SSR (soft)    : {ssr*100:.1f}%")
    print(f"    Avg CSL       : {avg_csl:.2f} / {max(levels)}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = SAMPLE_SIZE
    judge  = JUDGE_MODEL_ID
    level  = None   # ✅ ADD THIS

    if "--sample" in sys.argv:
        idx    = sys.argv.index("--sample")
        sample = int(sys.argv[idx + 1])
    if "--no-judge" in sys.argv:
        judge = None
    if "--level" in sys.argv:
        idx = sys.argv.index("--level")
        level = int(sys.argv[idx + 1])

    tasks = load_followbench_tasks(sample=sample, judge_model_id=judge, level=level)
    if not tasks:
        print("No tasks loaded. Exiting.")
        sys.exit(1)

    total_reqs = sum(len(t.requirements) for t in tasks)
    levels_present = sorted({t.level for t in tasks if t.level})
    print(f"\nFollowBench levels {levels_present}: {len(tasks)} tasks, {total_reqs} total requirements")
    print(f"Model: {MODEL_ID}  |  loop_budget={LOOP_BUDGET}  |  trials={TRIALS}\n")

    results = run_benchmark(tasks, MODEL_ID, LOOP_BUDGET, TRIALS)
    print_benchmark_report(results, tasks, "FollowBench (all levels)", MODEL_ID, LOOP_BUDGET, TRIALS)

    print("\n" + "=" * 80)
    print("OFFICIAL FOLLOWBENCH METRICS (HSR / SSR / CSL)")
    print("=" * 80)
    for strategy_name in ["RejectionSampling", "RepairTemplate", "MultiTurn", "AdaptiveRepair"]:
        compute_followbench_metrics(results, tasks, strategy_name)
