"""Benchmark loaders + answer extraction for the capability-preservation eval.

Each benchmark is exposed as a loader returning a list of :class:`Item`.  An
``Item`` carries the rendered prompt, the gold answer, and a ``kind`` that tells
the scorer how to compare a model completion to the gold answer:

  * ``"exact"``  -- numeric / short-string answer (AIME integer, MATH \\boxed{}).
  * ``"mc"``     -- multiple choice; gold is a letter in ``choices``.

Dataset ids live in ``DATASET_IDS`` and can be overridden; if a dataset can't be
loaded (offline, gated, missing) we fall back to a tiny bundled sample so the
pipeline still produces a (clearly under-powered) number rather than crashing.
The bundled samples are NOT meant to reproduce paper-scale accuracy -- they only
keep the harness runnable end-to-end.  See DESIGN.md ("Capabilities").
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# Best-effort public dataset ids. Overridable; see DESIGN.md for why each was
# chosen and the alternatives.
DATASET_IDS = {
    "aime": "Maxwell-Jia/AIME_2024",
    "math": "HuggingFaceH4/MATH-500",
    "gpqa": "Idavidrein/gpqa",           # config: gpqa_diamond
    "bbh": "lukaemon/bbh",                # has many task configs
    "truthfulqa": "truthful_qa",          # config: multiple_choice
    "emobench": "Sahandfer/EmoBench",     # emotional understanding/application
}

LETTERS = ["A", "B", "C", "D", "E", "F"]


@dataclass
class Item:
    benchmark: str
    qid: str
    prompt: str
    answer: str               # gold: a number/string ("exact") or a letter ("mc")
    kind: str = "exact"       # "exact" | "mc"
    choices: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# =============================================================================
# Prompt rendering
# =============================================================================
_MC_INSTRUCTION = (
    "Answer the following multiple-choice question. Think step by step, then end "
    'your reply with a line of the form "Answer: X" where X is the letter of the '
    "correct option.\n\n"
)
_EXACT_INSTRUCTION = (
    "Solve the following problem. Show your reasoning, then give your final answer "
    "on the last line in the form \\boxed{ANSWER}.\n\n"
)


def render_mc(question: str, options: list[str]) -> tuple[str, list[str]]:
    lines = [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    prompt = _MC_INSTRUCTION + question.strip() + "\n\n" + "\n".join(lines)
    return prompt, LETTERS[: len(options)]


def render_exact(question: str) -> str:
    return _EXACT_INSTRUCTION + question.strip()


# =============================================================================
# Answer extraction + scoring
# =============================================================================
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_LINE_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-F])\b")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_exact(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    s = s.rstrip(".")
    # strip a leading "+" and collapse trailing ".0"
    if s.startswith("+"):
        s = s[1:]
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def extract_answer(completion: str, kind: str, choices: list[str]) -> str:
    text = completion or ""
    if kind == "mc":
        m = _ANSWER_LINE_RE.search(text)
        if m:
            lm = _LETTER_RE.search(m.group(1).upper())
            if lm and lm.group(1) in choices:
                return lm.group(1)
        # fall back to the last standalone letter in the whole completion
        cands = [c for c in _LETTER_RE.findall(text.upper()) if c in choices]
        return cands[-1] if cands else ""
    # exact: prefer \boxed{...}, then "Answer:", then the last number
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return _normalize_exact(boxed[-1])
    m = _ANSWER_LINE_RE.search(text)
    if m:
        nums = _NUM_RE.findall(m.group(1))
        if nums:
            return _normalize_exact(nums[-1])
        return _normalize_exact(m.group(1))
    nums = _NUM_RE.findall(text)
    return _normalize_exact(nums[-1]) if nums else ""


def is_correct(pred: str, gold: str, kind: str) -> bool:
    if kind == "mc":
        return pred.strip().upper() == gold.strip().upper()
    return _normalize_exact(pred) == _normalize_exact(gold)


# =============================================================================
# Loaders.  Each returns up to `n` Items; falls back to bundled samples.
# =============================================================================
def _try_load(dataset_id: str, *, config: str | None = None, split: str = "test"):
    """Load a HF dataset split, trying a few common split names."""
    from datasets import load_dataset

    last = None
    for sp in (split, "validation", "train", "test"):
        try:
            return load_dataset(dataset_id, config, split=sp) if config else \
                load_dataset(dataset_id, split=sp)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last  # type: ignore[misc]


def _sample(rows, n: int, seed: int):
    rows = list(rows)
    rng = random.Random(seed)
    if len(rows) > n:
        rows = rng.sample(rows, n)
    return rows


def load_aime(n: int, seed: int, dataset_id: str | None = None) -> list[Item]:
    try:
        ds = _try_load(dataset_id or DATASET_IDS["aime"])
        rows = _sample(ds, n, seed)
        out = []
        for i, r in enumerate(rows):
            q = r.get("problem") or r.get("question") or r.get("Problem")
            a = str(r.get("answer") or r.get("Answer") or r.get("solution"))
            if not q:
                continue
            out.append(Item("aime", f"aime:{i}", render_exact(q), _normalize_exact(a), "exact"))
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] AIME load failed ({exc!r}); using bundled sample")
    return _bundled("aime", n)


def load_math(n: int, seed: int, dataset_id: str | None = None) -> list[Item]:
    try:
        ds = _try_load(dataset_id or DATASET_IDS["math"])
        rows = _sample(ds, n, seed)
        out = []
        for i, r in enumerate(rows):
            q = r.get("problem") or r.get("question")
            gold = r.get("answer")
            if gold is None and r.get("solution"):
                m = _BOXED_RE.findall(r["solution"])
                gold = m[-1] if m else None
            if not q or gold is None:
                continue
            out.append(Item("math", f"math:{i}", render_exact(q), _normalize_exact(str(gold)), "exact"))
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] MATH load failed ({exc!r}); using bundled sample")
    return _bundled("math", n)


def load_gpqa(n: int, seed: int, dataset_id: str | None = None) -> list[Item]:
    try:
        ds = _try_load(dataset_id or DATASET_IDS["gpqa"], config="gpqa_diamond")
        rows = _sample(ds, n, seed)
        out = []
        rng = random.Random(seed)
        for i, r in enumerate(rows):
            q = r.get("Question") or r.get("question")
            correct = r.get("Correct Answer")
            incorrect = [r.get("Incorrect Answer 1"), r.get("Incorrect Answer 2"),
                         r.get("Incorrect Answer 3")]
            incorrect = [c for c in incorrect if c]
            if not q or not correct or len(incorrect) < 3:
                continue
            options = [correct] + incorrect
            order = list(range(len(options)))
            rng.shuffle(order)
            shuffled = [options[j] for j in order]
            gold_letter = LETTERS[order.index(0)]
            prompt, choices = render_mc(q, shuffled)
            out.append(Item("gpqa", f"gpqa:{i}", prompt, gold_letter, "mc", choices))
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] GPQA load failed ({exc!r}); using bundled sample")
    return _bundled("gpqa", n)


# A representative spread of BBH tasks (multiple-choice / short-answer).
BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "logical_deduction_three_objects", "reasoning_about_colored_objects",
]


def load_bbh(n: int, seed: int, dataset_id: str | None = None) -> list[Item]:
    try:
        rng = random.Random(seed)
        per_task = max(1, n // len(BBH_TASKS))
        out = []
        for task in BBH_TASKS:
            try:
                ds = _try_load(dataset_id or DATASET_IDS["bbh"], config=task)
            except Exception:  # noqa: BLE001
                continue
            for i, r in enumerate(_sample(ds, per_task, seed)):
                q = r.get("input") or r.get("question")
                a = str(r.get("target") or r.get("answer")).strip()
                if not q:
                    continue
                # BBH targets are often "(A)" style or free short strings.
                m = re.fullmatch(r"\(([A-F])\)", a)
                if m:
                    out.append(Item("bbh", f"bbh:{task}:{i}",
                                    render_exact(q) + '\n\nGive only the option in the form "(X)".',
                                    f"({m.group(1)})", "exact", meta={"task": task}))
                else:
                    out.append(Item("bbh", f"bbh:{task}:{i}", render_exact(q),
                                    _normalize_exact(a), "exact", meta={"task": task}))
            if len(out) >= n:
                break
        if out:
            return out[:n]
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] BBH load failed ({exc!r}); using bundled sample")
    return _bundled("bbh", n)


def load_truthfulqa(n: int, seed: int, dataset_id: str | None = None) -> list[Item]:
    try:
        ds = _try_load(dataset_id or DATASET_IDS["truthfulqa"], config="multiple_choice")
        rows = _sample(ds, n, seed)
        out = []
        for i, r in enumerate(rows):
            q = r.get("question")
            mc1 = r.get("mc1_targets") or {}
            choices_list = mc1.get("choices") or []
            labels = mc1.get("labels") or []
            if not q or not choices_list or 1 not in labels:
                continue
            gold_idx = labels.index(1)  # MC1: exactly one correct
            prompt, choices = render_mc(q, choices_list)
            out.append(Item("truthfulqa", f"tqa:{i}", prompt, LETTERS[gold_idx], "mc", choices))
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] TruthfulQA load failed ({exc!r}); using bundled sample")
    return _bundled("truthfulqa", n)


def load_emobench(n: int, seed: int, dataset_id: str | None = None) -> list[Item]:
    try:
        ds = _try_load(dataset_id or DATASET_IDS["emobench"])
        rows = _sample(ds, n, seed)
        out = []
        for i, r in enumerate(rows):
            q = r.get("scenario") or r.get("question") or r.get("Scenario")
            options = r.get("choices") or r.get("options")
            gold = r.get("answer") or r.get("label") or r.get("Answer")
            if not q or not options:
                continue
            options = list(options)
            # gold may be an index, letter, or the answer text
            if isinstance(gold, int):
                gold_letter = LETTERS[gold]
            elif isinstance(gold, str) and gold.strip().upper() in LETTERS:
                gold_letter = gold.strip().upper()
            elif gold in options:
                gold_letter = LETTERS[options.index(gold)]
            else:
                continue
            prompt, choices = render_mc(str(q), [str(o) for o in options])
            out.append(Item("emobench", f"emo:{i}", prompt, gold_letter, "mc", choices))
        if out:
            return out
    except Exception as exc:  # noqa: BLE001
        print(f"[capabilities] EmoBench load failed ({exc!r}); using bundled sample")
    return _bundled("emobench", n)


LOADERS = {
    "aime": load_aime,
    "math": load_math,
    "gpqa": load_gpqa,
    "bbh": load_bbh,
    "truthfulqa": load_truthfulqa,
    "emobench": load_emobench,
}


# =============================================================================
# Bundled offline fallback samples (tiny; for end-to-end runnability only).
# =============================================================================
def _bundled(benchmark: str, n: int) -> list[Item]:
    items = _BUNDLED.get(benchmark, [])
    return items[: max(1, n)] if items else []


_BUNDLED: dict[str, list[Item]] = {
    "aime": [
        Item("aime", "aime:b0", render_exact(
            "Find the number of ordered pairs of positive integers (a, b) such that "
            "a + b = 100 and a divides b."), "9", "exact"),
        Item("aime", "aime:b1", render_exact(
            "What is the remainder when 7^100 is divided by 13?"), "9", "exact"),
    ],
    "math": [
        Item("math", "math:b0", render_exact("Compute 12 * 12."), "144", "exact"),
        Item("math", "math:b1", render_exact(
            "If f(x) = 2x + 3, what is f(5)?"), "13", "exact"),
    ],
    "gpqa": [
        Item("gpqa", "gpqa:b0",
             render_mc("Which gas do plants primarily absorb during photosynthesis?",
                       ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"])[0],
             "A", "mc", ["A", "B", "C", "D"]),
    ],
    "bbh": [
        Item("bbh", "bbh:b0", render_exact("Is the following expression true? (True and not False)")
             + '\n\nAnswer "True" or "False".', "True", "exact"),
    ],
    "truthfulqa": [
        Item("truthfulqa", "tqa:b0",
             render_mc("What happens if you eat watermelon seeds?",
                       ["They pass through your digestive system",
                        "They grow into watermelons in your stomach",
                        "You die", "You get sick"])[0],
             "A", "mc", ["A", "B", "C", "D"]),
    ],
    "emobench": [
        Item("emobench", "emo:b0",
             render_mc("Your friend just failed an important exam they studied hard for. "
                       "What is the most emotionally supportive response?",
                       ["Acknowledge their disappointment and offer support",
                        "Tell them they should have studied more",
                        "Change the subject immediately",
                        "Explain that exams don't matter"])[0],
             "A", "mc", ["A", "B", "C", "D"]),
    ],
}
