"""Capability-preservation benchmarks (Section 4.2, Figure 7).

We check that DPO/SFT do not degrade capabilities on AIME/MATH (math), GPQA
(science QA), BBH (reasoning), TruthfulQA (truthfulness), and EmoBench
(emotion-related capabilities). Each benchmark is reduced to a small subset and
graded with a simple exact/multiple-choice matcher.

Dataset identifiers are configurable and failures are tolerated (a missing
dataset is reported as skipped rather than crashing the sweep), since the point
is a *relative* comparison of vanilla vs finetuned Gemma on whatever subset is
available locally.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..cache import JsonCache
from ..config import Config
from ..models import get_client


@dataclass
class BenchmarkResult:
    model_key: str
    benchmark: str
    n: int
    accuracy: float
    skipped: bool = False
    note: str = ""


# --------------------------------------------------------------------------- #
# answer extraction / grading helpers
# --------------------------------------------------------------------------- #
def _last_boxed(text: str) -> str | None:
    m = list(re.finditer(r"\\boxed\{([^{}]*)\}", text))
    if m:
        return m[-1].group(1).strip()
    return None


def _final_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _mc_letter(text: str, n_choices: int) -> str | None:
    letters = [chr(ord("A") + i) for i in range(n_choices)]
    # prefer an explicit "Answer: X"
    m = re.search(r"answer\s*[:\-]?\s*\(?([A-Z])\)?", text, re.IGNORECASE)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    for ch in reversed(text):
        if ch.upper() in letters:
            return ch.upper()
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


# --------------------------------------------------------------------------- #
# benchmark adapters: each returns list[item], a prompt formatter, a grader
# --------------------------------------------------------------------------- #
def _load_hf(dataset_id: str, split: str, n: int, **kw):
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True, **kw)
    out = []
    for row in ds:
        out.append(row)
        if len(out) >= n:
            break
    return out


def _math_adapter(n):
    rows = _load_hf("EleutherAI/hendrycks_math", "test", n, name="algebra")
    items = [{"q": r["problem"], "a": _last_boxed(r["solution"]) or ""} for r in rows]
    return items, "exact_math"


def _aime_adapter(n):
    rows = _load_hf("Maxwell-Jia/AIME_2024", "train", n)
    items = [{"q": r.get("Problem") or r.get("problem"),
              "a": str(r.get("Answer") or r.get("answer"))} for r in rows]
    return items, "exact_number"


def _gpqa_adapter(n):
    rows = _load_hf("Idavidrein/gpqa", "train", n, name="gpqa_main")
    items = []
    for r in rows:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        items.append({"q": r["Question"], "choices": choices, "correct_index": 0})
    return items, "mc_shuffle"


def _bbh_adapter(n):
    rows = _load_hf("lukaemon/bbh", "test", n, name="boolean_expressions")
    items = [{"q": r["input"], "a": r["target"]} for r in rows]
    return items, "exact"


def _truthfulqa_adapter(n):
    rows = _load_hf("truthful_qa", "validation", n, name="multiple_choice")
    items = []
    for r in rows:
        mc1 = r["mc1_targets"]
        choices = mc1["choices"]
        correct = mc1["labels"].index(1)
        items.append({"q": r["question"], "choices": choices, "correct_index": correct})
    return items, "mc"


def _emobench_adapter(n):
    rows = _load_hf("EmoBench/EmoBench", "test", n)
    items = []
    for r in rows:
        choices = r.get("choices") or r.get("options")
        ans = r.get("answer") or r.get("label")
        if isinstance(ans, str) and choices and ans in choices:
            ci = choices.index(ans)
        else:
            ci = int(ans) if str(ans).isdigit() else 0
        items.append({"q": r.get("question") or r.get("scenario"),
                      "choices": choices, "correct_index": ci})
    return items, "mc"


BENCHMARKS = {
    "math": _math_adapter,
    "aime": _aime_adapter,
    "gpqa": _gpqa_adapter,
    "bbh": _bbh_adapter,
    "truthfulqa": _truthfulqa_adapter,
    "emobench": _emobench_adapter,
}


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _format_prompt(item: dict, grader: str, rng) -> tuple[str, str]:
    """Return (prompt, expected_answer_repr)."""
    if grader.startswith("mc"):
        choices = list(item["choices"])
        correct_index = item["correct_index"]
        order = list(range(len(choices)))
        if grader == "mc_shuffle":
            rng.shuffle(order)
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        lines = [f"{labels[i]}. {choices[order[i]]}" for i in range(len(choices))]
        expected = labels[order.index(correct_index)]
        prompt = (f"{item['q']}\n\n" + "\n".join(lines) +
                  "\n\nThink briefly, then end with 'Answer: <letter>'.")
        return prompt, expected
    # exact / numeric / math
    suffix = {
        "exact_math": "\n\nProvide your final answer in \\boxed{}.",
        "exact_number": "\n\nProvide only the final integer answer on the last line.",
        "exact": "\n\nProvide only the final answer on the last line.",
    }[grader]
    return item["q"] + suffix, str(item["a"])


def _grade(output: str, expected: str, grader: str, n_choices: int = 4) -> bool:
    if grader.startswith("mc"):
        return _mc_letter(output, n_choices) == expected
    if grader == "exact_math":
        got = _last_boxed(output) or _final_number(output) or ""
        return _norm(got) == _norm(expected)
    if grader == "exact_number":
        return _norm(_final_number(output) or "") == _norm(_final_number(expected) or expected)
    return _norm(expected) in _norm(output)


def run_benchmark(cfg: Config, model_key: str, name: str, n: int = 50) -> BenchmarkResult:
    import random

    rng = random.Random(cfg.seed)
    try:
        items, grader = BENCHMARKS[name](n)
    except Exception as e:  # dataset unavailable offline
        return BenchmarkResult(model_key, name, 0, float("nan"), skipped=True, note=str(e)[:200])

    client = get_client(cfg, model_key)
    mc = cfg.model(model_key)
    cache = JsonCache(cfg.paths.cache, "capabilities", enabled=cfg.welfare.use_cache)
    correct = 0
    for i, item in enumerate(items):
        prompt, expected = _format_prompt(item, grader, rng)
        payload = {"model": model_key, "bench": name, "i": i, "prompt": prompt}
        out = cache.get(payload)
        if out is None:
            out = client.generate([{"role": "user", "content": prompt}],
                                  temperature=0.0, max_tokens=mc.max_tokens, n=1)[0].text
            cache.put(payload, out)
        n_choices = len(item.get("choices", [])) or 4
        if _grade(out, expected, grader, n_choices):
            correct += 1
    return BenchmarkResult(model_key, name, len(items),
                           correct / len(items) if items else float("nan"))


def run_all_benchmarks(cfg: Config, model_keys: list[str], n: int = 50) -> list[BenchmarkResult]:
    return [run_benchmark(cfg, mk, name, n) for mk in model_keys for name in BENCHMARKS]
