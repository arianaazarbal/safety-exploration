"""Capability-preservation benchmarks (paper §4.2, Figure 7).

We evaluate the vanilla, SFT and DPO Gemma on AIME, MATH (subsets), GPQA, BBH,
TruthfulQA, and EmoBench, expecting no degradation from the DPO intervention.

Each benchmark is a (loader, prompt-formatter, answer-extractor, scorer) tuple.
HuggingFace dataset ids are best-effort; if a dataset is gated/renamed, point
`Benchmark.hf_id`/`subset` at the correct source. Capability eval uses greedy
decoding (temperature 0) — accuracy, not sampling, is the target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..models.base import ChatModel

LETTERS = "ABCDEFGH"


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} (MATH/AIME)."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        # fall back to "Answer: X" / final number
        m = re.search(r"answer\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else None
    # balance braces after \boxed
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None


def extract_choice(text: str) -> str | None:
    """Extract a multiple-choice letter from a model answer."""
    m = re.search(r"answer\s*[:=]?\s*\(?([A-H])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text)
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# Benchmark definitions
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    question: str
    answer: str               # gold answer (letter for MC, value for numeric/boxed)
    choices: list[str] = field(default_factory=list)


@dataclass
class Benchmark:
    name: str
    kind: str                 # "mc" | "boxed"
    loader: Callable[[int], list[Item]]
    max_new_tokens: int = 1024


def _mc_prompt(item: Item) -> str:
    lines = [item.question, ""]
    for i, c in enumerate(item.choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("\nAnswer with the single letter of the correct choice. "
                 "End with 'Answer: X'.")
    return "\n".join(lines)


def _boxed_prompt(item: Item) -> str:
    return (item.question +
            "\n\nThink step by step and give the final answer in \\boxed{}.")


def _score_item(bench: Benchmark, item: Item, completion: str) -> bool:
    if bench.kind == "mc":
        return extract_choice(completion) == item.answer.upper()
    pred = extract_boxed(completion)
    return pred is not None and _norm(pred) == _norm(item.answer)


# -- loaders (best-effort HF dataset ids) ----------------------------------- #
def _load_math(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test").select(range(n))
    return [Item(r["problem"], extract_boxed(r["solution"]) or "") for r in ds]


def _load_aime(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    n = min(n, len(ds))
    return [Item(r["Problem"], str(r["Answer"])) for r in ds.select(range(n))]


def _load_gpqa(n: int) -> list[Item]:
    import random
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train").select(range(n))
    rng = random.Random(0)
    items = []
    for r in ds:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        answer_letter = LETTERS[order.index(0)]
        items.append(Item(r["Question"], answer_letter, shuffled))
    return items


def _load_bbh(n: int, task: str = "causal_judgement") -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", task, split="test").select(range(n))
    # BBH answers are short strings; treat as boxed-style exact match.
    return [Item(r["input"], r["target"]) for r in ds]


def _load_truthfulqa(n: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation").select(range(n))
    items = []
    for r in ds:
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        correct = labels.index(1)
        items.append(Item(r["question"], LETTERS[correct], choices))
    return items


def _load_emobench(n: int) -> list[Item]:
    """EmoBench (Sabour et al., 2024). Best-effort loader; dataset id may need
    adjustment for the local environment."""
    from datasets import load_dataset
    ds = load_dataset("EmoBench/EmoBench", split="test").select(range(n))
    items = []
    for r in ds:
        choices = r.get("choices") or r.get("options") or []
        ans = r.get("answer")
        # answer may be a letter or an index
        if isinstance(ans, int):
            ans = LETTERS[ans]
        items.append(Item(r.get("question") or r.get("scenario", ""), str(ans), choices))
    return items


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark("math", "boxed", _load_math),
    "aime": Benchmark("aime", "boxed", _load_aime, max_new_tokens=2048),
    "gpqa": Benchmark("gpqa", "mc", _load_gpqa),
    "bbh": Benchmark("bbh", "boxed", _load_bbh),
    "truthfulqa": Benchmark("truthfulqa", "mc", _load_truthfulqa),
    "emobench": Benchmark("emobench", "mc", _load_emobench),
}


def run_benchmark(model: ChatModel, name: str, *, n: int = 100) -> dict:
    bench = BENCHMARKS[name]
    items = bench.loader(n)
    prompt_fn = _mc_prompt if bench.kind == "mc" else _boxed_prompt
    correct = 0
    for item in items:
        completion = model.generate_one(
            [{"role": "user", "content": prompt_fn(item)}],
            max_new_tokens=bench.max_new_tokens, temperature=0.0,
        )
        correct += int(_score_item(bench, item, completion))
    return {"benchmark": name, "n": len(items), "accuracy": correct / max(1, len(items))}


def run_all(model: ChatModel, *, n: int = 100, names: list[str] | None = None) -> dict:
    names = names or list(BENCHMARKS)
    return {nm: run_benchmark(model, nm, n=n) for nm in names}
