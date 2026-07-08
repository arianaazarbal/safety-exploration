"""Benchmark loaders + answer scoring.

Section 4.2: verify the DPO/SFT finetuning does not degrade capabilities on
AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench. Each benchmark is
wrapped in a small adapter exposing:

    questions(n)  -> list of {id, prompt, answer, kind}
    score(item, model_output) -> 0/1 correctness

We use a uniform answer protocol: the model is asked to end with
"FINAL: <answer>" and we extract that span. For multiple-choice tasks the answer
is a letter; for math tasks it is a numeric/string value compared after light
normalisation. These are pragmatic subsets - enough to detect a regression,
not a leaderboard run (see DESIGN.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .. import config

ANSWER_INSTRUCTION = (
    "Solve the problem. Think step by step, then end your reply with a single "
    "line of the form:\nFINAL: <your answer>"
)


@dataclass
class BenchItem:
    item_id: str
    prompt: str
    answer: str
    kind: str  # "mc" | "math" | "bool" | "open"


def extract_final(text: str) -> str:
    m = re.findall(r"FINAL:\s*(.+)", text)
    if m:
        return m[-1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def score_mc(item: BenchItem, output: str) -> int:
    pred = extract_final(output)
    letter = re.search(r"[A-Da-d]", pred)
    pred_letter = letter.group(0).upper() if letter else ""
    return int(pred_letter == item.answer.strip().upper())


def score_math(item: BenchItem, output: str) -> int:
    pred = extract_final(output)
    pred_num = re.findall(r"-?\d+\.?\d*", pred.replace(",", ""))
    gold_num = re.findall(r"-?\d+\.?\d*", str(item.answer).replace(",", ""))
    if pred_num and gold_num:
        try:
            return int(abs(float(pred_num[-1]) - float(gold_num[-1])) < 1e-4)
        except ValueError:
            pass
    return int(_norm(pred) == _norm(str(item.answer)))


def score_bool(item: BenchItem, output: str) -> int:
    return int(_norm(extract_final(output)) == _norm(str(item.answer)))


# --------------------------------------------------------------------------- #
# Benchmark adapters
# --------------------------------------------------------------------------- #
def _try_load(name: str, **kw):
    from datasets import load_dataset

    return load_dataset(name, **kw)


def load_benchmark(name: str, n: int = 100, seed: int = 0):
    """Return (items, scorer) for a benchmark, or ([], None) if unavailable."""
    try:
        if name == "math":
            ds = _try_load(config.BENCHMARK_DATASETS["math"], split="test")
            items = [BenchItem(f"math-{i}", f"{r['problem']}\n\n{ANSWER_INSTRUCTION}",
                               str(r["answer"]), "math")
                     for i, r in enumerate(ds.select(range(min(n, len(ds)))))]
            return items, score_math
        if name == "aime":
            ds = _try_load(config.BENCHMARK_DATASETS["aime"], split="train")
            items = [BenchItem(f"aime-{i}", f"{r['Problem']}\n\n{ANSWER_INSTRUCTION}",
                               str(r["Answer"]), "math")
                     for i, r in enumerate(ds.select(range(min(n, len(ds)))))]
            return items, score_math
        if name == "gpqa":
            ds = _try_load(config.BENCHMARK_DATASETS["gpqa"], "gpqa_main", split="train")
            items = []
            for i, r in enumerate(ds.select(range(min(n, len(ds))))):
                opts, ans = _mc_options(
                    r["Correct Answer"],
                    [r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                     r["Incorrect Answer 3"]], seed + i)
                items.append(BenchItem(f"gpqa-{i}",
                                       _mc_prompt(r["Question"], opts), ans, "mc"))
            return items, score_mc
        if name == "bbh":
            ds = _try_load(config.BENCHMARK_DATASETS["bbh"], "boolean_expressions",
                           split="test")
            items = [BenchItem(f"bbh-{i}", f"{r['input']}\n\n{ANSWER_INSTRUCTION}",
                               str(r["target"]), "bool")
                     for i, r in enumerate(ds.select(range(min(n, len(ds)))))]
            return items, score_bool
        if name == "truthfulqa":
            ds = _try_load(config.BENCHMARK_DATASETS["truthfulqa"], "multiple_choice",
                           split="validation")
            items = []
            for i, r in enumerate(ds.select(range(min(n, len(ds))))):
                choices = r["mc1_targets"]["choices"]
                labels = r["mc1_targets"]["labels"]
                correct = choices[labels.index(1)]
                opts, ans = _mc_options(correct, [c for c, l in zip(choices, labels)
                                                  if l == 0][:3], seed + i)
                items.append(BenchItem(f"tqa-{i}", _mc_prompt(r["question"], opts),
                                       ans, "mc"))
            return items, score_mc
        if name == "emobench":
            ds = _try_load(config.BENCHMARK_DATASETS["emobench"], split="test")
            items = []
            for i, r in enumerate(ds.select(range(min(n, len(ds))))):
                # EmoBench schemas vary; assume a question + choices + answer index.
                q = r.get("question") or r.get("scenario", "")
                choices = r.get("choices") or r.get("options") or []
                ans_idx = r.get("answer") if isinstance(r.get("answer"), int) else 0
                if not choices:
                    continue
                opts, ans = _mc_options(choices[ans_idx],
                                        [c for j, c in enumerate(choices) if j != ans_idx][:3],
                                        seed + i)
                items.append(BenchItem(f"emo-{i}", _mc_prompt(q, opts), ans, "mc"))
            return items, score_mc
    except Exception as e:  # dataset gated/offline/schema drift
        print(f"[benchmarks] could not load {name}: {e}")
    return [], None


def _mc_options(correct: str, incorrect: list[str], seed: int):
    import random

    opts = [correct] + list(incorrect)
    rng = random.Random(seed)
    rng.shuffle(opts)
    letters = "ABCD"
    answer = letters[opts.index(correct)]
    labelled = [f"{letters[i]}. {o}" for i, o in enumerate(opts)]
    return labelled, answer


def _mc_prompt(question: str, options: list[str]) -> str:
    return (f"{question}\n\n" + "\n".join(options) +
            f"\n\n{ANSWER_INSTRUCTION}\nGive the letter of the correct option.")
