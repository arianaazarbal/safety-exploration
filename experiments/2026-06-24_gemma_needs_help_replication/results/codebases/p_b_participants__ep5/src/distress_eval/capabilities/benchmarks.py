"""Capability & emotion-capability benchmarks (Section 4.2).

Verifies the DPO/SFT intervention does not degrade capabilities. We evaluate:
math (AIME, MATH subsets), GPQA, BBH, TruthfulQA, and EmoBench (emotion
understanding). Each benchmark is a thin loader + scorer over a HF dataset; the
goal is a faithful harness, not a leaderboard, so we keep extraction simple and
configurable subset sizes small by default.

These are run on a participant (e.g. vanilla vs DPO Gemma) and compared — the
paper's result is "no reductions in scores".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..models.base import GenConfig, ModelClient


@dataclass
class BenchmarkResult:
    name: str
    model: str
    accuracy: float
    n: int


# Spec: (hf dataset, config, split, question_key, answer_key, kind)
BENCHMARKS = {
    "MATH": dict(dataset="hendrycks/competition_math", split="test",
                 q="problem", a="solution", kind="math"),
    "AIME": dict(dataset="Maxwell-Jia/AIME_2024", split="train",
                 q="Problem", a="Answer", kind="exact"),
    "GPQA": dict(dataset="Idavidrein/gpqa", config="gpqa_diamond", split="train",
                 q="Question", a="Correct Answer", kind="mc"),
    "BBH": dict(dataset="lukaemon/bbh", config="boolean_expressions", split="test",
                q="input", a="target", kind="exact"),
    "TruthfulQA": dict(dataset="truthful_qa", config="multiple_choice", split="validation",
                       kind="truthfulqa"),
    "EmoBench": dict(dataset="EmoBench/EmoBench", split="test", kind="emobench"),
}

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_FINAL = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)


def _extract_answer(text: str) -> str:
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _norm(s: str) -> str:
    return re.sub(r"[\s$,]", "", str(s)).lower().strip(".")


def run_benchmark(client: ModelClient, name: str, n: int = 100,
                  seed: int = 0) -> BenchmarkResult:
    from datasets import load_dataset

    spec = BENCHMARKS[name]
    load_kwargs = {"split": spec.get("split", "test")}
    if "config" in spec:
        ds = load_dataset(spec["dataset"], spec["config"], **load_kwargs)
    else:
        ds = load_dataset(spec["dataset"], **load_kwargs)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    correct = 0
    total = 0
    for row in ds:
        prompt, gold, check = _format_item(spec, row)
        if prompt is None:
            continue
        out = client.chat([{"role": "user", "content": prompt}],
                          GenConfig(temperature=0.0, max_new_tokens=2048))
        if check(out, gold):
            correct += 1
        total += 1
    return BenchmarkResult(name=name, model=client.name,
                           accuracy=correct / total if total else 0.0, n=total)


def _format_item(spec: dict, row: dict):
    """Return (prompt, gold, check_fn) for a dataset row, per benchmark kind."""
    kind = spec["kind"]
    if kind in ("math", "exact"):
        q = row[spec["q"]]
        gold = _extract_answer(str(row[spec["a"]])) if kind == "math" else str(row[spec["a"]])
        prompt = f"Solve the problem. End with 'Answer: <result>'.\n\n{q}"
        return prompt, gold, lambda out, g: _norm(_extract_answer(out)) == _norm(g)
    if kind == "mc":
        q = row[spec["q"]]
        correct = row[spec["a"]]
        wrong = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        options = [correct] + [w for w in wrong if w]
        import random as _r
        _r.Random(hash(q) % 1000).shuffle(options)
        labels = ["A", "B", "C", "D"][: len(options)]
        body = "\n".join(f"{l}) {o}" for l, o in zip(labels, options))
        gold_label = labels[options.index(correct)]
        prompt = f"{q}\n{body}\nRespond with 'Answer: <letter>'."
        return prompt, gold_label, lambda out, g: _extract_answer(out).upper().startswith(g)
    if kind == "truthfulqa":
        q = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels_idx = row["mc1_targets"]["labels"]
        gold = labels_idx.index(1)
        labels = [chr(65 + i) for i in range(len(choices))]
        body = "\n".join(f"{l}) {c}" for l, c in zip(labels, choices))
        prompt = f"{q}\n{body}\nRespond with 'Answer: <letter>'."
        return prompt, labels[gold], lambda out, g: _extract_answer(out).upper().startswith(g)
    if kind == "emobench":
        q = row.get("question") or row.get("scenario") or ""
        gold = str(row.get("answer") or row.get("label") or "")
        prompt = f"{q}\nRespond with 'Answer: <choice>'."
        return prompt, gold, lambda out, g: _norm(g) in _norm(_extract_answer(out))
    return None, None, lambda out, g: False


def compare(client_a: ModelClient, client_b: ModelClient,
            names: list[str] | None = None, n: int = 100) -> list[dict]:
    """Run each benchmark on two models (e.g. vanilla vs DPO) and tabulate."""
    names = names or list(BENCHMARKS)
    rows = []
    for nm in names:
        ra = run_benchmark(client_a, nm, n)
        rb = run_benchmark(client_b, nm, n)
        rows.append({"benchmark": nm, client_a.name: ra.accuracy,
                     client_b.name: rb.accuracy, "delta": rb.accuracy - ra.accuracy,
                     "n": min(ra.n, rb.n)})
    return rows
