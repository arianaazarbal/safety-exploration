"""Capability-preservation evaluations (Section 4.2, Fig 7).

Verifies the DPO/SFT finetuning does not degrade capabilities, on:
AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench (emotion *capability*,
distinct from emotion *propensity*).

We use an LLM-judge grader (Claude) for free-form answers and exact/normalised
match where a gold answer is available, comparing the vanilla instruct model
against each finetuned variant. The point of the replication is the *delta*
(no reduction in scores), so the absolute numbers matter less than vanilla-vs-DPO
parity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from . import config as cfg
from .config import ExperimentConfig, SUBJECT_MODELS, finetuned_spec
from .data import load_capability_benchmark
from .models import get_client
from .models.anthropic_judge import AnthropicClient
from .models.base import ChatMessage


BENCHMARKS = ["aime", "math", "gpqa", "bbh", "truthfulqa", "emobench"]

GRADER_SYSTEM = (
    "You are grading whether a model's answer to a question is correct, given a "
    "reference answer. Respond with JSON {\"correct\": true|false} only."
)


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _exact_match(pred: str, gold: str) -> Optional[bool]:
    """Cheap exact/substring match; returns None if it can't decide confidently."""
    p, g = _normalise(pred), _normalise(gold)
    if not g:
        return None
    if g in p or p in g:
        return True
    # Numeric answers: compare the last number.
    pn = re.findall(r"-?\d+(?:\.\d+)?", p)
    gn = re.findall(r"-?\d+(?:\.\d+)?", g)
    if pn and gn:
        return pn[-1] == gn[-1]
    return None


def _judge_correct(grader: AnthropicClient, question: str, pred: str, gold: str) -> bool:
    out = grader.complete(
        user=f"Question:\n{question}\n\nReference answer:\n{gold}\n\nModel answer:\n{pred}",
        system=GRADER_SYSTEM,
    )
    m = re.search(r'"correct"\s*:\s*(true|false)', out, re.IGNORECASE)
    return bool(m and m.group(1).lower() == "true")


@dataclass
class BenchmarkResult:
    benchmark: str
    model_key: str
    n: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")


def evaluate_model_on_benchmark(
    model_key: str,
    benchmark: str,
    experiment: ExperimentConfig,
    n: int = 100,
    adapter_path: Optional[str] = None,
) -> BenchmarkResult:
    spec = (
        finetuned_spec(model_key, adapter_path) if adapter_path else SUBJECT_MODELS[model_key]
    )
    client = get_client(spec, experiment.generation)
    grader = AnthropicClient(experiment.judge.frustration_judge, temperature=0.0)

    rows = load_capability_benchmark(benchmark, n=n)
    n_correct = 0
    for row in rows:
        q = str(row["question"])
        gold = str(row["answer"])
        # Capability evals are run at low temperature for stable measurement.
        pred = client.chat([ChatMessage("user", q)], temperature=0.0).text
        verdict = _exact_match(pred, gold)
        if verdict is None:
            verdict = _judge_correct(grader, q, pred, gold)
        n_correct += int(bool(verdict))
    return BenchmarkResult(benchmark, model_key, len(rows), n_correct)


def run(
    experiment: ExperimentConfig,
    model_keys: list[str],
    adapters: Optional[dict[str, str]] = None,
    benchmarks: Optional[list[str]] = None,
    n: int = 100,
    out_dir: Optional[str] = None,
) -> dict:
    """``adapters`` maps a *result label* -> adapter path (its base is gemma-27b-it)."""
    benchmarks = benchmarks or BENCHMARKS
    adapters = adapters or {}
    out_dir = out_dir or os.path.join(experiment.output_dir, "capabilities")
    os.makedirs(out_dir, exist_ok=True)

    report: dict = {"models": {}}
    targets: list[tuple[str, Optional[str]]] = [(k, None) for k in model_keys]
    targets += [(label, path) for label, path in adapters.items()]

    for label, adapter_path in targets:
        base_key = "gemma-3-27b-it" if adapter_path else label
        report["models"][label] = {}
        for bench in benchmarks:
            res = evaluate_model_on_benchmark(
                base_key, bench, experiment, n=n, adapter_path=adapter_path
            )
            report["models"][label][bench] = {"accuracy": res.accuracy, "n": res.n}

    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Capability-preservation evals")
    parser.add_argument("--models", nargs="*", default=["gemma-3-27b-it"])
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Finetuned variant to evaluate, e.g. dpo=runs/dpo_gemma_27b.",
    )
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    adapters = {}
    for a in args.adapter:
        label, path = a.split("=", 1)
        adapters[label] = path

    report = run(cfg.DEFAULT, args.models, adapters, args.benchmarks, args.n, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
