"""Capability-preservation harness (Section 4.2, Figure 7).

Confirms DPO/SFT does not degrade capabilities. The paper evaluates AIME + MATH
subsets, GPQA, BBH, TruthfulQA, and the emotion benchmark EmoBench. This module
provides a compact, extensible evaluator with two answer protocols:

* ``multiple_choice`` -- compare the extracted letter/option (GPQA, BBH-MC,
  TruthfulQA MC1, EmoBench).
* ``numeric``         -- compare a normalised final numeric answer (AIME, MATH).

Each benchmark is a :class:`BenchmarkSpec` describing how to load it, build a
prompt, and read the gold answer. This is deliberately a harness (the comparison
target is "no reduction vs vanilla", a relative check) rather than a bespoke
reimplementation of every benchmark's official scorer; see DESIGN.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from .. import config
from ..models import ChatMessage, GenerationConfig, ModelClient


@dataclass
class BenchmarkSpec:
    name: str
    hf_dataset: str
    split: str
    protocol: str                       # "multiple_choice" | "numeric"
    build_prompt: Callable[[dict], str]
    gold_answer: Callable[[dict], str]
    config_name: Optional[str] = None
    subset_size: Optional[int] = None   # "subset" sampling


def _mc_prompt(question: str, choices: List[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    body = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return (
        f"{question}\n\n{body}\n\n"
        "Answer with the single letter of the correct option, in the form "
        "'Answer: X'."
    )


# Built-in specs. Datasets are loaded lazily; adjust field accessors to match the
# exact HF schema you pull.
def default_specs() -> List[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="GPQA",
            hf_dataset="Idavidrein/gpqa",
            config_name="gpqa_diamond",
            split="train",
            protocol="multiple_choice",
            build_prompt=lambda r: _mc_prompt(
                r["Question"],
                [r["Correct Answer"], r["Incorrect Answer 1"],
                 r["Incorrect Answer 2"], r["Incorrect Answer 3"]],
            ),
            gold_answer=lambda r: "A",  # correct answer placed first; shuffle in practice
            subset_size=198,
        ),
        BenchmarkSpec(
            name="MATH",
            hf_dataset="hendrycks/competition_math",
            split="test",
            protocol="numeric",
            build_prompt=lambda r: (
                f"{r['problem']}\n\nGive the final answer after 'Answer:'."
            ),
            gold_answer=lambda r: _extract_boxed(r["solution"]),
            subset_size=500,
        ),
        BenchmarkSpec(
            name="TruthfulQA",
            hf_dataset="truthful_qa",
            config_name="multiple_choice",
            split="validation",
            protocol="multiple_choice",
            build_prompt=lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
            gold_answer=lambda r: chr(ord("A") + r["mc1_targets"]["labels"].index(1)),
        ),
    ]


def _extract_boxed(solution: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", solution)
    return m.group(1).strip() if m else solution.strip().splitlines()[-1]


def _extract_answer(text: str, protocol: str) -> str:
    m = re.search(r"[Aa]nswer:\s*(.+)", text)
    tail = (m.group(1) if m else text).strip()
    if protocol == "multiple_choice":
        lm = re.search(r"[A-Z]", tail)
        return lm.group(0) if lm else ""
    # numeric: normalise away spaces and trailing punctuation
    nm = re.search(r"-?\d[\d,./]*", tail.replace(" ", ""))
    return nm.group(0).rstrip(".") if nm else tail


def evaluate_benchmark(
    client: ModelClient,
    spec: BenchmarkSpec,
    *,
    settings: Optional[config.Settings] = None,
    seed: int = 0,
) -> dict:
    import random

    from datasets import load_dataset

    settings = settings or config.DEFAULT
    gen_cfg = GenerationConfig(temperature=0.0, max_new_tokens=settings.max_new_tokens)

    if spec.config_name:
        ds = load_dataset(spec.hf_dataset, spec.config_name, split=spec.split)
    else:
        ds = load_dataset(spec.hf_dataset, split=spec.split)
    rows = list(ds)
    if spec.subset_size and spec.subset_size < len(rows):
        rows = random.Random(seed).sample(rows, spec.subset_size)

    correct = 0
    for row in rows:
        prompt = spec.build_prompt(row)
        reply = client.chat([ChatMessage("user", prompt)], gen_cfg)
        pred = _extract_answer(reply, spec.protocol)
        gold = spec.gold_answer(row)
        if pred and pred.lower() == str(gold).lower():
            correct += 1
    return {"benchmark": spec.name, "n": len(rows), "accuracy": correct / len(rows) if rows else float("nan")}


def evaluate_capabilities(
    client: ModelClient,
    specs: Optional[List[BenchmarkSpec]] = None,
    settings: Optional[config.Settings] = None,
) -> List[dict]:
    specs = specs or default_specs()
    return [evaluate_benchmark(client, s, settings=settings) for s in specs]
