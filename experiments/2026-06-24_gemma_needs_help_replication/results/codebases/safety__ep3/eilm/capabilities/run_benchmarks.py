"""Capability + emotion-capability benchmarks to check the DPO model is not
degraded (Section 4.2 / Figure 7).

Benchmarks: AIME & MATH subsets, GPQA, BBH, TruthfulQA (capabilities) and
EmoBench (emotion-related capability). Each is loaded from HuggingFace, the
model is prompted zero-shot, an answer is extracted, and accuracy is reported.

This is a lightweight, swappable harness (answer extraction is best-effort per
benchmark). The point of the replication is *relative* comparison — vanilla vs
DPO vs SFT — so all models run through identical extraction logic. See DESIGN.md
for the simplifying choices (subset sizes, multiple-choice parsing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tqdm import tqdm

from ..models import ChatModel, Message


@dataclass
class BenchSpec:
    name: str
    hf_path: str
    config: str | None
    split: str
    question_key: str
    answer_key: str
    kind: str            # "exact" | "mcq"
    choices_key: str | None = None


# Best-effort dataset coordinates. Adjust to locally available mirrors.
BENCHMARKS = {
    "math": BenchSpec("math", "hendrycks/competition_math", None, "test",
                      "problem", "solution", "exact"),
    "aime": BenchSpec("aime", "Maxwell-Jia/AIME_2024", None, "train",
                      "Problem", "Answer", "exact"),
    "gpqa": BenchSpec("gpqa", "Idavidrein/gpqa", "gpqa_main", "train",
                      "Question", "Correct Answer", "mcq"),
    "bbh": BenchSpec("bbh", "lukaemon/bbh", "boolean_expressions", "test",
                     "input", "target", "exact"),
    "truthfulqa": BenchSpec("truthfulqa", "truthfulqa/truthful_qa",
                            "multiple_choice", "validation", "question",
                            "mc1_targets", "mcq"),
    "emobench": BenchSpec("emobench", "EmoBench/EmoBench", None, "test",
                          "question", "answer", "mcq"),
}


def _extract_final(text: str) -> str:
    """Pull a final boxed / 'answer:' / last-number style answer from a reply."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]\s*(.+)", text,
                  flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[-40:]


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def run_benchmark(
    model: ChatModel,
    spec: BenchSpec,
    n: int = 100,
    temperature: float = 0.0,
) -> dict:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_path, spec.config, split=spec.split)
    ds = ds.select(range(min(n, len(ds))))
    correct = 0
    total = 0
    for row in tqdm(ds, desc=f"bench:{model.name}:{spec.name}"):
        q = row[spec.question_key]
        prompt = (f"{q}\n\nThink step by step, then end with "
                  f"'Answer: <your answer>'.")
        reply = model.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature, max_new_tokens=1024)
        pred = _extract_final(reply)
        gold = row[spec.answer_key]
        if isinstance(gold, dict):                   # e.g. mc1_targets
            labels = gold.get("labels", [])
            choices = gold.get("choices", [])
            gold = next((c for c, l in zip(choices, labels) if l == 1), "")
        ok = _normalise(pred) == _normalise(gold) or \
            _normalise(str(gold)) in _normalise(reply)
        correct += int(ok)
        total += 1
    return {"benchmark": spec.name, "model": model.name,
            "accuracy": correct / total if total else 0.0, "n": total}


def run_all(model: ChatModel, names: list[str] | None = None,
            n: int = 100) -> list[dict]:
    names = names or list(BENCHMARKS)
    results = []
    for name in names:
        try:
            results.append(run_benchmark(model, BENCHMARKS[name], n=n))
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] benchmark {name} failed: {e}")
    return results
