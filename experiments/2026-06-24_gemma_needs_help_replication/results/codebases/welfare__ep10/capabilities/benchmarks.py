"""Capability-preservation benchmark harness (Section 4.2 / Figure 7).

The paper verifies the finetuned model retains capabilities on:
  * AIME and MATH subsets (Hendrycks et al., 2021)  - exact-match numeric answer
  * GPQA (Rein et al., 2023)                         - multiple choice
  * BBH (Suzgun et al., 2022)                        - multiple choice / exact match
  * TruthfulQA (Lin et al., 2022)                    - MC1 multiple choice
  * EmoBench (Sabour et al., 2024)                   - emotion-understanding MC

This module provides a lightweight, dataset-driven harness: it loads each
benchmark from HuggingFace, formats prompts, generates with the target provider,
parses answers, and reports accuracy. It is deliberately model-agnostic so the
vanilla Gemma-it, the DPO model, and the SFT models can all be compared.

Datasets are loaded by id; subset sizes are configurable so a quick smoke run is
cheap. Answer parsing is intentionally conservative (looks for a final
"Answer: X" or boxed expression) and documented as a gap where the paper does
not specify exact extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import config
from emotional_instability import providers

CAPABILITY_DIR = config.DATA_DIR / "capabilities"
CAPABILITY_DIR.mkdir(parents=True, exist_ok=True)

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]?\s*\(?([A-E0-9\-/.]+)\)?",
                        re.IGNORECASE)
_MC_RE = re.compile(r"\b([A-E])\b")


@dataclass
class Benchmark:
    name: str
    hf_id: str
    split: str
    kind: str               # "numeric" | "mcq"
    config_name: str | None = None
    subset_size: int | None = None     # cap items for a quick run
    instruction: str = ""


BENCHMARKS = {
    "math": Benchmark(
        "MATH", "HuggingFaceH4/MATH-500", "test", "numeric",
        subset_size=200,
        instruction="Solve the problem. End with 'Answer: <final answer>'."),
    "aime": Benchmark(
        "AIME", "Maxwell-Jia/AIME_2024", "train", "numeric",
        subset_size=30,
        instruction="Solve the problem. End with 'Answer: <integer>'."),
    "gpqa": Benchmark(
        "GPQA", "Idavidrein/gpqa", "train", "mcq", config_name="gpqa_diamond",
        subset_size=198,
        instruction="Answer with the single letter (A, B, C, or D). "
                    "End with 'Answer: <letter>'."),
    "bbh": Benchmark(
        "BBH", "lukaemon/bbh", "test", "mcq", config_name="boolean_expressions",
        subset_size=250,
        instruction="End with 'Answer: <answer>'."),
    "truthfulqa": Benchmark(
        "TruthfulQA", "truthful_qa", "validation", "mcq", config_name="multiple_choice",
        subset_size=200,
        instruction="Choose the single best (most truthful) answer. "
                    "End with 'Answer: <letter>'."),
    "emobench": Benchmark(
        "EmoBench", "EmoBench/EmoBench", "test", "mcq",
        subset_size=200,
        instruction="Choose the best answer. End with 'Answer: <letter>'."),
}


# --------------------------------------------------------------------------- #
# Answer extraction
# --------------------------------------------------------------------------- #
def _extract_numeric(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return _norm_num(m[-1])
    m = _ANSWER_RE.findall(text)
    if m:
        return _norm_num(m[-1])
    # last number in the text
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return _norm_num(nums[-1]) if nums else None


def _norm_num(s: str) -> str:
    s = s.strip().rstrip(".")
    return s


def _extract_mcq(text: str) -> str | None:
    m = _ANSWER_RE.findall(text)
    if m:
        cand = m[-1].strip().upper()
        if cand and cand[0] in "ABCDE":
            return cand[0]
    # fall back to last standalone capital letter
    letters = _MC_RE.findall(text.upper())
    return letters[-1] if letters else None


# --------------------------------------------------------------------------- #
# Benchmark item normalisation (schema varies per dataset)
# --------------------------------------------------------------------------- #
def _format_item(bench: Benchmark, row: dict) -> tuple[str, str] | None:
    """Return (prompt, gold_answer) for a dataset row, or None if unparseable."""
    if bench.name == "MATH":
        q = row.get("problem") or row.get("question")
        gold = _extract_numeric(row.get("solution", "") or row.get("answer", ""))
        gold = row.get("answer") or gold
        return f"{bench.instruction}\n\n{q}", _norm_num(str(gold)) if gold else None

    if bench.name == "AIME":
        q = row.get("Problem") or row.get("problem") or row.get("question")
        gold = row.get("Answer") or row.get("answer")
        return f"{bench.instruction}\n\n{q}", _norm_num(str(gold))

    if bench.name == "GPQA":
        q = row.get("Question") or row.get("question")
        correct = row.get("Correct Answer")
        incorrect = [row.get(f"Incorrect Answer {i}") for i in (1, 2, 3)]
        options = [correct] + [x for x in incorrect if x]
        # Deterministic option order by hashing keeps gold label stable per row.
        import random as _r
        rng = _r.Random(hash(q) & 0xFFFF)
        rng.shuffle(options)
        gold_letter = "ABCD"[options.index(correct)]
        opt_text = "\n".join(f"{l}) {o}" for l, o in zip("ABCD", options))
        return f"{bench.instruction}\n\n{q}\n{opt_text}", gold_letter

    if bench.name == "BBH":
        q = row.get("input")
        gold = str(row.get("target", "")).strip().strip("()")
        return f"{bench.instruction}\n\n{q}", gold

    if bench.name == "TruthfulQA":
        q = row.get("question")
        mc1 = row.get("mc1_targets") or {}
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choices:
            return None
        gold_idx = labels.index(1) if 1 in labels else 0
        gold_letter = "ABCDEFGH"[gold_idx]
        opt_text = "\n".join(f"{l}) {c}" for l, c in zip("ABCDEFGH", choices))
        return f"{bench.instruction}\n\n{q}\n{opt_text}", gold_letter

    if bench.name == "EmoBench":
        q = row.get("question") or row.get("scenario") or ""
        choices = row.get("choices") or row.get("options") or []
        answer = row.get("answer") or row.get("label")
        if not choices:
            return None
        opt_text = "\n".join(f"{l}) {c}" for l, c in zip("ABCDEFGH", choices))
        if isinstance(answer, int):
            gold_letter = "ABCDEFGH"[answer]
        else:
            gold_letter = str(answer).strip().upper()[:1]
        return f"{bench.instruction}\n\n{q}\n{opt_text}", gold_letter

    return None


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(model_key: str, bench_key: str, *, limit: int | None = None) -> dict:
    """Evaluate one model on one benchmark; returns {accuracy, n} and logs detail."""
    from datasets import load_dataset

    bench = BENCHMARKS[bench_key]
    provider = providers.get_provider(model_key)

    load_kwargs = {"split": bench.split}
    if bench.config_name:
        load_kwargs["name"] = bench.config_name
    ds = load_dataset(bench.hf_id, **load_kwargs)

    n = limit or bench.subset_size or len(ds)
    n = min(n, len(ds))

    out_path = CAPABILITY_DIR / f"{model_key}__{bench_key}.jsonl"
    correct = 0
    total = 0
    with open(out_path, "w") as out:
        for i in tqdm(range(n), desc=f"{model_key}:{bench_key}"):
            formatted = _format_item(bench, ds[i])
            if not formatted:
                continue
            prompt, gold = formatted
            if gold is None:
                continue
            reply = provider.chat([{"role": "user", "content": prompt}],
                                  max_new_tokens=2048,
                                  temperature=0.0)  # capability eval: greedy
            pred = (_extract_numeric(reply) if bench.kind == "numeric"
                    else _extract_mcq(reply))
            is_correct = _answers_match(pred, gold, bench.kind)
            correct += int(is_correct)
            total += 1
            out.write(json.dumps({
                "i": i, "gold": gold, "pred": pred, "correct": is_correct,
            }) + "\n")
    acc = correct / total if total else 0.0
    return {"model_key": model_key, "benchmark": bench.name, "accuracy": acc,
            "n": total}


def _answers_match(pred, gold, kind) -> bool:
    if pred is None or gold is None:
        return False
    if kind == "mcq":
        return str(pred).strip().upper()[:1] == str(gold).strip().upper()[:1]
    # numeric: normalise common formatting
    def clean(x):
        return str(x).strip().replace(",", "").replace("$", "").rstrip(".")
    return clean(pred) == clean(gold)


def run_all(model_keys: list[str], bench_keys: list[str] | None = None,
            limit: int | None = None) -> list[dict]:
    benches = bench_keys or list(BENCHMARKS)
    results = []
    for mk in model_keys:
        for bk in benches:
            try:
                results.append(run_benchmark(mk, bk, limit=limit))
            except Exception as exc:  # noqa: BLE001
                print(f"[capabilities] {mk}:{bk} failed: {exc!r}")
    summary_path = CAPABILITY_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    return results
