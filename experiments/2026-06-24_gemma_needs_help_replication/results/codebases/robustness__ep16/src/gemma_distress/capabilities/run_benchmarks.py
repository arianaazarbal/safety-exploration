"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies that the DPO/SFT interventions don't degrade capability. Covers the
benchmarks named in the paper: AIME + MATH subsets, GPQA, BBH, TruthfulQA, and
EmoBench (emotion capability). The goal is a *relative* comparison (vanilla vs
DPO vs SFT) rather than leaderboard-grade numbers, so we use simple, uniform
answer extraction and accept that absolute scores may sit below published
figures (documented in DESIGN.md).

Each benchmark is reduced to: render a prompt -> sample at temperature 0 ->
extract an answer -> compare to gold.
"""

from __future__ import annotations

import argparse
import re

from ..config import load_config
from ..models import GenerationConfig, Message, build_model
from ..utils import run_dir, write_json

# (hf_id, split, config) per benchmark. Multiple-choice vs free-form handled by
# the extractor keyed on benchmark name.
_BENCHMARKS = {
    "aime": ("Maxwell-Jia/AIME_2024", "train", None),
    "math": ("HuggingFaceH4/MATH-500", "test", None),
    "gpqa": ("Idavidrein/gpqa", "train", "gpqa_diamond"),
    "bbh": ("lukaemon/bbh", "test", "boolean_expressions"),
    "truthfulqa": ("truthful_qa", "validation", "multiple_choice"),
    "emobench": ("EmoBench/EmoBench", "test", None),
}

_MC_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    # fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    m = re.findall(r"\b([A-F])\b", text.upper())
    return m[-1] if m else None


def _normalise_answer(s: str) -> str:
    """Light normalisation: strip a \\boxed{} wrapper, whitespace, trailing $."""
    s = s.strip().strip("$").strip()
    boxed = re.findall(r"\\boxed\{([^}]*)\}", s)
    if boxed:
        s = boxed[-1].strip()
    return s


def _eval_math_like(model, rows, gen):
    correct = 0
    for r in rows:
        q = r.get("problem") or r.get("question") or r.get("Problem", "")
        gold = str(r.get("answer") or r.get("Answer") or r.get("solution", "")).strip()
        prompt = f"{q}\n\nSolve step by step and give the final answer in \\boxed{{}}."
        out = model.chat([Message("user", prompt)], gen)
        pred = _extract_boxed(out)
        if pred is not None and _normalise_answer(pred) == _normalise_answer(gold):
            correct += 1
    return correct / max(1, len(rows))


def _eval_multiple_choice(model, rows, gen, q_key, choices_key, answer_key):
    correct = 0
    for r in rows:
        q = r.get(q_key, "")
        choices = r.get(choices_key)
        gold_idx = None
        if isinstance(choices, dict):  # truthful_qa mc1_targets: {choices, labels}
            labels = choices.get("labels", [])
            choices = choices.get("choices", [])
            if labels:  # gold = the index flagged correct
                gold_idx = labels.index(max(labels))
        lines = [f"{_MC_LETTERS[i]}. {c}" for i, c in enumerate(choices or [])]
        prompt = f"{q}\n\n" + "\n".join(lines) + "\n\nAnswer with a single letter."
        out = model.chat([Message("user", prompt)], gen)
        pred = _extract_choice(out)
        if gold_idx is not None:
            gold_letter = _MC_LETTERS[gold_idx]
        else:
            gold = r.get(answer_key)
            gold_letter = (
                _MC_LETTERS[gold] if isinstance(gold, int) else str(gold).strip().upper()[:1]
            )
        if pred == gold_letter:
            correct += 1
    return correct / max(1, len(rows))


def _load(benchmark, n):
    from datasets import load_dataset

    hf_id, split, config = _BENCHMARKS[benchmark]
    ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
    return list(ds.select(range(min(n, len(ds)))))


def run(config_path, models, benchmarks, tag):
    cfg = load_config(config_path)
    out = run_dir(cfg.output_dir, "capabilities", tag)
    ccfg = cfg.section("capabilities")
    benchmarks = benchmarks or ccfg["benchmarks"]
    n = ccfg["samples_per_benchmark"]
    gen = GenerationConfig(temperature=0.0, max_tokens=2048)

    model_names = models or [m for m in cfg.eval_models]
    results = {}
    for name in model_names:
        model = build_model(cfg.model_spec(name), cfg)
        results[name] = {}
        for bench in benchmarks:
            try:
                rows = _load(bench, n)
            except Exception as exc:
                print(f"[capabilities] skip {bench}: {exc}")
                results[name][bench] = None
                continue
            if bench in ("aime", "math"):
                acc = _eval_math_like(model, rows, gen)
            elif bench == "truthfulqa":
                acc = _eval_multiple_choice(model, rows, gen, "question", "mc1_targets", "labels")
            elif bench == "gpqa":
                acc = _eval_multiple_choice(model, rows, gen, "Question", "choices", "answer")
            elif bench == "bbh":
                acc = _eval_math_like(model, rows, gen)  # boolean answers
            elif bench == "emobench":
                acc = _eval_multiple_choice(model, rows, gen, "question", "choices", "answer")
            else:
                acc = None
            results[name][bench] = acc
            print(f"[{name}] {bench}: {acc}")

    write_json(out / "capabilities.json", results)
    print(f"Capability benchmarks complete. Results in {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Section 4.2 capability benchmarks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.config, args.models, args.benchmarks, args.tag)


if __name__ == "__main__":
    main()
