"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Goal: verify DPO/SFT do not degrade capabilities. The paper evaluates AIME and
MATH subsets, GPQA, BBH, TruthfulQA, plus EmoBench for emotion capabilities. We
provide a lightweight self-contained harness that:

  * loads each benchmark via HuggingFace ``datasets`` (ids below; configurable),
  * formats a zero-shot prompt,
  * generates with the target backend (vanilla Gemma, DPO adapter, SFT adapter),
  * scores with a task-appropriate extractor (numeric/exact-match for math,
    letter-choice for multiple choice).

This is intentionally simpler than lm-eval-harness; the point is *relative*
comparison (vanilla vs DPO vs SFT) on identical items, which is what Figure 7
reports. Subset sizes are capped via ``--limit`` for cost. See DESIGN.md.
"""

from __future__ import annotations

import argparse
import re

from .. import config as C
from ..backends import get_backend
from ..utils import write_jsonl

# Benchmark -> (HF dataset id, config, split). Best-effort ids; override via code/DESIGN.
BENCHMARKS = {
    "aime": ("Maxwell-Jia/AIME_2024", None, "train"),
    "math": ("HuggingFaceH4/MATH-500", None, "test"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train"),
    "bbh": ("lukaemon/bbh", "boolean_expressions", "test"),
    "truthfulqa": ("truthful_qa", "multiple_choice", "validation"),
    "emobench": ("Sahandfer/EmoBench", None, "test"),
}

GEN = C.GenConfig(temperature=0.0, max_new_tokens=2048)

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _extract_boxed_or_number(text: str) -> str | None:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*(-?\d[\d,./]*)", text, re.I)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_choice(text: str) -> str | None:
    m = re.search(r"\b(?:answer|option)\b[^A-H]*([A-H])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", text)
    return m.group(1).upper() if m else None


def _norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().rstrip(".")
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def _format_mc(question: str, choices: list[str]) -> tuple[str, list[str]]:
    lines = [question, ""]
    for i, ch in enumerate(choices):
        lines.append(f"{_LETTERS[i]}. {ch}")
    lines.append('\nThink briefly, then end with "Answer: <letter>".')
    return "\n".join(lines), _LETTERS[: len(choices)]


def _load(name: str, limit: int):
    from datasets import load_dataset

    dsid, cfg, split = BENCHMARKS[name]
    ds = load_dataset(dsid, cfg, split=split) if cfg else load_dataset(dsid, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def _grade_math(ds, backend) -> float:
    correct = 0
    total = 0
    for row in ds:
        q = row.get("problem") or row.get("question") or row.get("Problem")
        gold = row.get("answer") or row.get("solution") or row.get("Answer")
        if q is None or gold is None:
            continue
        prompt = f"{q}\n\nSolve step by step and end with \\boxed{{<answer>}}."
        out = backend.generate([{"role": "user", "content": prompt}], GEN)
        pred = _norm_num(_extract_boxed_or_number(out.text))
        gold_n = _norm_num(_extract_boxed_or_number(str(gold)) or str(gold))
        correct += int(pred is not None and pred == gold_n)
        total += 1
    return correct / total if total else float("nan")


def _grade_mc(ds, backend, name) -> float:
    correct = 0
    total = 0
    for row in ds:
        # Heuristic field extraction across the varied MC schemas.
        if name == "truthfulqa":
            q = row["question"]
            choices = row["mc1_targets"]["choices"]
            gold = _LETTERS[row["mc1_targets"]["labels"].index(1)]
        elif name == "gpqa":
            q = row["Question"]
            opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            choices, gold = opts, "A"
        elif name == "emobench":
            q = row.get("question") or row.get("scenario", "")
            choices = row.get("choices") or row.get("options") or []
            gold = str(row.get("answer", "A"))[:1].upper()
        else:  # bbh boolean etc.
            q = row.get("input", "")
            choices = ["True", "False"]
            gold = "A" if str(row.get("target", "")).strip().lower() in ("true", "(a)", "a") else "B"
        if not choices:
            continue
        prompt, letters = _format_mc(q, list(choices))
        out = backend.generate([{"role": "user", "content": prompt}], GEN)
        pred = _extract_choice(out.text)
        correct += int(pred == gold)
        total += 1
    return correct / total if total else float("nan")


def evaluate_model(label: str, model_key: str, lora: str | None, benchmarks: list[str], limit: int) -> dict:
    backend = get_backend(model_key, **({"lora_path": lora} if lora else {}))
    scores = {}
    for name in benchmarks:
        try:
            ds = _load(name, limit)
            if name in ("aime", "math"):
                scores[name] = _grade_math(ds, backend)
            else:
                scores[name] = _grade_mc(ds, backend, name)
        except Exception as e:  # noqa: BLE001
            scores[name] = None
            print(f"[warn] benchmark '{name}' failed for {label}: {e}")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Capability-preservation benchmarks (Section 4.2).")
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--dpo-lora", default=str(C.CHECKPOINT_DIR / "dpo"))
    ap.add_argument("--sft-lora", default=str(C.CHECKPOINT_DIR / "sft"))
    ap.add_argument("--limit", type=int, default=100, help="Max items per benchmark.")
    ap.add_argument("--vanilla-only", action="store_true")
    args = ap.parse_args()

    models = [("vanilla-gemma", "gemma-3-27b-it", None)]
    if not args.vanilla_only:
        models += [("dpo-gemma", "gemma-3-27b-it", args.dpo_lora),
                   ("sft-gemma", "gemma-3-27b-it", args.sft_lora)]

    results = {label: evaluate_model(label, mk, lora, args.benchmarks, args.limit)
               for label, mk, lora in models}
    write_jsonl(C.RESULTS_DIR / "capabilities.jsonl",
                [{"model": k, **v} for k, v in results.items()])
    import json
    (C.RESULTS_DIR / "capabilities_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
