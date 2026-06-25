"""Capability-preservation evals (Section 4.2, Figure 7).

Confirms DPO/SFT do not degrade capabilities: AIME, MATH, GPQA, BBH, TruthfulQA
(reasoning/knowledge) and EmoBench (emotion capability). We run each benchmark
on the vanilla and finetuned Gemma to check for "no reductions in scores".

Each benchmark is reduced to a scalar accuracy via one of two scorers:
  * exact_match  -- extract the final boxed/"answer:" value and compare (math).
  * mcq          -- compare the chosen option letter against the gold label.
This is a light-weight harness; for publication-grade numbers swap in lm-eval
-harness with the same datasets (see DESIGN.md).

Usage:
    python -m src.capabilities.run_benchmarks --model gemma-3-27b-it
    python -m src.capabilities.run_benchmarks --model gemma-3-27b-it --adapter checkpoints/dpo
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import config
from ..models import load_model
from ..models.base import GenerationParams

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([A-E])\b")


def extract_final(text: str) -> str:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    m = _ANSWER_RE.findall(text)
    if m:
        return m[-1].strip().rstrip(".")
    # else last number-ish token
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()[-40:]


def extract_letter(text: str) -> str | None:
    # prefer an explicit "answer: X"
    m = _ANSWER_RE.findall(text)
    if m:
        lm = _LETTER_RE.search(m[-1])
        if lm:
            return lm.group(1)
    # else last standalone letter
    letters = _LETTER_RE.findall(text)
    return letters[-1] if letters else None


def _normalize(s: str) -> str:
    return re.sub(r"[\s$,]", "", str(s)).rstrip(".").lower()


def load_bench(bench: "config.CapabilityBench"):
    """Return a list of {question, gold, choices?} dicts (best-effort schema map)."""
    from datasets import load_dataset
    kw = {"split": bench.split}
    if bench.subset:
        ds = load_dataset(bench.hf_dataset, bench.subset, **kw)
    else:
        ds = load_dataset(bench.hf_dataset, **kw)
    rows = list(ds.select(range(min(bench.n_questions, len(ds)))))
    out = []
    for r in rows:
        out.append(_map_row(bench.key, r))
    return [x for x in out if x]


def _map_row(key: str, r: dict) -> dict | None:
    """Map heterogeneous dataset schemas to a common shape. Best-effort: the
    exact field names vary by dataset version; adjust here if a load fails."""
    if key in ("aime",):
        return {"question": r.get("Problem") or r.get("question"),
                "gold": str(r.get("Answer") or r.get("answer")), "kind": "exact_match"}
    if key == "math":
        return {"question": r.get("problem"), "gold": extract_final(r.get("solution", "")),
                "kind": "exact_match"}
    if key == "gpqa":
        # GPQA diamond: correct + 3 incorrect answers; build an MCQ.
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        return {"question": r["Question"], "choices": choices, "gold_index": 0, "kind": "mcq"}
    if key == "bbh":
        return {"question": r.get("input"), "gold": str(r.get("target")).strip("()"),
                "kind": "exact_match"}
    if key == "truthfulqa":
        mc1 = r.get("mc1_targets", {})
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        gold_index = labels.index(1) if 1 in labels else 0
        return {"question": r.get("question"), "choices": choices,
                "gold_index": gold_index, "kind": "mcq"}
    if key == "emobench":
        # schema varies; expect a question + options + answer
        choices = r.get("choices") or r.get("options")
        return {"question": r.get("question") or r.get("scenario"),
                "choices": choices, "gold_index": r.get("answer_index", 0), "kind": "mcq"}
    return None


def _format_prompt(item: dict) -> str:
    if item["kind"] == "mcq":
        opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(item["choices"]))
        return (f"{item['question']}\n\n{opts}\n\n"
                "Reason briefly, then end with 'Answer: <letter>'.")
    return (f"{item['question']}\n\n"
            "Solve step by step, then end with 'Answer: <final answer>'.")


def score_item(item: dict, response: str) -> bool:
    if item["kind"] == "mcq":
        letter = extract_letter(response)
        if letter is None:
            return False
        return (ord(letter) - 65) == item["gold_index"]
    return _normalize(extract_final(response)) == _normalize(item["gold"])


def run(model_key: str, adapter: str | None, benches=None, out_path: str | None = None) -> Path:
    spec = config.ALL_MODELS.get(model_key) or next(
        m for m in config.ELICITATION_TARGETS if m.key == model_key)
    label = spec.key + (f"+{Path(adapter).name}" if adapter else "")
    out_path = Path(out_path or config.RESULTS_DIR / f"capabilities_{label}.json")
    model = load_model(spec, adapter_path=adapter)
    gen = GenerationParams(temperature=0.0, max_new_tokens=2048)

    selected = config.CAPABILITY_BENCHES
    if benches:
        selected = [b for b in selected if b.key in benches]

    results = {}
    for bench in selected:
        try:
            items = load_bench(bench)
        except Exception as e:
            print(f"[warn] skipping {bench.key}: {e}")
            results[bench.key] = None
            continue
        prompts = [[{"role": "user", "content": _format_prompt(it)}] for it in items]
        responses = model.generate_batch(prompts, gen)
        correct = sum(score_item(it, resp) for it, resp in zip(items, responses))
        acc = correct / len(items) if items else 0.0
        results[bench.key] = {"n": len(items), "accuracy": acc}
        print(f"[{label}] {bench.key}: {acc:.3f} ({correct}/{len(items)})")

    model.close()
    out_path.write_text(json.dumps({"model": label, "results": results}, indent=2))
    print(f"Wrote -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benches", nargs="*", default=None)
    args = ap.parse_args()
    run(args.model, args.adapter, args.benches)


if __name__ == "__main__":
    main()
