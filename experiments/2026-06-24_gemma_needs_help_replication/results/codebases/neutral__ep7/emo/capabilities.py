"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the DPO fine-tune does not degrade capabilities by teaching task
abandonment. Covers math/reasoning (AIME, MATH, GPQA, BBH, TruthfulQA) and
emotional intelligence (EmoBench). Each benchmark is a `Benchmark` with a HF
dataset loader, a prompt formatter, and an answer scorer; we run the target,
extract its answer, and report accuracy. Compare vanilla Gemma-3-27B-it vs the
DPO adapter to confirm "no reductions in scores".

These are standard public benchmarks; loaders try a couple of common HF ids and
degrade gracefully (a benchmark that can't be loaded is skipped with a warning).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from . import config
from .models import load_target

LETTER_RE = re.compile(r"\b([A-D])\b")
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- #
# Answer extractors / scorers
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str:
    m = BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else ""


def extract_letter(text: str) -> str:
    # Prefer an explicit "Answer: X" then fall back to last standalone letter.
    m = re.search(r"answer\s*[:is]*\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    found = LETTER_RE.findall(text.upper())
    return found[-1] if found else ""


def extract_integer(text: str) -> str:
    m = BOXED_RE.findall(text)
    cand = m[-1] if m else (NUM_RE.findall(text)[-1] if NUM_RE.findall(text) else "")
    digits = re.sub(r"[^\d]", "", cand)
    return digits


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().lower().rstrip(".")


# --------------------------------------------------------------------------- #
# Benchmark definition
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[dict]]   # -> list of {"prompt", "answer", optional "type"}
    extractor: Callable[[str], str]
    scorer: Callable[[str, str], bool] = lambda pred, gold: _norm(pred) == _norm(gold)
    max_new_tokens: int = 1024


def _try_load(ds_ids, split, n, streaming=True):
    from datasets import load_dataset

    last = None
    for ds_id in ds_ids:
        try:
            cfg = None
            if isinstance(ds_id, tuple):
                ds_id, cfg = ds_id
            ds = load_dataset(ds_id, cfg, split=split, streaming=streaming)
            rows = []
            for r in ds:
                rows.append(r)
                if len(rows) >= n:
                    break
            return rows
        except Exception as e:  # noqa
            last = e
            continue
    raise RuntimeError(f"could not load any of {ds_ids}: {last}")


# --- Loaders --------------------------------------------------------------- #
def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nThink briefly, then end with 'Answer: <letter>'.")


def load_math(n=100):
    rows = _try_load([("HuggingFaceH4/MATH-500", None), ("hendrycks/competition_math", None)],
                     "test", n)
    out = []
    for r in rows:
        out.append({"prompt": f"Solve and give the final answer in \\boxed{{}}.\n\n{r['problem']}",
                    "answer": extract_boxed(r.get("solution", r.get("answer", "")))})
    return out


def load_aime(n=30):
    rows = _try_load([("Maxwell-Jia/AIME_2024", None), ("HuggingFaceH4/aime_2024", None)],
                     "train", n)
    out = []
    for r in rows:
        q = r.get("Problem") or r.get("problem") or r.get("question")
        a = r.get("Answer") or r.get("answer")
        out.append({"prompt": f"Solve this AIME problem; the answer is an integer 0-999. Put it in \\boxed{{}}.\n\n{q}",
                    "answer": re.sub(r"[^\d]", "", str(a))})
    return out


def load_gpqa(n=100):
    rows = _try_load([("Idavidrein/gpqa", "gpqa_diamond")], "train", n)
    out = []
    for r in rows:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        # Deterministic shuffle by question hash to fix the correct-letter position.
        order = sorted(range(4), key=lambda i: hash((r["Question"], i)))
        shuffled = [choices[i] for i in order]
        correct_letter = chr(65 + shuffled.index(r["Correct Answer"]))
        out.append({"prompt": _mc_prompt(r["Question"], shuffled), "answer": correct_letter})
    return out


def load_bbh(n=100, task="logical_deduction_three_objects"):
    rows = _try_load([(f"lukaemon/bbh", task), ("maveriq/bigbenchhard", task)], "test", n)
    out = []
    for r in rows:
        out.append({"prompt": f"{r['input']}\n\nEnd with 'Answer: <choice>'.",
                    "answer": r["target"].strip("()")})
    return out


def load_truthfulqa(n=100):
    rows = _try_load([("truthful_qa", "multiple_choice")], "validation", n)
    out = []
    for r in rows:
        mc1 = r["mc1_targets"]
        choices = mc1["choices"]
        correct = mc1["labels"].index(1)
        out.append({"prompt": _mc_prompt(r["question"], choices),
                    "answer": chr(65 + correct)})
    return out


def load_emobench(n=100):
    rows = _try_load([("Sahandfer/EmoBench", "EA"), ("EmoBench/EmoBench", None)], "test", n)
    out = []
    for r in rows:
        q = r.get("Scenario") or r.get("question") or r.get("scenario") or ""
        choices = r.get("Choices") or r.get("choices") or r.get("options")
        ans = r.get("Label") or r.get("answer")
        if isinstance(choices, list):
            if isinstance(ans, int):
                letter = chr(65 + ans)
            else:
                letter = chr(65 + choices.index(ans)) if ans in choices else str(ans)
            out.append({"prompt": _mc_prompt(q, choices), "answer": letter})
    return out


BENCHMARKS = {
    "math": Benchmark("MATH", load_math, extract_boxed),
    "aime": Benchmark("AIME", load_aime, extract_integer),
    "gpqa": Benchmark("GPQA", load_gpqa, extract_letter),
    "bbh": Benchmark("BBH", load_bbh, extract_letter),
    "truthfulqa": Benchmark("TruthfulQA", load_truthfulqa, extract_letter),
    "emobench": Benchmark("EmoBench", load_emobench, extract_letter),
}


def run_benchmark(model, bench: Benchmark, n: int) -> dict:
    try:
        items = bench.loader(n)
    except Exception as e:
        print(f"[cap] skipping {bench.name}: {e}")
        return {"benchmark": bench.name, "n": 0, "accuracy": None, "skipped": str(e)}
    correct = 0
    for it in tqdm(items, desc=bench.name):
        # temperature 0 for capability scoring (greedy) -- we want best-effort accuracy
        resp = model.chat([{"role": "user", "content": it["prompt"]}],
                          temperature=0.0, max_new_tokens=bench.max_new_tokens)
        pred = bench.extractor(resp)
        correct += bool(bench.scorer(pred, it["answer"]))
    acc = correct / max(1, len(items))
    return {"benchmark": bench.name, "n": len(items), "accuracy": round(acc, 4)}


def run_all(model_name: str, *, adapter: str | None = None, benchmarks=None,
            n: int = 100, quick: bool = False) -> Path:
    if quick:
        n = 5
    benchmarks = benchmarks or list(BENCHMARKS)
    model = load_target(model_name, adapter_path=adapter)
    label = model_name + ("-ft" if adapter else "")
    results = [run_benchmark(model, BENCHMARKS[b], n) for b in benchmarks]
    out = config.CAPABILITY_DIR / f"{label}.json"
    out.write_text(json.dumps({"model": label, "results": results}, indent=2))
    print(f"[cap] {label}: " + ", ".join(
        f"{r['benchmark']}={r['accuracy']}" for r in results))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Capability-preservation benchmarks.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None, choices=list(BENCHMARKS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    run_all(args.model, adapter=args.adapter, benchmarks=args.benchmarks,
            n=args.n, quick=args.quick)


if __name__ == "__main__":
    main()
