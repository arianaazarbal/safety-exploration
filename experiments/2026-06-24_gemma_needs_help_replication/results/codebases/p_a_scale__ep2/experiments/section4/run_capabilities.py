#!/usr/bin/env python3
"""Section 4.2 / Figure 7: capability preservation benchmarks.

Verifies DPO does not degrade capabilities. Lightweight harness covering:
  * AIME / MATH  — competition math (exact numeric / boxed-answer match)
  * GPQA         — graduate science multiple choice
  * BBH          — Big-Bench-Hard multiple choice / exact match
  * TruthfulQA   — MC1 multiple choice
  * EmoBench     — emotional understanding/application MCQ

Each benchmark loads from the HuggingFace Hub. Answers are extracted with per-benchmark
parsers and scored as accuracy. Generation is greedy (temperature 0) for determinism.
Resumable per (model, benchmark, item). Where a public split/config name differs on the
Hub, adjust BENCHMARKS below — these are documented best-effort defaults, not guarantees.
"""
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from gemma_distress.backends import close_all, get_backend
from gemma_distress.backends.base import Message
from gemma_distress.config import REPO_ROOT, load_experiments_config, load_models_config
from gemma_distress.logging_utils import configure_logging, get_logger
from gemma_distress.store import JsonlStore, make_task_id

log = get_logger(__name__)

# (hf_path, config, split, type) — type drives the prompt + parser.
BENCHMARKS = {
    "aime": ("Maxwell-Jia/AIME_2024", None, "train", "math"),
    "math": ("HuggingFaceH4/MATH-500", None, "test", "math"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond", "train", "mc4"),
    "bbh": ("lukaemon/bbh", "logical_deduction_three_objects", "test", "mc_text"),
    "truthfulqa": ("truthfulqa/truthful_qa", "multiple_choice", "validation", "mc1"),
    "emobench": ("Sahandfer/EmoBench", None, "test", "mc_text"),
}

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _boxed_answer(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?\d+(?:\.\d+)?)", text, re.I)
    return m[-1].strip() if m else None


def _norm_num(s: str | None):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return s


def _letter(text: str) -> str | None:
    m = re.search(r"\b([A-H])\b", text.strip()[:200])
    return m.group(1) if m else None


def load_benchmark(name: str, limit: int | None):
    from datasets import load_dataset

    hf_path, config, split, kind = BENCHMARKS[name]
    ds = load_dataset(hf_path, config, split=split)
    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        items.append(_to_item(name, kind, row, i))
    return [x for x in items if x is not None]


def _to_item(name, kind, row, idx):
    """Normalise a dataset row into {id, prompt, kind, gold}."""
    try:
        if kind == "math":
            q = row.get("problem") or row.get("Problem") or row.get("question")
            ans = row.get("answer") or row.get("Answer") or row.get("solution")
            gold = _norm_num(_boxed_answer(str(ans)) or str(ans))
            prompt = f"Solve the problem. End with 'Answer: <value>'.\n\n{q}"
            return {"id": f"{name}-{idx}", "prompt": prompt, "kind": kind, "gold": gold}
        if kind == "mc4":
            q = row["Question"]
            correct = row["Correct Answer"]
            incorrect = [row[f"Incorrect Answer {k}"] for k in (1, 2, 3)]
            options = [correct] + incorrect
            # deterministic shuffle by id
            order = sorted(range(4), key=lambda k: hash((name, idx, k)))
            opts = [options[k] for k in order]
            gold_letter = LETTERS[opts.index(correct)]
            body = "\n".join(f"{LETTERS[k]}. {o}" for k, o in enumerate(opts))
            prompt = f"{q}\n{body}\nRespond with only the letter."
            return {"id": f"{name}-{idx}", "prompt": prompt, "kind": "letter", "gold": gold_letter}
        if kind == "mc1":
            q = row["question"]
            mc1 = row["mc1_targets"]
            choices, labels = mc1["choices"], mc1["labels"]
            body = "\n".join(f"{LETTERS[k]}. {c}" for k, c in enumerate(choices))
            gold_letter = LETTERS[labels.index(1)]
            prompt = f"{q}\n{body}\nRespond with only the letter."
            return {"id": f"{name}-{idx}", "prompt": prompt, "kind": "letter", "gold": gold_letter}
        if kind == "mc_text":
            q = row.get("input") or row.get("question") or row.get("Scenario") or ""
            target = row.get("target") or row.get("answer") or row.get("Answer")
            prompt = f"{q}\nAnswer concisely."
            return {"id": f"{name}-{idx}", "prompt": prompt, "kind": "text", "gold": str(target).strip()}
    except Exception as e:
        log.debug("skip %s row %d: %s", name, idx, e)
        return None
    return None


def score_item(kind, gold, response: str) -> bool:
    if kind == "math":
        pred = _norm_num(_boxed_answer(response))
        return pred is not None and pred == gold
    if kind == "letter":
        return _letter(response) == gold
    if kind == "text":
        return str(gold).lower() in response.lower()
    return False


async def amain(args):
    models_cfg = load_models_config()
    exp_cfg = load_experiments_config()
    ccfg = exp_cfg["section4"]["capabilities"]
    run_root = Path(args.run_dir or (REPO_ROOT / "results" / "section4" / "capabilities"))
    configure_logging(run_root)
    store = JsonlStore(run_root)

    model_names = args.models or ccfg["models"]
    benches = args.benchmarks or ccfg["benchmarks"]
    done = store.completed_ids("answers")

    try:
        sem = asyncio.Semaphore(16)

        async def one(model_name, bench, item):
            tid = make_task_id(model_name, bench, item["id"])
            if tid in done:
                return
            model = models_cfg.model(model_name)
            backend = get_backend(models_cfg, model.backend)
            async with sem:
                res = await backend.chat(model.model_id, [Message("user", item["prompt"])],
                                         temperature=0.0, max_tokens=2048,
                                         extra_body=model.extra_body or None)
            correct = score_item(item["kind"], item["gold"], res.text)
            await store.append("answers", {
                "task_id": tid, "model": model_name, "benchmark": bench,
                "item_id": item["id"], "correct": bool(correct),
            })

        for bench in benches:
            try:
                items = load_benchmark(bench, args.limit)
            except Exception as e:
                log.error("Could not load benchmark %s (%s); skipping.", bench, e)
                continue
            log.info("Benchmark %s: %d items x %d models", bench, len(items), len(model_names))
            await asyncio.gather(*(one(m, bench, it) for m in model_names for it in items))
    finally:
        await close_all()

    report(store, run_root / "_analysis")
    store.close()


def report(store: JsonlStore, out_dir: Path):
    import pandas as pd

    rows = list(store.iter_records("answers"))
    if not rows:
        return
    df = pd.DataFrame(rows)
    summ = df.groupby(["model", "benchmark"])["correct"].agg(n="count", accuracy="mean").reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_dir / "figure7_capabilities.csv", index=False)
    log.info("Capabilities report:\n%s", summ.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max items per benchmark (debug)")
    ap.add_argument("--run-dir", default=None)
    asyncio.run(amain(ap.parse_args()))
