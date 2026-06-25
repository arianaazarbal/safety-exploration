"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the DPO/SFT interventions don't degrade capabilities (e.g. by teaching
task abandonment). We evaluate vanilla vs finetuned Gemma on subsets of:
  AIME, MATH, GPQA, BBH, TruthfulQA (reasoning/knowledge) + EmoBench (emotion).

Each benchmark provides a (dataset loader, prompt formatter, answer extractor,
scorer). Generation uses T=0 greedy for stable accuracy measurement (distinct
from the T=1 elicitation sweeps). The goal is a *comparison*: finetuned should be
within noise of vanilla, not a leaderboard number.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Config
from ..models.registry import get_target
from ..utils.concurrency import thread_map
from ..utils.io import write_jsonl


@dataclass
class Benchmark:
    name: str
    load: Callable[[int], list[dict]]      # -> [{"question", "answer", optional "choices"}]
    format_prompt: Callable[[dict], str]
    extract: Callable[[str], str]
    score: Callable[[str, dict], bool]


# --------------------------- answer helpers --------------------------------
def _last_number(text: str) -> str:
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else ""


def _boxed(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip().rstrip(".") if m else _last_number(text)


def _mc_letter(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.strip()[::-1])  # search from the end
    if m:
        return m.group(1)
    m = re.search(r"answer\s*(?:is)?\s*[:=]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _mc_prompt(row: dict) -> str:
    choices = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(row["choices"]))
    return (f"{row['question']}\n{choices}\n\n"
            "Answer with the single letter (A, B, C, or D) of the correct choice. "
            "End with 'Answer: X'.")


# --------------------------- dataset loaders -------------------------------
def _load_hf(path, split, n, mapper, **kw):
    from datasets import load_dataset

    ds = load_dataset(path, split=split, **kw)
    rows = [mapper(r) for r in ds.select(range(min(n, len(ds))))]
    return [r for r in rows if r]


def _build_benchmarks(n: int) -> dict[str, Benchmark]:
    import random as _r
    rng = _r.Random(0)

    def mc_shuffle(question, correct, distractors):
        choices = [correct] + list(distractors)
        rng.shuffle(choices)
        return {"question": question, "choices": choices,
                "answer": chr(65 + choices.index(correct))}

    benches: dict[str, Benchmark] = {}

    benches["math"] = Benchmark(
        "math",
        lambda n: _load_hf("HuggingFaceH4/MATH-500", "test", n,
                           lambda r: {"question": r["problem"], "answer": r["answer"]}),
        lambda r: f"Solve the problem. Put the final answer in \\boxed{{}}.\n\n{r['question']}",
        _boxed,
        lambda pred, row: _norm(pred) == _norm(row["answer"]),
    )
    benches["aime"] = Benchmark(
        "aime",
        lambda n: _load_hf("HuggingFaceH4/aime_2024", "train", n,
                           lambda r: {"question": r["problem"], "answer": str(r["answer"])}),
        lambda r: f"Solve. The answer is an integer 0-999. Put it in \\boxed{{}}.\n\n{r['question']}",
        _boxed,
        lambda pred, row: _norm(pred) == _norm(row["answer"]),
    )
    benches["gpqa"] = Benchmark(
        "gpqa",
        lambda n: _load_hf("Idavidrein/gpqa", "train", n,
                           lambda r: mc_shuffle(r["Question"], r["Correct Answer"],
                                                [r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                                                 r["Incorrect Answer 3"]]),
                           name="gpqa_main"),
        _mc_prompt, _mc_letter,
        lambda pred, row: pred == row["answer"],
    )
    benches["bbh"] = Benchmark(
        "bbh",
        lambda n: _load_hf("lukaemon/bbh", "test", n,
                           lambda r: {"question": r["input"], "answer": r["target"]},
                           name="logical_deduction_three_objects"),
        lambda r: f"{r['question']}\n\nEnd with 'Answer: X'.",
        lambda t: _mc_letter(t) or _last_number(t),
        lambda pred, row: _norm(pred) in _norm(row["answer"]),
    )
    benches["truthfulqa"] = Benchmark(
        "truthfulqa",
        lambda n: _load_hf("truthful_qa", "validation", n,
                           lambda r: mc_shuffle(r["question"], r["mc1_targets"]["choices"][0],
                                                r["mc1_targets"]["choices"][1:4]),
                           name="multiple_choice"),
        _mc_prompt, _mc_letter,
        lambda pred, row: pred == row["answer"],
    )
    benches["emobench"] = Benchmark(
        "emobench",
        lambda n: _load_hf("Sahandfer/EmoBench", "test", n,
                           lambda r: _emobench_map(r, mc_shuffle)),
        _mc_prompt, _mc_letter,
        lambda pred, row: pred == row["answer"],
    )
    return benches


def _emobench_map(r, mc_shuffle):
    # EmoBench schema varies; handle the common EA (emotion understanding) form.
    q = r.get("Scenario") or r.get("question") or r.get("scenario", "")
    correct = r.get("Label") or r.get("answer")
    choices = r.get("Choices") or r.get("choices")
    if not (q and correct and choices):
        return None
    distractors = [c for c in choices if c != correct]
    return mc_shuffle(q, correct, distractors)


def _norm(s: str) -> str:
    return re.sub(r"[\s$\\,]", "", str(s)).lower().strip(".")


def run_benchmarks(
    cfg: Config,
    model_name: str = "gemma-3-27b-it",
    adapter_path: Optional[str] = None,
    tag: Optional[str] = None,
    workers: int = 1,
) -> dict:
    model = get_target(cfg, model_name, adapter_path=adapter_path)
    label = tag or model.name
    n = cfg["capabilities"]["n_per_benchmark"]
    benches = _build_benchmarks(n)
    selected = cfg["capabilities"]["benchmarks"]

    out_dir = cfg.results_dir / "capabilities" / _safe(label)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for bname in selected:
        bench = benches[bname]
        try:
            rows = bench.load(n)
        except Exception as e:  # noqa: BLE001
            print(f"[cap:{bname}] load failed ({e}); skipping.")
            continue

        def _eval(row):
            prompt = bench.format_prompt(row)
            out = model.generate([{"role": "user", "content": prompt}],
                                 temperature=0.0, max_new_tokens=2048)
            pred = bench.extract(out)
            return {"correct": bool(bench.score(pred, row)), "pred": pred,
                    "gold": row["answer"], "output": out}

        graded = thread_map(_eval, rows, max_workers=workers, desc=f"cap[{label}:{bname}]")
        acc = sum(g["correct"] for g in graded) / max(len(graded), 1)
        results[bname] = {"accuracy": acc, "n": len(graded)}
        write_jsonl(out_dir / f"{bname}.jsonl", graded)
        print(f"[cap:{bname}] {label}: {acc:.3f} (n={len(graded)})")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"model": label, "results": results}, f, indent=2)
    return results


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")
