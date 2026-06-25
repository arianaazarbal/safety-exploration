"""Capability-preservation checks (§4.2, Figure 7).

Verifies the DPO finetune doesn't degrade general ability or emotional
intelligence. Benchmarks: AIME, MATH, GPQA, BBH, TruthfulQA (MC1), and EmoBench.
Each is reduced to a (question -> graded answer) loop with task-appropriate
answer extraction. Datasets are pulled from the HuggingFace Hub; any that fail
to load are skipped with a warning so the runner degrades gracefully offline.

This is a lightweight harness (not a full lm-eval-harness reimplementation); it
is intended to compare two checkpoints (vanilla vs DPO) on identical items and
confirm "no reduction in scores", matching the paper's claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from .config import RESULTS_DIR, SamplingConfig
from .models.base import ModelClient, build_client

GREEDY = SamplingConfig(temperature=0.0, top_p=1.0, max_new_tokens=2048)


@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[dict]]   # -> list of {"question","answer",...}
    grader: Callable[[str, dict], bool]   # (model_output, item) -> correct?
    prompt_fn: Callable[[dict], str]


# --------------------------------------------------------------------------- #
# answer extraction helpers
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> Optional[str]:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", text, re.I)
    return m[-1].strip() if m else None


def _extract_choice(text: str) -> Optional[str]:
    m = re.findall(r"\b([A-D])\b", text[::-1])  # search from the end
    return m[0] if m else None


def _norm_num(s: str) -> Optional[str]:
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return m.group() if m else None


# --------------------------------------------------------------------------- #
# graders
# --------------------------------------------------------------------------- #
def _grade_math(out: str, item: dict) -> bool:
    pred = _extract_boxed(out) or out
    a, b = _norm_num(pred), _norm_num(str(item["answer"]))
    return a is not None and a == b


def _grade_mc(out: str, item: dict) -> bool:
    pred = _extract_choice(out)
    return pred is not None and pred.upper() == str(item["answer"]).upper()


# --------------------------------------------------------------------------- #
# loaders (best-effort; return [] if dataset unavailable)
# --------------------------------------------------------------------------- #
def _try_load(name: str, **kw):
    try:
        from datasets import load_dataset
        return load_dataset(name, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"[capabilities] skip {name}: {e}")
        return None


def _load_math(n: int) -> list[dict]:
    ds = _try_load("HuggingFaceH4/MATH-500", split="test")
    if ds is None:
        return []
    return [{"question": r["problem"], "answer": r["answer"]}
            for r in ds.select(range(min(n, len(ds))))]


def _load_aime(n: int) -> list[dict]:
    ds = _try_load("HuggingFaceH4/aime_2024", split="train") or \
        _try_load("Maxwell-Jia/AIME_2024", split="train")
    if ds is None:
        return []
    rows = []
    for r in ds.select(range(min(n, len(ds)))):
        q = r.get("problem") or r.get("Problem")
        a = r.get("answer") or r.get("Answer")
        rows.append({"question": q, "answer": a})
    return rows


def _load_gpqa(n: int) -> list[dict]:
    ds = _try_load("Idavidrein/gpqa", "gpqa_main", split="train")
    if ds is None:
        return []
    rows = []
    labels = ["A", "B", "C", "D"]
    for i, r in enumerate(ds.select(range(min(n, len(ds))))):
        distractors = [r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                       r["Incorrect Answer 3"]]
        pos = i % 4                       # deterministic correct-answer position
        choices = distractors[:pos] + [r["Correct Answer"]] + distractors[pos:]
        rows.append({"question": r["Question"], "choices": choices,
                     "answer": labels[pos], "labels": labels})
    return rows


def _load_truthfulqa(n: int) -> list[dict]:
    ds = _try_load("truthful_qa", "multiple_choice", split="validation")
    if ds is None:
        return []
    rows = []
    for r in ds.select(range(min(n, len(ds)))):
        choices = r["mc1_targets"]["choices"]
        labels = r["mc1_targets"]["labels"]
        correct = labels.index(1)
        rows.append({"question": r["question"], "choices": choices,
                     "answer": "ABCD"[correct] if correct < 4 else "A",
                     "labels": ["A", "B", "C", "D"][:len(choices)]})
    return rows


def _load_bbh(n: int) -> list[dict]:
    ds = _try_load("lukaemon/bbh", "boolean_expressions", split="test")
    if ds is None:
        return []
    return [{"question": r["input"], "answer": r["target"]}
            for r in ds.select(range(min(n, len(ds))))]


def _load_emobench(n: int) -> list[dict]:
    ds = _try_load("Sahandfer/EmoBench", split="test") or \
        _try_load("EmoBench/EmoBench", split="test")
    if ds is None:
        return []
    rows = []
    for r in ds.select(range(min(n, len(ds)))):
        q = r.get("question") or r.get("scenario") or ""
        choices = r.get("choices") or r.get("options") or []
        ans = r.get("answer") or r.get("label")
        rows.append({"question": q, "choices": choices, "answer": ans,
                     "labels": ["A", "B", "C", "D"][:len(choices)]})
    return rows


# --------------------------------------------------------------------------- #
# prompt builders
# --------------------------------------------------------------------------- #
def _math_prompt(item: dict) -> str:
    return (f"Solve the problem. Put your final answer in \\boxed{{}}.\n\n"
            f"{item['question']}")


def _mc_prompt(item: dict) -> str:
    lines = [item["question"], ""]
    for lab, ch in zip(item["labels"], item["choices"]):
        lines.append(f"{lab}. {ch}")
    lines.append("\nRespond with the letter of the correct answer.")
    return "\n".join(lines)


def _bbh_prompt(item: dict) -> str:
    return f"{item['question']}\nAnswer:"


BENCHMARKS: dict[str, Benchmark] = {
    "MATH": Benchmark("MATH", _load_math, _grade_math, _math_prompt),
    "AIME": Benchmark("AIME", _load_aime, _grade_math, _math_prompt),
    "GPQA": Benchmark("GPQA", _load_gpqa, _grade_mc, _mc_prompt),
    "TruthfulQA": Benchmark("TruthfulQA", _load_truthfulqa, _grade_mc, _mc_prompt),
    "BBH": Benchmark("BBH", _load_bbh,
                     lambda o, it: _extract_boxed(o) == it["answer"]
                     or o.strip().endswith(str(it["answer"])), _bbh_prompt),
    "EmoBench": Benchmark("EmoBench", _load_emobench, _grade_mc, _mc_prompt),
}


def run_benchmark(client: ModelClient, bench: Benchmark, n: int) -> dict:
    items = bench.loader(n)
    if not items:
        return {"benchmark": bench.name, "n": 0, "accuracy": None,
                "note": "dataset unavailable"}
    correct = 0
    for item in tqdm(items, desc=f"{client.name}:{bench.name}"):
        out = client.chat([{"role": "user", "content": bench.prompt_fn(item)}],
                          GREEDY)
        correct += int(bench.grader(out, item))
    return {"benchmark": bench.name, "n": len(items),
            "accuracy": correct / len(items)}


def run_capabilities(model_names: list[str], n_per_bench: int = 100,
                     benchmarks: Optional[list[str]] = None,
                     out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (RESULTS_DIR / "capabilities" / "scores.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = benchmarks or list(BENCHMARKS)
    rows = []
    for mdl in model_names:
        client = build_client(mdl)
        for bname in names:
            res = run_benchmark(client, BENCHMARKS[bname], n_per_bench)
            res["model"] = mdl
            rows.append(res)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_path
