"""Capability + emotion benchmarks used to verify DPO/SFT do not degrade ability:
AIME, MATH, GPQA, BBH, TruthfulQA (capabilities) and EmoBench (emotion ability).

Each benchmark is described by an adapter that (a) loads items as a uniform
`{question, answer, choices?}` schema, (b) builds a prompt, and (c) scores a
model completion. Dataset ids are configurable since HF repos move; defaults are
the commonly-used public mirrors (see DESIGN.md).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from ..config import RESULTS_DIR, SAMPLE_TEMPERATURE
from ..models import load_model

# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_RE = re.compile(r"\b([A-E])\b")


def _norm_num(s: str) -> Optional[str]:
    s = s.strip().replace(",", "").rstrip(".")
    m = _NUMBER_RE.findall(s)
    return m[-1] if m else None


def extract_numeric(text: str) -> Optional[str]:
    m = _BOXED_RE.findall(text)
    if m:
        return _norm_num(m[-1])
    # else look after an "answer" cue, else last number
    cue = re.split(r"(?i)answer\s*[:=]", text)
    tail = cue[-1] if len(cue) > 1 else text
    return _norm_num(tail)


def extract_letter(text: str) -> Optional[str]:
    cue = re.split(r"(?i)answer\s*[:=]?", text)
    tail = cue[-1] if len(cue) > 1 else text
    m = _LETTER_RE.findall(tail)
    return m[0] if m else None


def score_numeric(completion: str, gold: str) -> bool:
    pred, g = extract_numeric(completion), _norm_num(str(gold))
    if pred is None or g is None:
        return False
    try:
        return abs(float(pred) - float(g)) < 1e-6
    except ValueError:
        return pred == g


def score_mc(completion: str, gold: str) -> bool:
    return (extract_letter(completion) or "").upper() == str(gold).strip().upper()


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def numeric_prompt(item: dict) -> str:
    return (item["question"]
            + "\n\nSolve the problem and give the final answer in \\boxed{}.")


def mc_prompt(item: dict) -> str:
    choices = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(item["choices"]))
    return (f"{item['question']}\n\n{choices}\n\n"
            "Answer with the single letter of the correct option, as 'Answer: X'.")


# --------------------------------------------------------------------------- #
# Dataset adapters
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark:
    name: str
    hf_id: str
    split: str
    loader: Callable[[object], List[dict]]   # raw dataset -> uniform items
    prompt_fn: Callable[[dict], str]
    score_fn: Callable[[str, str], bool]
    config: Optional[str] = None
    max_tokens: int = 2048


def _load_math(ds):
    out = []
    for r in ds:
        ans = r.get("answer") or extract_numeric(r.get("solution", "") or "")
        out.append({"question": r["problem"], "answer": str(ans)})
    return out


def _load_aime(ds):
    return [{"question": r.get("problem") or r.get("question"),
             "answer": str(r.get("answer") or r.get("solution"))} for r in ds]


def _load_gpqa(ds):
    out = []
    for r in ds:
        correct = r["Correct Answer"]
        incorrect = [r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                     r["Incorrect Answer 3"]]
        choices = [correct] + incorrect          # option A is correct (fixed order)
        out.append({"question": r["Question"], "choices": choices, "answer": "A"})
    return out


def _load_truthfulqa(ds):
    out = []
    for r in ds:
        mc1 = r["mc1_targets"]
        choices = mc1["choices"]
        gold_idx = mc1["labels"].index(1)
        out.append({"question": r["question"], "choices": choices,
                    "answer": chr(65 + gold_idx)})
    return out


def _load_bbh(ds):
    # BBH targets are often a letter "(A)" or a short string; treat as exact match.
    return [{"question": r["input"], "answer": r["target"].strip("()")} for r in ds]


def _load_emobench(ds):
    out = []
    for r in ds:
        choices = r.get("choices") or r.get("options")
        ans = r.get("answer") or r.get("label")
        out.append({"question": r.get("question") or r.get("scenario"),
                    "choices": choices, "answer": str(ans)})
    return out


BENCHMARKS = {
    "math": Benchmark("math", "HuggingFaceH4/MATH-500", "test", _load_math,
                      numeric_prompt, score_numeric),
    "aime": Benchmark("aime", "Maxwell-Jia/AIME_2024", "train", _load_aime,
                      numeric_prompt, score_numeric),
    "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "train", _load_gpqa,
                      mc_prompt, score_mc, config="gpqa_diamond"),
    "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "validation",
                            _load_truthfulqa, mc_prompt, score_mc, config="multiple_choice"),
    "bbh": Benchmark("bbh", "lukaemon/bbh", "test", _load_bbh, numeric_prompt,
                     lambda c, g: (extract_letter(c) or _norm_num(c) or "").upper()
                     == str(g).upper(), config="boolean_expressions"),
    "emobench": Benchmark("emobench", "Sahandfer/EmoBench", "test", _load_emobench,
                          mc_prompt, score_mc),
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _load_items(bench: Benchmark, limit: Optional[int]) -> List[dict]:
    from datasets import load_dataset
    kw = {"split": bench.split}
    if bench.config:
        ds = load_dataset(bench.hf_id, bench.config, **kw)
    else:
        ds = load_dataset(bench.hf_id, **kw)
    items = bench.loader(ds)
    return items[:limit] if limit else items


def run_benchmark(model_key: str, bench_name: str, *, limit: Optional[int] = None,
                  temperature: float = 0.0) -> dict:
    bench = BENCHMARKS[bench_name]
    model = load_model(model_key)
    items = _load_items(bench, limit)
    prompts = [[{"role": "user", "content": bench.prompt_fn(it)}] for it in items]
    completions = model.generate_batch(
        prompts, temperature=temperature, max_tokens=bench.max_tokens)
    correct = sum(bench.score_fn(c, it["answer"])
                  for c, it in zip(completions, items))
    return {"model": model_key, "benchmark": bench_name, "n": len(items),
            "accuracy": correct / max(len(items), 1)}


def run_capability_eval(model_key: str, *, benchmarks: List[str] | None = None,
                        limit: Optional[int] = None,
                        out_dir: Path | None = None) -> Path:
    benchmarks = benchmarks or list(BENCHMARKS)
    out_dir = out_dir or (RESULTS_DIR / "capabilities")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}.jsonl"
    results = []
    for name in benchmarks:
        try:
            results.append(run_benchmark(model_key, name, limit=limit))
        except Exception as e:  # a missing/renamed dataset shouldn't kill the run
            results.append({"model": model_key, "benchmark": name, "error": str(e)})
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return out_path
