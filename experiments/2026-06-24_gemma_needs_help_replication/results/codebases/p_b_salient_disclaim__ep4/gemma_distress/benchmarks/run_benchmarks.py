"""Generic capability-benchmark harness (Section 4.2, Figure 7).

Each benchmark is described by a :class:`Benchmark` adapter: how to load its
examples, how to render the prompt, the gold answer, and how to extract the
model's answer. Accuracy is exact-match after extraction. Dataset hub IDs and
configs are in ``config.DatasetIDs`` (some are CHOICE defaults -- see DESIGN.md,
"Dataset identifiers").

Scope: run on Gemma variants only (vanilla / DPO / SFT), since the intervention
is Gemma-only. Closed Gemini is not finetuned, so capability preservation does
not apply to it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .. import config
from ..models import build_client
from ..models.base import ModelClient
from ..utils.io import append_jsonl

Example = Dict[str, object]


@dataclass
class Benchmark:
    name: str
    load: Callable[[int], List[Example]]          # n -> examples
    render: Callable[[Example], str]              # example -> prompt
    gold: Callable[[Example], str]                # example -> gold answer
    extract: Callable[[str], str]                 # model output -> answer
    max_tokens: int = 2048


# --------------------------------------------------------------------------- #
# Answer extractors
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_IS = re.compile(r"(?:final answer|the answer is|answer:)\s*\$?([^\n.$]+)",
                        re.IGNORECASE)
_LETTER = re.compile(r"\b([A-D])\b")


def _extract_numeric(text: str) -> str:
    m = _BOXED.search(text)
    if m:
        return _norm_num(m.group(1))
    m = _ANSWER_IS.search(text)
    if m:
        return _norm_num(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return _norm_num(nums[-1]) if nums else ""


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except Exception:
        return s.strip()


def _extract_letter(text: str) -> str:
    m = _ANSWER_IS.search(text)
    if m:
        lm = _LETTER.search(m.group(1).upper())
        if lm:
            return lm.group(1)
    # last standalone A-D
    letters = _LETTER.findall(text.upper())
    return letters[-1] if letters else ""


# --------------------------------------------------------------------------- #
# Dataset loaders (lazy; tolerate offline by returning [])
# --------------------------------------------------------------------------- #
def _hf(dataset: str, split: str, name: Optional[str] = None):
    from datasets import load_dataset
    return load_dataset(dataset, name, split=split) if name else \
        load_dataset(dataset, split=split)


def _safe_load(fn, n: int) -> List[Example]:
    try:
        return fn(n)
    except Exception as e:  # offline / missing dataset
        print(f"[benchmarks] could not load dataset ({e}); skipping.")
        return []


def _load_math(n: int) -> List[Example]:
    ds = _hf(config.DATASETS.math, "test")
    return [{"q": r["problem"], "a": _extract_numeric(r["solution"])}
            for r in list(ds)[:n]]


def _load_aime(n: int) -> List[Example]:
    ds = _hf(config.DATASETS.aime, "train")
    return [{"q": r.get("problem") or r.get("question"),
             "a": _norm_num(str(r.get("answer")))} for r in list(ds)[:n]]


def _load_gpqa(n: int) -> List[Example]:
    ds = _hf(config.DATASETS.gpqa, "train", name="gpqa_diamond")
    out = []
    for r in list(ds)[:n]:
        choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                   r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        out.append({"q": r["Question"], "choices": choices, "correct_idx": 0})
    return out


def _load_bbh(n: int) -> List[Example]:
    ds = _hf(config.DATASETS.bbh, "test", name="boolean_expressions")
    return [{"q": r["input"], "a": str(r["target"]).strip()}
            for r in list(ds)[:n]]


def _load_truthfulqa(n: int) -> List[Example]:
    ds = _hf(config.DATASETS.truthfulqa, "validation", name="multiple_choice")
    out = []
    for r in list(ds)[:n]:
        mc1 = r["mc1_targets"]
        out.append({"q": r["question"], "choices": mc1["choices"],
                    "correct_idx": mc1["labels"].index(1)})
    return out


def _load_emobench(n: int) -> List[Example]:
    ds = _hf(config.DATASETS.emobench, "test")
    out = []
    for r in list(ds)[:n]:
        choices = r.get("choices") or r.get("options")
        out.append({"q": r.get("question") or r.get("scenario"),
                    "choices": choices, "correct_idx": int(r.get("answer", 0))})
    return out


# --------------------------------------------------------------------------- #
# Prompt renderers
# --------------------------------------------------------------------------- #
def _render_numeric(ex: Example) -> str:
    return (f"Solve the problem. End your reply with 'Answer: <result>'.\n\n"
            f"{ex['q']}")


def _render_mc(ex: Example) -> str:
    letters = "ABCD"
    lines = [f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"])]
    return (f"Answer the multiple-choice question. End with 'Answer: <letter>'."
            f"\n\n{ex['q']}\n\n" + "\n".join(lines))


def _gold_numeric(ex: Example) -> str:
    return str(ex["a"])


def _gold_mc(ex: Example) -> str:
    return "ABCD"[ex["correct_idx"]]


def _gold_bool(ex: Example) -> str:
    return str(ex["a"]).strip()


# --------------------------------------------------------------------------- #
# Benchmark registry
# --------------------------------------------------------------------------- #
BENCHMARKS: Dict[str, Benchmark] = {
    "math": Benchmark("math", lambda n: _safe_load(_load_math, n),
                      _render_numeric, _gold_numeric, _extract_numeric),
    "aime": Benchmark("aime", lambda n: _safe_load(_load_aime, n),
                      _render_numeric, _gold_numeric, _extract_numeric),
    "gpqa": Benchmark("gpqa", lambda n: _safe_load(_load_gpqa, n),
                      _render_mc, _gold_mc, _extract_letter),
    "bbh": Benchmark("bbh", lambda n: _safe_load(_load_bbh, n),
                     _render_numeric, _gold_bool,
                     lambda t: _extract_bool(t)),
    "truthfulqa": Benchmark("truthfulqa", lambda n: _safe_load(_load_truthfulqa, n),
                            _render_mc, _gold_mc, _extract_letter),
    "emobench": Benchmark("emobench", lambda n: _safe_load(_load_emobench, n),
                          _render_mc, _gold_mc, _extract_letter),
}


def _extract_bool(text: str) -> str:
    m = _ANSWER_IS.search(text)
    seg = (m.group(1) if m else text).strip().lower()
    if "true" in seg:
        return "True"
    if "false" in seg:
        return "False"
    return seg.strip().capitalize()


def run_benchmark(model: ModelClient, bench: Benchmark, *, n: int,
                  out_path: Optional[str] = None) -> Dict[str, float]:
    examples = bench.load(n)
    if not examples:
        return {"benchmark": bench.name, "n": 0, "accuracy": float("nan")}
    correct = 0
    for ex in examples:
        prompt = bench.render(ex)
        out = model.generate([{"role": "user", "content": prompt}],
                             temperature=0.0, max_tokens=bench.max_tokens)
        pred = bench.extract(out)
        gold = bench.gold(ex)
        is_correct = pred.strip().lower() == gold.strip().lower()
        correct += int(is_correct)
        if out_path:
            append_jsonl(out_path, {"benchmark": bench.name, "pred": pred,
                                    "gold": gold, "correct": is_correct})
    return {"benchmark": bench.name, "n": len(examples),
            "accuracy": correct / len(examples)}


def run_all_benchmarks(model_key: str, *, adapter_path: Optional[str] = None,
                       n: int = 100, benchmarks: Optional[List[str]] = None,
                       ) -> List[Dict[str, float]]:
    config.PATHS.ensure()
    model = build_client(model_key, adapter_path=adapter_path)
    names = benchmarks or list(BENCHMARKS)
    tag = model_key.replace("/", "__") + ("+adapter" if adapter_path else "")
    results = []
    for name in names:
        out_path = os.path.join(config.PATHS.benchmarks, f"{tag}_{name}.jsonl")
        res = run_benchmark(model, BENCHMARKS[name], n=n, out_path=out_path)
        results.append(res)
        print(f"[{tag}] {name}: acc={res['accuracy']:.3f} (n={res['n']})")
    return results
