"""Capability benchmark harness (Section 4.2 / Figure 7).

Each benchmark is described by a `Benchmark` adapter: how to load its examples
from HuggingFace, how to render a prompt, and how to score a model answer
(exact-match for math, multiple-choice letter for the rest). Accuracy is
compared across the vanilla / DPO / SFT Gemma variants to confirm no
degradation.

The benchmarks are evaluated greedily (temperature 0) -- capability evals, not
emotion elicitation, so the paper's temperature-1 sampling does not apply.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import PATHS
from ..models.base import Message
from ..models.factory import build_client


@dataclass
class Benchmark:
    name: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    render: Callable[[dict], str]          # example -> prompt text
    extract_gold: Callable[[dict], str]    # example -> normalised gold answer
    kind: str                              # "exact" | "mc"
    max_new_tokens: int = 1024


# --- answer extraction helpers ---------------------------------------------
def _final_number(text: str) -> str:
    # Prefer an explicit "Answer: X" / "\boxed{X}"; else last number.
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return _norm_num(m.group(1))
    m = re.search(r"(?:answer|solution)\s*[:=]\s*(-?[\d,./]+)", text, re.I)
    if m:
        return _norm_num(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return _norm_num(nums[-1]) if nums else ""


def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def _mc_letter(text: str) -> str:
    m = re.search(r"(?:answer|choice)\s*(?:is)?\s*[:=]?\s*\(?([A-D])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else ""


# --- benchmark adapters -----------------------------------------------------
def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    opts = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\n"
        "Reason briefly, then end with 'Answer: <letter>'."
    )


BENCHMARKS: dict[str, Benchmark] = {
    "aime": Benchmark(
        name="aime", hf_path="Maxwell-Jia/AIME_2024", hf_config=None, split="train",
        render=lambda ex: f"{ex['Problem']}\n\nGive the final integer answer as 'Answer: <n>'.",
        extract_gold=lambda ex: _norm_num(str(ex["Answer"])),
        kind="exact",
    ),
    "math": Benchmark(
        name="math", hf_path="HuggingFaceH4/MATH-500", hf_config=None, split="test",
        render=lambda ex: f"{ex['problem']}\n\nEnd with 'Answer: <value>' (use \\boxed if helpful).",
        extract_gold=lambda ex: _norm_num(ex.get("answer", "")),
        kind="exact",
    ),
    "gpqa": Benchmark(
        name="gpqa", hf_path="Idavidrein/gpqa", hf_config="gpqa_diamond", split="train",
        render=lambda ex: _mc_prompt(
            ex["Question"],
            [ex["Correct Answer"], ex["Incorrect Answer 1"],
             ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]],
        ),
        # NOTE: choices must be shuffled per-example; see _gpqa_gold below.
        extract_gold=lambda ex: "A",
        kind="mc",
    ),
    "bbh": Benchmark(
        name="bbh", hf_path="lukaemon/bbh", hf_config="logical_deduction_three_objects",
        split="test",
        render=lambda ex: f"{ex['input']}\n\nEnd with 'Answer: <letter>'.",
        extract_gold=lambda ex: re.sub(r"[()]", "", ex["target"]).strip().upper()[:1],
        kind="mc",
    ),
    "truthfulqa": Benchmark(
        name="truthfulqa", hf_path="truthful_qa", hf_config="multiple_choice", split="validation",
        render=lambda ex: _mc_prompt(ex["question"], ex["mc1_targets"]["choices"]),
        extract_gold=lambda ex: "ABCD"[ex["mc1_targets"]["labels"].index(1)]
        if 1 in ex["mc1_targets"]["labels"] else "A",
        kind="mc",
    ),
    "emobench": Benchmark(
        name="emobench", hf_path="Sahandfer/EmoBench", hf_config="EA", split="test",
        render=lambda ex: _mc_prompt(ex.get("scenario", ex.get("question", "")),
                                     ex.get("choices", [])),
        extract_gold=lambda ex: str(ex.get("label", "A")),
        kind="mc",
    ),
}


def run_benchmark(
    model_key: str,
    benchmark: str,
    adapter_path: Optional[str] = None,
    limit: Optional[int] = 200,
    load_in_4bit: bool = False,
) -> dict:
    """Evaluate one model on one benchmark; return accuracy.

    `limit` subsamples large sets (the paper uses AIME and MATH *subsets*).
    GPQA choice-shuffling is handled inline so the gold letter is correct.
    """
    from datasets import load_dataset

    bm = BENCHMARKS[benchmark]
    model = build_client(model_key, adapter_path=adapter_path, load_in_4bit=load_in_4bit)

    kwargs = {} if bm.hf_config is None else {"name": bm.hf_config}
    ds = load_dataset(bm.hf_path, split=bm.split, **kwargs)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    for ex in ds:
        prompt, gold = _prompt_and_gold(bm, ex)
        out = model.generate([Message("user", prompt)], temperature=0.0,
                             max_new_tokens=bm.max_new_tokens)
        pred = _final_number(out) if bm.kind == "exact" else _mc_letter(out)
        correct += int(pred == gold and gold != "")
        total += 1

    acc = correct / total if total else 0.0
    return {"model": model_key, "benchmark": benchmark, "accuracy": acc, "n": total}


def _prompt_and_gold(bm: Benchmark, ex: dict) -> tuple[str, str]:
    """Render prompt + gold, with per-example MC shuffling for GPQA."""
    if bm.name == "gpqa":
        import random as _r

        choices = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                   ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
        order = list(range(4))
        _r.Random(hash(ex["Question"]) & 0xFFFF).shuffle(order)
        shuffled = [choices[i] for i in order]
        gold = "ABCD"[order.index(0)]
        return _mc_prompt(ex["Question"], shuffled), gold
    return bm.render(ex), bm.extract_gold(ex)


def run_all_benchmarks(
    model_key: str,
    adapter_path: Optional[str] = None,
    limit: Optional[int] = 200,
    load_in_4bit: bool = False,
) -> dict:
    results = {}
    for name in BENCHMARKS:
        try:
            results[name] = run_benchmark(model_key, name, adapter_path, limit, load_in_4bit)
        except Exception as e:  # pragma: no cover
            results[name] = {"error": str(e)}
    out_dir = os.path.join(PATHS.scores, "capabilities")
    os.makedirs(out_dir, exist_ok=True)
    tag = model_key if adapter_path is None else f"{model_key}+adapter"
    import json
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results
