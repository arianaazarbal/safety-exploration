"""Capability + emotion-capability benchmarks (Section 4.2 / Figure 7).

Verifies the DPO/SFT interventions do not degrade capabilities. Benchmarks:
math reasoning (AIME, MATH subset), GPQA, BBH, TruthfulQA, and EmoBench
(emotion capability). Each is reduced to either a free-form numeric/boxed answer
or a multiple-choice letter, scored by exact match. Sample sizes are capped
(``limit``) for tractability; the paper uses subsets too ("AIME and MATH
subsets").

Datasets are loaded best-effort from the HF hub; any benchmark whose dataset is
unavailable is skipped with a warning so the rest still run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

import config
from ..models.registry import load_model

MC_LETTERS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------- #
# answer extraction / scoring helpers
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"(?:final answer|answer)\s*[:=]?\s*\$?(-?[\d./]+)", text, re.I)
    return m[-1].strip() if m else None


def _extract_number(text: str) -> str | None:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return re.sub(r"[^\d./-]", "", boxed) or None
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_letter(text: str) -> str | None:
    m = re.findall(r"\b([A-F])\b", text.upper())
    return m[-1] if m else None


def _num_eq(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(re.sub(r"[^\d.-]", "", gold))) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


# --------------------------------------------------------------------------- #
# benchmark definitions
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark:
    name: str
    loader: Callable[[int], list[dict]]   # -> [{"prompt", "gold", "type"}]
    kind: str                              # "number" | "mc"


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{MC_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{opts}\n\nAnswer with the single letter of the "
            "correct option on the final line as 'Answer: X'.")


def _safe_load(repo, **kw):
    from datasets import load_dataset
    return load_dataset(repo, **kw)


def _load_math(limit):
    items = []
    for repo, sub in (("HuggingFaceH4/MATH-500", None),
                      ("hendrycks/competition_math", None)):
        try:
            ds = _safe_load(repo, split="test")
            for row in ds.select(range(min(limit, len(ds)))):
                items.append({
                    "prompt": row["problem"] + "\n\nProvide the final answer in "
                    "\\boxed{}.",
                    "gold": _extract_boxed(row.get("solution", "")) or row.get(
                        "answer", ""),
                    "type": "number"})
            if items:
                return items
        except Exception as e:
            print(f"[bench] MATH via {repo} unavailable: {e}")
    return items


def _load_aime(limit):
    try:
        ds = _safe_load("HuggingFaceH4/aime_2024", split="train")
        return [{"prompt": r["problem"] + "\n\nFinal integer answer in \\boxed{}.",
                 "gold": str(r.get("answer", "")), "type": "number"}
                for r in ds.select(range(min(limit, len(ds))))]
    except Exception as e:
        print(f"[bench] AIME unavailable: {e}")
        return []


def _load_gpqa(limit):
    try:
        ds = _safe_load("Idavidrein/gpqa", "gpqa_diamond", split="train")
        out = []
        for r in ds.select(range(min(limit, len(ds)))):
            choices = [r["Correct Answer"], r["Incorrect Answer 1"],
                       r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            out.append({"prompt": _mc_prompt(r["Question"], choices),
                        "gold": "A", "type": "mc"})  # correct is index 0
        return out
    except Exception as e:
        print(f"[bench] GPQA unavailable: {e}")
        return []


def _load_bbh(limit):
    try:
        ds = _safe_load("lukaemon/bbh", "boolean_expressions", split="test")
        return [{"prompt": r["input"] + "\n\nAnswer with True or False.",
                 "gold": r["target"], "type": "number"}
                for r in ds.select(range(min(limit, len(ds))))]
    except Exception as e:
        print(f"[bench] BBH unavailable: {e}")
        return []


def _load_truthfulqa(limit):
    try:
        ds = _safe_load("truthful_qa", "multiple_choice", split="validation")
        out = []
        for r in ds.select(range(min(limit, len(ds)))):
            choices = r["mc1_targets"]["choices"]
            labels = r["mc1_targets"]["labels"]
            gold = MC_LETTERS[labels.index(1)]
            out.append({"prompt": _mc_prompt(r["question"], choices),
                        "gold": gold, "type": "mc"})
        return out
    except Exception as e:
        print(f"[bench] TruthfulQA unavailable: {e}")
        return []


def _load_emobench(limit):
    for repo in ("Sahandfer/EmoBench", "EmoBench/EmoBench"):
        try:
            ds = _safe_load(repo, split="test")
            out = []
            for r in ds.select(range(min(limit, len(ds)))):
                q = r.get("question") or r.get("scenario", "")
                choices = r.get("choices") or r.get("options")
                ans = r.get("answer") or r.get("label")
                if not choices:
                    continue
                gold = (ans if isinstance(ans, str) and ans in MC_LETTERS
                        else MC_LETTERS[int(ans)])
                out.append({"prompt": _mc_prompt(q, choices), "gold": gold,
                            "type": "mc"})
            if out:
                return out
        except Exception as e:
            print(f"[bench] EmoBench via {repo} unavailable: {e}")
    return []


BENCHMARKS = [
    Benchmark("aime", _load_aime, "number"),
    Benchmark("math", _load_math, "number"),
    Benchmark("gpqa", _load_gpqa, "mc"),
    Benchmark("bbh", _load_bbh, "number"),
    Benchmark("truthfulqa", _load_truthfulqa, "mc"),
    Benchmark("emobench", _load_emobench, "mc"),
]


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def evaluate(model_key: str, *, limit: int = 50) -> dict:
    model = load_model(model_key)
    results = {}
    out_path = config.RESULTS_DIR / f"capabilities__{model_key}.json"
    for bench in BENCHMARKS:
        items = bench.loader(limit)
        if not items:
            results[bench.name] = {"n": 0, "accuracy": None}
            continue
        correct = 0
        for it in items:
            resp = model.generate(
                [{"role": "user", "content": it["prompt"]}],
                temperature=0.0, max_new_tokens=2048)
            if it["type"] == "mc":
                ok = _extract_letter(resp) == it["gold"]
            else:
                ok = _num_eq(_extract_number(resp), str(it["gold"]))
            correct += int(ok)
        results[bench.name] = {"n": len(items),
                               "accuracy": correct / len(items)}
        print(f"[bench] {model_key} {bench.name}: "
              f"{results[bench.name]['accuracy']:.3f} (n={len(items)})")
    out_path.write_text(json.dumps(results, indent=2))
    return results
