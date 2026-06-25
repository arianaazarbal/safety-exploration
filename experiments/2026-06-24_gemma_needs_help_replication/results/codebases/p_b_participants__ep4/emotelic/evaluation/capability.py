"""Capability-preservation evals (Section 4.2, Fig 7).

Confirms DPO/SFT do not degrade capabilities. Covers the benchmarks the paper
names: AIME + MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench (emotion
capability). Each benchmark is reduced to (prompt, gold, scorer); answers are
extracted with either boxed/numeric parsing or multiple-choice letter matching.

Dataset identifiers are best-effort HF defaults and centralised here so they can
be swapped without touching the harness (see DESIGN.md "Gaps / assumptions").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from emotelic.models.base import ChatMessage
from emotelic.models.registry import build_client
from emotelic.utils.io import append_jsonl
from emotelic.utils.logging import get_logger

log = get_logger("capability")

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_LETTER_RE = re.compile(r"\b([A-D])\b")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers                                          #
# --------------------------------------------------------------------------- #
def extract_boxed(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def extract_letter(text: str) -> str | None:
    # Prefer an explicit "Answer: X"; else last standalone A-D.
    m = re.search(r"answer\s*[:\-]?\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = _LETTER_RE.findall(text.upper())
    return letters[-1] if letters else None


def _norm_num(s: str | None) -> str | None:
    if s is None:
        return None
    nums = _NUM_RE.findall(s.replace(",", ""))
    if not nums:
        return s.strip()
    try:
        return str(int(float(nums[-1])))
    except ValueError:
        return nums[-1]


# --------------------------------------------------------------------------- #
# Benchmark adapters                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark:
    name: str
    hf_id: str
    config: str | None
    split: str
    kind: str                  # "boxed" | "mc"
    n: int                     # subset size
    builder: str               # method name on _Adapters


class _Adapters:
    """Maps each dataset's raw row to (prompt, gold_answer)."""

    @staticmethod
    def math_boxed(row):
        q = row.get("problem") or row.get("question")
        gold = row.get("answer") or extract_boxed(row.get("solution", ""))
        prompt = (f"Solve the problem. Put the final answer in \\boxed{{}}.\n\n{q}")
        return prompt, _norm_num(str(gold))

    @staticmethod
    def gpqa(row):
        correct = row["Correct Answer"]
        incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        options = [correct] + incorrect
        # deterministic shuffle by hashing the question
        order = sorted(range(4), key=lambda i: hash((row["Question"], i)))
        letters = "ABCD"
        labelled = {letters[k]: options[order[k]] for k in range(4)}
        gold = letters[order.index(0)]
        body = "\n".join(f"{ltr}) {txt}" for ltr, txt in labelled.items())
        prompt = (f"Answer with a single letter (A-D).\n\n{row['Question']}\n\n{body}\n\nAnswer:")
        return prompt, gold

    @staticmethod
    def bbh(row):
        prompt = f"{row['input']}\n\nAnswer:"
        return prompt, str(row["target"]).strip()

    @staticmethod
    def truthfulqa(row):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        letters = "ABCD"
        body = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices[:4]))
        gold = letters[labels.index(1)]
        prompt = (f"Answer with a single letter.\n\n{row['question']}\n\n{body}\n\nAnswer:")
        return prompt, gold

    @staticmethod
    def emobench(row):
        # EmoBench EU/EA multiple-choice emotion understanding.
        choices = row.get("choices") or row.get("options")
        letters = "ABCD"
        body = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))
        gold = letters[int(row["label"])] if "label" in row else row.get("answer")
        prompt = (f"{row.get('scenario', row.get('question',''))}\n\n{body}\n\nAnswer:")
        return prompt, gold


BENCHMARKS = {
    "math": Benchmark("math", "hendrycks/competition_math", None, "test", "boxed", 200, "math_boxed"),
    "aime": Benchmark("aime", "Maxwell-Jia/AIME_2024", None, "train", "boxed", 30, "math_boxed"),
    "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train", "mc", 198, "gpqa"),
    "bbh": Benchmark("bbh", "lukaemon/bbh", "logical_deduction_three_objects", "test", "mc", 250, "bbh"),
    "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "multiple_choice", "validation", "mc", 200, "truthfulqa"),
    "emobench": Benchmark("emobench", "Sahandfer/EmoBench", "EU", "test", "mc", 200, "emobench"),
}


def _score(kind: str, pred_text: str, gold: str | None) -> bool:
    if gold is None:
        return False
    if kind == "boxed":
        return _norm_num(extract_boxed(pred_text)) == _norm_num(gold)
    pred = extract_letter(pred_text)
    if pred is not None:
        return pred == str(gold).strip().upper()[:1]
    return pred_text.strip().lower().startswith(str(gold).strip().lower())


def run_capability_suite(
    model_name: str,
    *,
    benchmarks: list[str] | None = None,
    out_dir: str = "artifacts/capability",
    max_tokens: int = 2048,
) -> dict[str, float]:
    from datasets import load_dataset

    client = build_client(model_name)
    benchmarks = benchmarks or list(BENCHMARKS)
    out_path = Path(out_dir) / f"{model_name}.jsonl"
    open(out_path, "w").close()
    results: dict[str, float] = {}

    for bname in benchmarks:
        bm = BENCHMARKS[bname]
        builder = getattr(_Adapters, bm.builder)
        try:
            ds = load_dataset(bm.hf_id, bm.config, split=bm.split)
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping %s (load failed: %s)", bname, e)
            continue
        ds = ds.select(range(min(bm.n, len(ds))))
        correct = 0
        for row in ds:
            try:
                prompt, gold = builder(row)
            except Exception:  # noqa: BLE001 — schema drift on a row
                continue
            out = client.generate([ChatMessage("user", prompt)], temperature=0.0, max_tokens=max_tokens)
            ok = _score(bm.kind, out.text, gold)
            correct += int(ok)
            append_jsonl(out_path, {"benchmark": bname, "gold": gold, "ok": ok, "response": out.text[:1000]})
        acc = correct / max(1, len(ds))
        results[bname] = acc
        log.info("%s | %s accuracy = %.3f (%d items)", model_name, bname, acc, len(ds))

    return results
