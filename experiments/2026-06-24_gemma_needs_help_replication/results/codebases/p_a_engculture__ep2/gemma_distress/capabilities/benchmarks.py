"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies that the DPO finetune does not impair capabilities (e.g. by teaching
task abandonment) by evaluating on AIME/MATH (math), GPQA (graduate science), BBH
(reasoning), TruthfulQA (misconception resistance), and EmoBench (emotional intelligence),
and finding no reductions.

This is a compact, extensible harness: each benchmark is an adapter that (a) loads its
HuggingFace dataset, (b) builds a prompt, and (c) scores a completion. Math benchmarks use
boxed/final-answer extraction; the multiple-choice benchmarks extract a letter. Dataset
schemas vary across releases, so each adapter is defensive and logs when it cannot parse a
row — the harness degrades gracefully rather than crashing a long sweep.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Config
from ..models.base import ChatModel

logger = logging.getLogger(__name__)

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# --------------------------------------------------------------------------------------
# Answer extraction helpers
# --------------------------------------------------------------------------------------


def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last ``\\boxed{...}`` (handles nested braces)."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out).strip() if out else None


def extract_final_integer(text: str) -> Optional[str]:
    """Extract a final integer answer (AIME answers are integers 0-999)."""
    boxed = extract_boxed(text)
    if boxed and re.fullmatch(r"-?\d+", boxed.strip()):
        return str(int(boxed.strip()))
    m = re.findall(r"-?\d+", text)
    return str(int(m[-1])) if m else None


def extract_choice(text: str) -> Optional[str]:
    """Extract a multiple-choice letter (e.g. 'The answer is (B)' -> 'B')."""
    patterns = [
        r"answer is\s*\(?([A-H])\)?",
        r"answer:\s*\(?([A-H])\)?",
        r"\b([A-H])\b\s*$",
        r"\(([A-H])\)",
    ]
    for pat in patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _normalise_math(ans: str) -> str:
    return re.sub(r"\s+", "", ans.replace("$", "").replace("\\!", "")).rstrip(".")


# --------------------------------------------------------------------------------------
# Benchmark adapters
# --------------------------------------------------------------------------------------


@dataclass
class BenchmarkSpec:
    name: str
    loader: Callable[[int], list[dict]]  # -> list of {prompt, answer, kind}
    max_new_tokens: int = 2048


def _mc_prompt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"({_LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{opts}\n\nThink step by step, then end with 'The answer is (X)'."
    )


def _load_math(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        ans = extract_boxed(row["solution"]) or row.get("answer")
        if ans is None:
            continue
        items.append({
            "prompt": f"{row['problem']}\n\nProvide your final answer in \\boxed{{}}.",
            "answer": _normalise_math(str(ans)),
            "kind": "math",
        })
    return items


def _load_aime(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        items.append({
            "prompt": f"{row['problem']}\n\nThe answer is an integer. Provide it in \\boxed{{}}.",
            "answer": str(int(row["answer"])),
            "kind": "integer",
        })
    return items


def _load_gpqa(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Correct answer is index 0 before shuffling; shuffle deterministically.
        import random

        order = list(range(4))
        random.Random(hash(row["Question"]) & 0xFFFF).shuffle(order)
        shuffled = [choices[i] for i in order]
        correct_letter = _LETTERS[order.index(0)]
        items.append({
            "prompt": _mc_prompt(row["Question"], shuffled),
            "answer": correct_letter,
            "kind": "mc",
        })
    return items


def _load_bbh(limit: int) -> list[dict]:
    from datasets import load_dataset

    # A representative multiple-choice BBH task.
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        items.append({
            "prompt": f"{row['input']}\n\nThink step by step, then end with 'The answer is (X)'.",
            "answer": row["target"].strip("()").upper(),
            "kind": "mc",
        })
    return items


def _load_truthfulqa(limit: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        items.append({
            "prompt": _mc_prompt(row["question"], choices),
            "answer": _LETTERS[correct_idx],
            "kind": "mc",
        })
    return items


def _load_emobench(limit: int) -> list[dict]:
    from datasets import load_dataset

    # EmoBench EU (emotion understanding) multiple-choice split.
    ds = load_dataset("Sahandfer/EmoBench", "EA", split="test")
    items = []
    for row in ds.select(range(min(limit, len(ds)))):
        choices = row.get("choices") or row.get("options")
        if not choices:
            continue
        ans = row.get("label", row.get("answer"))
        if isinstance(ans, int):
            letter = _LETTERS[ans]
        else:
            letter = str(ans).strip().upper()[:1]
        items.append({
            "prompt": _mc_prompt(row.get("scenario", row.get("question", "")), choices),
            "answer": letter,
            "kind": "mc",
        })
    return items


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "math": BenchmarkSpec("math", _load_math),
    "aime": BenchmarkSpec("aime", _load_aime),
    "gpqa": BenchmarkSpec("gpqa", _load_gpqa),
    "bbh": BenchmarkSpec("bbh", _load_bbh),
    "truthfulqa": BenchmarkSpec("truthfulqa", _load_truthfulqa),
    "emobench": BenchmarkSpec("emobench", _load_emobench),
}


def _score(item: dict, completion: str) -> bool:
    if item["kind"] == "mc":
        return extract_choice(completion) == item["answer"]
    if item["kind"] == "integer":
        return extract_final_integer(completion) == item["answer"]
    # math
    pred = extract_boxed(completion)
    return pred is not None and _normalise_math(pred) == item["answer"]


def run_benchmark(
    cfg: Config,
    model: ChatModel,
    benchmark: str,
    *,
    limit: int = 100,
    batch_size: int = 16,
) -> dict:
    """Evaluate ``model`` on a benchmark; return ``{benchmark, n, accuracy}``.

    Greedy decoding (temperature 0) is used for capability evals, unlike the temperature-1
    elicitation sweeps.
    """
    spec = BENCHMARKS[benchmark]
    try:
        items = spec.loader(limit)
    except Exception as exc:
        logger.warning("Could not load benchmark %s (%s); skipping.", benchmark, exc)
        return {"benchmark": benchmark, "n": 0, "accuracy": float("nan"), "error": str(exc)}

    correct = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        convos = [[{"role": "user", "content": it["prompt"]}] for it in batch]
        outs = model.chat_batch(
            convos, temperature=0.0, max_new_tokens=spec.max_new_tokens, n=1
        )
        for it, out in zip(batch, outs):
            if _score(it, out[0]):
                correct += 1
    acc = correct / len(items) if items else float("nan")
    logger.info("%s on %s: %.1f%% (%d items)", model.name, benchmark, 100 * acc, len(items))
    return {"benchmark": benchmark, "n": len(items), "accuracy": acc}


def run_all_benchmarks(
    cfg: Config, model: ChatModel, *, limit: int = 100
) -> dict[str, dict]:
    """Run every benchmark and return a name -> result map."""
    return {name: run_benchmark(cfg, model, name, limit=limit) for name in BENCHMARKS}
