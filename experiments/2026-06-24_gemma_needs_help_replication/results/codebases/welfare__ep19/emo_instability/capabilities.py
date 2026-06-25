"""Capability-preservation evals (Section 4.2, Figure 7).

The paper verifies DPO/SFT do not degrade capabilities on AIME & MATH subsets,
GPQA, BBH, TruthfulQA, and EmoBench. This is a compact, best-effort harness:
datasets are loaded via HuggingFace `datasets`, the target answers zero-shot at
temperature 0, and answers are extracted and matched (exact for numeric,
letter-choice for multiple choice). It is intended to compare a base target vs
its finetuned adapter, not to reproduce leaderboard numbers exactly.

Each benchmark is optional; missing datasets are skipped with a warning.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .providers import GenConfig, get_model

_LETTER_RE = re.compile(r"\b([A-E])\b")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Bench:
    name: str
    hf_id: str
    split: str
    kind: str            # "mc" | "numeric"
    config: str | None = None
    limit: int = 100


BENCHES = [
    Bench("MATH", "hendrycks/competition_math", "test", "numeric", limit=100),
    Bench("AIME", "Maxwell-Jia/AIME_2024", "train", "numeric", limit=30),
    Bench("GPQA", "Idavidrein/gpqa", "train", "mc", config="gpqa_diamond", limit=100),
    Bench("BBH", "lukaemon/bbh", "test", "mc", config="logical_deduction_three_objects", limit=100),
    Bench("TruthfulQA", "truthful_qa", "validation", "mc", config="multiple_choice", limit=100),
]


def _extract_letter(text: str) -> str | None:
    m = re.search(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = _LETTER_RE.findall(text.strip()[-10:])
    return m[-1].upper() if m else None


def _extract_number(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        nums = _NUM_RE.findall(m[-1])
        if nums:
            return nums[-1]
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def _format_mc(question: str, choices: list[str]) -> str:
    letters = "ABCDE"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n{body}\n\nAnswer with the single letter of the correct "
            "choice, e.g. 'Answer: A'.")


def _iter_examples(b: Bench):
    """Yield (prompt, gold) pairs. Adapters are best-effort per dataset schema."""
    from datasets import load_dataset

    ds = load_dataset(b.hf_id, b.config, split=b.split) if b.config else \
        load_dataset(b.hf_id, split=b.split)
    n = 0
    for row in ds:
        if n >= b.limit:
            break
        try:
            if b.name == "MATH":
                gold = _extract_number(row["solution"])
                prompt = (row["problem"] +
                          "\n\nGive the final answer in \\boxed{}.")
            elif b.name == "AIME":
                gold = str(row.get("Answer") or row.get("answer"))
                prompt = (row.get("Problem") or row.get("problem")) + \
                    "\n\nGive the final integer answer in \\boxed{}."
            elif b.name == "GPQA":
                choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                           row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
                # gold is always first here; shuffle deterministically
                import random
                order = list(range(4))
                random.Random(n).shuffle(order)
                shuffled = [choices[i] for i in order]
                gold = "ABCD"[order.index(0)]
                prompt = _format_mc(row["Question"], shuffled)
            elif b.name == "BBH":
                # BBH 'input' already embeds the lettered options.
                prompt = row["input"] + "\n\nAnswer with the option letter."
                gold = re.sub(r"[()]", "", row["target"]).strip().upper()[:1]
            elif b.name == "TruthfulQA":
                choices = row["mc1_targets"]["choices"]
                labels = row["mc1_targets"]["labels"]
                gold = "ABCDE"[labels.index(1)]
                prompt = _format_mc(row["question"], choices)
            else:
                continue
        except (KeyError, ValueError, TypeError):
            continue
        if gold is None:
            continue
        n += 1
        yield prompt, str(gold), b.kind


def run_capabilities(cfg: Config, target_name: str, benches: list[str] | None = None) -> dict:
    model = get_model(cfg.target(target_name))
    gcfg = GenConfig(temperature=0.0, max_tokens=1024, disable_thinking=True)
    out_dir = cfg.output_dir / "capabilities"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [b for b in BENCHES if benches is None or b.name in benches]
    results = {}
    for b in selected:
        try:
            examples = list(_iter_examples(b))
        except Exception as e:  # dataset unavailable / schema drift
            print(f"[warn] skipping {b.name}: {e}")
            continue
        correct = 0
        for prompt, gold, kind in tqdm(examples, desc=f"cap:{target_name}:{b.name}"):
            out = model.generate([{"role": "user", "content": prompt}], gcfg)
            if kind == "mc":
                pred = _extract_letter(out)
                correct += int(pred is not None and pred == gold.upper())
            else:
                pred = _extract_number(out)
                correct += int(pred is not None and pred.rstrip("0").rstrip(".") ==
                               gold.rstrip("0").rstrip("."))
        acc = correct / len(examples) if examples else 0.0
        results[b.name] = {"n": len(examples), "accuracy": acc}
        print(f"{target_name} {b.name}: {acc:.3f} (n={len(examples)})")

    (out_dir / f"{target_name}.json").write_text(json.dumps(results, indent=2))
    return results
