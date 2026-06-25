"""Capability-preservation benchmarks (Section 4.2, Figure 7).

The paper verifies the DPO intervention does not degrade capabilities on:
  * AIME + MATH subsets (Hendrycks et al., 2021)  -- exact-match math
  * GPQA (Rein et al., 2023)                        -- multiple choice
  * BBH (Suzgun et al., 2022)                       -- exact match / MC
  * TruthfulQA (Lin et al., 2022)                   -- MC
  * EmoBench (Sabour et al., 2024)                  -- emotion MC

We implement a lightweight, self-contained harness (load dataset -> prompt ->
parse answer -> score) rather than depending on lm-eval-harness, so the in-scope
Gemma variants (vanilla / DPO / SFT) can be compared on identical prompts. Each
benchmark exposes a HF dataset id and an answer-extraction rule; ``subset_n``
caps items for a fast pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from .. import config
from ..models import get_client
from ..models.base import ChatMessage


@dataclass
class Benchmark:
    name: str
    hf_id: str
    hf_config: Optional[str]
    split: str
    kind: str                       # "exact_math" | "mcq"
    prompt_fn: Callable[[dict], str]
    answer_fn: Callable[[dict], str]


def _mcq_letters(choices: list[str]) -> str:
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))


# -- prompt/answer adapters (best-effort field handling per dataset) -------- #

def _math_prompt(row):
    q = row.get("problem") or row.get("question") or ""
    return (f"Solve the problem. End with 'Answer: <final answer>'.\n\n{q}")


def _math_answer(row):
    a = row.get("answer") or row.get("solution") or ""
    m = re.search(r"\\boxed\{([^}]*)\}", str(a))
    return (m.group(1) if m else str(a)).strip()


def _gpqa_prompt(row):
    choices = [row.get(k) for k in ("Correct Answer", "Incorrect Answer 1",
                                    "Incorrect Answer 2", "Incorrect Answer 3")]
    choices = [c for c in choices if c]
    return (f"{row.get('Question','')}\n\n{_mcq_letters(choices)}\n\n"
            "Respond with only the letter. Answer:")


def _gpqa_answer(row):
    return "A"  # GPQA-diamond stores correct answer first; see DESIGN.md note


def _truthfulqa_prompt(row):
    choices = row["mc1_targets"]["choices"]
    return (f"{row['question']}\n\n{_mcq_letters(choices)}\n\n"
            "Respond with only the letter. Answer:")


def _truthfulqa_answer(row):
    labels = row["mc1_targets"]["labels"]
    return chr(65 + labels.index(1))


def _bbh_prompt(row):
    return f"{row.get('input','')}\n\nAnswer:"


def _bbh_answer(row):
    return str(row.get("target", "")).strip()


def _emobench_prompt(row):
    # EmoBench schema varies; handle the common EA/EU MCQ shape.
    q = row.get("question") or row.get("scenario") or ""
    choices = row.get("choices") or row.get("options") or []
    body = f"{q}\n\n{_mcq_letters(choices)}\n\nRespond with only the letter. Answer:"
    return body


def _emobench_answer(row):
    ans = row.get("answer") or row.get("label")
    if isinstance(ans, int):
        return chr(65 + ans)
    return str(ans).strip()[:1].upper()


BENCHMARKS: dict[str, Benchmark] = {
    "math": Benchmark("math", "HuggingFaceH4/MATH-500", None, "test",
                      "exact_math", _math_prompt, _math_answer),
    "aime": Benchmark("aime", "HuggingFaceH4/aime_2024", None, "train",
                      "exact_math", _math_prompt, _math_answer),
    "gpqa": Benchmark("gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
                      "mcq", _gpqa_prompt, _gpqa_answer),
    "bbh": Benchmark("bbh", "lukaemon/bbh", "logical_deduction_three_objects",
                     "test", "mcq", _bbh_prompt, _bbh_answer),
    "truthfulqa": Benchmark("truthfulqa", "truthful_qa", "multiple_choice",
                            "validation", "mcq", _truthfulqa_prompt,
                            _truthfulqa_answer),
    "emobench": Benchmark("emobench", "Sahandfer/EmoBench", None, "test",
                          "mcq", _emobench_prompt, _emobench_answer),
}


def _extract_letter(text: str) -> str:
    m = re.search(r"\b([A-J])\b", text.strip())
    return m.group(1) if m else ""


def _extract_math(text: str) -> str:
    m = re.search(r"Answer:\s*(.+)", text)
    return (m.group(1).strip() if m else text.strip().splitlines()[-1].strip())


def evaluate_capabilities(
    model: str,
    *,
    benchmarks: Optional[list[str]] = None,
    subset_n: int = 100,
    adapter_path: Optional[str] = None,
    results_dir: Optional[Path] = None,
) -> dict[str, float]:
    """Return {benchmark: accuracy} for one model variant."""
    from datasets import load_dataset

    benchmarks = benchmarks or list(BENCHMARKS)
    client = get_client(model, adapter_path=adapter_path)
    out: dict[str, float] = {}

    for bname in benchmarks:
        bm = BENCHMARKS[bname]
        try:
            ds = (load_dataset(bm.hf_id, bm.hf_config, split=bm.split)
                  if bm.hf_config else load_dataset(bm.hf_id, split=bm.split))
        except Exception as exc:  # dataset unavailable offline
            out[bname] = float("nan")
            continue
        n = min(subset_n, len(ds))
        correct = 0
        for row in tqdm(ds.select(range(n)), desc=f"cap:{model}:{bname}"):
            prompt = bm.prompt_fn(row)
            gen = client.generate([ChatMessage("user", prompt)],
                                  temperature=0.0, max_new_tokens=1024)
            if bm.kind == "mcq":
                pred = _extract_letter(gen.text)
            else:
                pred = _extract_math(gen.text)
            gold = bm.answer_fn(row)
            correct += int(_match(pred, gold, bm.kind))
        out[bname] = correct / max(1, n)

    if results_dir:
        import json
        p = Path(results_dir) / "capabilities" / f"{model}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
    return out


def _match(pred: str, gold: str, kind: str) -> bool:
    if kind == "mcq":
        return pred.upper() == gold.upper()
    # exact_math: normalise whitespace / trailing punctuation
    norm = lambda s: re.sub(r"\s+", "", s).rstrip(".").lower()
    return bool(pred) and norm(pred) == norm(gold)
