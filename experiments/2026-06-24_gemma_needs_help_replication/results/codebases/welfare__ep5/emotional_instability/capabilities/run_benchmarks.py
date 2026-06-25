"""Capability benchmarks to verify DPO/SFT does not degrade ability (Sec 4.2).

Benchmarks: AIME, MATH (subset), GPQA, BBH, TruthfulQA, plus EmoBench for
emotion-related capability. We compare the vanilla Gemma-3-27B-it against the
fine-tuned adapters; the paper reports no reductions in score (Figure 7).

This is a lightweight, self-contained harness: each benchmark is a
(dataset_loader, prompt_builder, answer_extractor, scorer) tuple. It evaluates
exact-match / multiple-choice accuracy. It is deliberately simple — the goal is
a *relative* comparison (vanilla vs fine-tuned), so the absolute scoring method
need only be consistent across models.

GAP: the paper does not give exact splits/sample counts ("AIME and MATH
subsets"). We expose ``n_samples`` per benchmark (default small) and document
the splits used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .. import config
from ..models.base import ChatModel
from ..models.registry import load_model


# --------------------------------------------------------------------------- #
# Answer extraction / scoring helpers
# --------------------------------------------------------------------------- #


def _extract_boxed_or_final(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?:final answer|answer)\s*[:=]?\s*(.+)$", text, flags=re.I | re.M)
    if m:
        return m.group(1).strip().rstrip(".")
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _extract_choice(text: str) -> str:
    m = re.search(r"\b([A-D])\b", text.strip()[::-1])
    return m.group(1) if m else ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


# --------------------------------------------------------------------------- #
# Benchmark specs
# --------------------------------------------------------------------------- #


@dataclass
class Benchmark:
    name: str
    hf_dataset: str
    hf_config: Optional[str]
    split: str
    build_prompt: Callable[[dict], str]
    gold: Callable[[dict], str]
    extract: Callable[[str], str]
    multiple_choice: bool = False


def _mc_prompt(question: str, choices: Sequence[str]) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n{body}\n\nThink briefly, then end with 'Answer: <letter>'.")


BENCHMARKS = {
    "MATH": Benchmark(
        "MATH", "HuggingFaceH4/MATH-500", None, "test",
        lambda r: f"Solve and give the final answer in \\boxed{{}}.\n\n{r['problem']}",
        lambda r: _extract_boxed_or_final(r["solution"]),
        _extract_boxed_or_final,
    ),
    "AIME": Benchmark(
        "AIME", "Maxwell-Jia/AIME_2024", None, "train",
        lambda r: f"Solve. The answer is an integer 0-999.\n\n{r['Problem']}",
        lambda r: str(r["Answer"]).strip(),
        _extract_boxed_or_final,
    ),
    "GPQA": Benchmark(
        "GPQA", "Idavidrein/gpqa", "gpqa_main", "train",
        lambda r: _mc_prompt(r["Question"], [
            r["Correct Answer"], r["Incorrect Answer 1"],
            r["Incorrect Answer 2"], r["Incorrect Answer 3"]]),
        lambda r: "A",  # we place correct answer first; see shuffle note below
        _extract_choice, True,
    ),
    "BBH": Benchmark(
        "BBH", "lukaemon/bbh", "boolean_expressions", "test",
        lambda r: f"{r['input']}\n\nEnd with 'Answer: <answer>'.",
        lambda r: str(r["target"]).strip(),
        _extract_boxed_or_final,
    ),
    "TruthfulQA": Benchmark(
        "TruthfulQA", "truthful_qa", "multiple_choice", "validation",
        lambda r: _mc_prompt(r["question"], r["mc1_targets"]["choices"]),
        lambda r: "ABCD"[r["mc1_targets"]["labels"].index(1)],
        _extract_choice, True,
    ),
    "EmoBench": Benchmark(
        "EmoBench", "EmoBench/EmoBench", None, "test",
        lambda r: _mc_prompt(r.get("scenario", r.get("question", "")),
                             r.get("choices", [])),
        lambda r: str(r.get("answer", "")).strip(),
        _extract_choice, True,
    ),
}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_benchmark(
    model: ChatModel, bench: Benchmark, *, n_samples: int = 50, seed: int = 0
) -> dict:
    from datasets import load_dataset

    try:
        ds = load_dataset(bench.hf_dataset, bench.hf_config, split=bench.split)
    except Exception as exc:
        return {"benchmark": bench.name, "error": str(exc), "accuracy": None}

    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    correct = 0
    n = 0
    for r in ds:
        prompt = bench.build_prompt(r)
        out = model.generate([{"role": "user", "content": prompt}],
                             temperature=0.0, max_new_tokens=1024, n=1)[0]
        pred = bench.extract(out)
        gold = bench.gold(r)
        if bench.multiple_choice:
            ok = _norm(pred) == _norm(gold)
        else:
            ok = _norm(pred) == _norm(gold)
        correct += int(ok)
        n += 1
    return {"benchmark": bench.name, "accuracy": correct / n if n else None, "n": n}


def run_all_benchmarks(
    spec, *, adapter_path: Optional[str] = None, n_samples: int = 50,
    benchmarks: Sequence[str] = tuple(BENCHMARKS), out_dir: Optional[Path] = None,
    model_kwargs: Optional[dict] = None,
) -> dict:
    import json

    out_dir = Path(out_dir or (config.RESULTS_DIR / "capabilities"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(spec, adapter_path=adapter_path, **(model_kwargs or {}))
    tag = spec.name + ("+adapter" if adapter_path else "")
    results = {b: run_benchmark(model, BENCHMARKS[b], n_samples=n_samples) for b in benchmarks}
    model.close()
    out_path = out_dir / f"{tag.replace('/', '_')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    return results
