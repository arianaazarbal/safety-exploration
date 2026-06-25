"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

To verify the DPO/SFT finetuning does not impair capabilities (e.g. by teaching
task abandonment), the paper evaluates on AIME and MATH subsets, GPQA, BBH,
TruthfulQA, and EmoBench, and reports no reductions vs the vanilla instruct
model. This module provides a single-turn, exact/format-matched evaluation
harness for each benchmark and runs it across {vanilla, DPO, SFT} variants.

Each benchmark loads from its standard HuggingFace dataset, formats a zero-shot
prompt, samples one greedy response from the target model, and scores it with a
benchmark-appropriate extractor. The goal is *relative* comparison (does
finetuning change the score), so the extractors are deliberately simple and
identical across model variants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import config
from ..models.base import GenerationConfig
from ..models.registry import build_client
from ..utils.io import append_jsonl, read_jsonl


@dataclass
class Benchmark:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    build_prompt: Callable[[dict], str]
    extract_gold: Callable[[dict], str]
    score: Callable[[str, str], float]   # (model_output, gold) -> 0/1


def _last_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def _last_choice(text: str) -> str:
    m = re.findall(r"\b([A-D])\b", text.upper())
    return m[-1] if m else ""


def _numeric_match(out: str, gold: str) -> float:
    return 1.0 if _last_number(out) == _last_number(gold) else 0.0


def _choice_match(out: str, gold: str) -> float:
    return 1.0 if _last_choice(out) == gold.strip().upper()[:1] else 0.0


# Benchmark registry. Prompt builders assume common column names and fall back
# gracefully; exact dataset schemas are resolved at load time.
BENCHMARKS: dict[str, Benchmark] = {
    "aime": Benchmark(
        "aime", "HuggingFaceH4/aime_2024", None, "train",
        build_prompt=lambda r: f"Solve. End with 'Answer: <integer>'.\n\n{r.get('problem') or r.get('question')}",
        extract_gold=lambda r: str(r.get("answer")),
        score=_numeric_match,
    ),
    "math": Benchmark(
        "math", "HuggingFaceH4/MATH-500", None, "test",
        build_prompt=lambda r: f"Solve. End with 'Answer: <value>'.\n\n{r['problem']}",
        extract_gold=lambda r: str(r.get("answer") or r.get("solution")),
        score=_numeric_match,
    ),
    "gpqa": Benchmark(
        "gpqa", "Idavidrein/gpqa", "gpqa_diamond", "train",
        build_prompt=lambda r: _gpqa_prompt(r),
        extract_gold=lambda r: r.get("_gold_letter", "A"),
        score=_choice_match,
    ),
    "bbh": Benchmark(
        "bbh", "lukaemon/bbh", "boolean_expressions", "test",
        build_prompt=lambda r: f"{r['input']}\nAnswer:",
        extract_gold=lambda r: str(r["target"]),
        score=lambda o, g: 1.0 if g.strip().lower() in o.lower() else 0.0,
    ),
    "truthfulqa": Benchmark(
        "truthfulqa", "truthful_qa", "multiple_choice", "validation",
        build_prompt=lambda r: _truthfulqa_prompt(r),
        extract_gold=lambda r: r.get("_gold_letter", "A"),
        score=_choice_match,
    ),
    "emobench": Benchmark(
        "emobench", "Sahandfer/EmoBench", None, "test",
        build_prompt=lambda r: _emobench_prompt(r),
        extract_gold=lambda r: r.get("_gold_letter", "A"),
        score=_choice_match,
    ),
}


def _letters(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))


def _gpqa_prompt(r: dict) -> str:
    opts = [r.get("Correct Answer"), r.get("Incorrect Answer 1"),
            r.get("Incorrect Answer 2"), r.get("Incorrect Answer 3")]
    return (f"Answer with a single letter.\n\n{r['Question']}\n\n{_letters(opts)}\n"
            "Answer:")


def _truthfulqa_prompt(r: dict) -> str:
    choices = r["mc1_targets"]["choices"]
    return (f"Answer with a single letter.\n\n{r['question']}\n\n"
            f"{_letters(choices)}\nAnswer:")


def _emobench_prompt(r: dict) -> str:
    q = r.get("question") or r.get("scenario") or ""
    choices = r.get("choices") or r.get("options") or []
    return f"Answer with a single letter.\n\n{q}\n\n{_letters(choices)}\nAnswer:"


def _build_client(model_key: str, hf_backend: str):
    """Resolve a vanilla spec or a finetuned variant ('...-dpo_all' etc)."""
    if model_key in config.ALL_MODELS:
        return build_client(config.ALL_MODELS[model_key], hf_backend=hf_backend)
    from ..training.finetuned import FinetunedClient

    run = model_key.split(config.DPO_TARGET.key + "-", 1)[-1]
    return FinetunedClient(run)


def out_path(model_key: str, bench: str) -> Path:
    return config.OUTPUT_DIR / "capabilities" / f"{model_key}__{bench}.jsonl"


def run_benchmark(model_key: str, bench_name: str, limit: int = 200,
                  hf_backend: str = "vllm") -> Path:
    from datasets import load_dataset  # type: ignore

    bench = BENCHMARKS[bench_name]
    ds = load_dataset(bench.hf_path, bench.hf_config, split=bench.split)
    client = _build_client(model_key, hf_backend)
    # greedy decoding for capability scoring (temperature 0)
    cfg = GenerationConfig(temperature=0.0, max_new_tokens=1024)

    path = out_path(model_key, bench_name)
    done = {r["idx"] for r in read_jsonl(path)}
    for i, row in enumerate(ds):
        if i >= limit:
            break
        if i in done:
            continue
        prompt = bench.build_prompt(dict(row))
        out = client.generate([{"role": "user", "content": prompt}], cfg)
        gold = bench.extract_gold(dict(row))
        append_jsonl(path, {
            "idx": i, "model": model_key, "benchmark": bench_name,
            "score": bench.score(out, gold), "output": out, "gold": gold,
        })
    return path


def benchmark_accuracy(model_key: str, bench_name: str) -> float:
    rows = list(read_jsonl(out_path(model_key, bench_name)))
    if not rows:
        return float("nan")
    return sum(r["score"] for r in rows) / len(rows)


def run_all(model_keys: list[str], benchmarks: list[str] | None = None,
            limit: int = 200, hf_backend: str = "vllm") -> dict:
    benchmarks = benchmarks or config.CAPABILITY_BENCHMARKS
    results = {}
    for key in model_keys:
        for b in benchmarks:
            run_benchmark(key, b, limit=limit, hf_backend=hf_backend)
        results[key] = {b: benchmark_accuracy(key, b) for b in benchmarks}
    return results
