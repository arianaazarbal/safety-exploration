"""Capability-preservation benchmarks (Figure 7).

Runs each suite on the base instruct model and on trained adapters, reporting
accuracy. The paper's claim is that DPO does not reduce capability scores; this
harness produces the side-by-side numbers to check that.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from ..config import Config, load_config
from ..models import build_model
from ..models.base import GenConfig
from ..utils.io import write_json, write_jsonl
from .suites import load_suite

log = logging.getLogger(__name__)


@dataclass
class ItemResult:
    suite: str
    prompt: str
    gold: str
    output: str
    correct: bool


def _evaluate_suite(model, suite_name, max_examples, gen_cfg, batch_size=8):
    examples, sampled = load_suite(suite_name, max_examples)
    results: list[ItemResult] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = [[{"role": "user", "content": ex.prompt}] for ex in chunk]
        gens = model.generate_batch(batch, gen_cfg)
        for ex, gen in zip(chunk, gens):
            results.append(
                ItemResult(
                    suite=suite_name, prompt=ex.prompt, gold=ex.answer,
                    output=gen.full_text, correct=ex.is_correct(gen.full_text),
                )
            )
    acc = sum(r.correct for r in results) / len(results) if results else 0.0
    return results, {"accuracy": acc, "n": len(results), "sampled": sampled}


def run_benchmarks(
    cfg: Config | None = None,
    targets: dict[str, str | None] | None = None,
) -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.output_root() / "benchmarks"
    # Capability eval should be greedy/deterministic, not temperature 1.
    gen_cfg = GenConfig(temperature=0.0, top_p=1.0, max_new_tokens=2048, thinking=False)

    if targets is None:
        targets = {"gemma-instruct": None}

    summary: dict[str, dict] = {}
    for label, adapter_path in targets.items():
        model = build_model(cfg, cfg.training.base_model, adapter_path=adapter_path)
        suite_scores = {}
        for suite_name in cfg.benchmarks.suites:
            results, stats = _evaluate_suite(
                model, suite_name, cfg.benchmarks.max_examples_per_suite, gen_cfg
            )
            write_jsonl(out_dir / label / f"{suite_name}.jsonl",
                        [asdict(r) for r in results])
            suite_scores[suite_name] = stats
            log.info("[%s] %s acc=%.3f (n=%d)", label, suite_name,
                     stats["accuracy"], stats["n"])
        summary[label] = suite_scores

    write_json(out_dir / "summary.json", summary)
    return summary
