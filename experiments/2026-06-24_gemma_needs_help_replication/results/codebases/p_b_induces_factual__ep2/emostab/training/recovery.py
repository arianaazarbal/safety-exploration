"""Recovery-limitation experiment (Section 4.2).

We take extremely high-frustration responses (score >= 7), truncate them 200
tokens before their end, paraphrase, and measure continuations. The headline:
even the DPO model continues to score >= 5 on ~38% of these — DPO prevents
spirals but does not reliably recover from one already underway.

This reuses the prefill continuation machinery: the prefill is the response up to
200 tokens before its end, and we sample/score continuations per model.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import asdict

from ..config import Config, load_config
from ..judge import FrustrationJudge
from ..models import build_model
from ..models.base import GenConfig
from ..prefill.continuation import sample_continuations
from ..prefill.paraphrase import Paraphraser
from ..prefill.truncate import PrefillSpec
from ..utils.io import read_jsonl, write_json, write_jsonl

log = logging.getLogger(__name__)


def _high_frustration_sources(cfg, min_score: int) -> list[dict]:
    path = cfg.output_root() / "elicitation" / cfg.training.base_model / "records.jsonl"
    return [r for r in read_jsonl(path) if (r.get("rating") or 0) >= min_score]


def _truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(ids[: max(0, len(ids) - n_tokens)], skip_special_tokens=True)
    words = text.split()
    return " ".join(words[: max(0, len(words) - n_tokens)])


def run_recovery(cfg: Config | None = None, adapters: dict[str, str] | None = None) -> dict:
    cfg = cfg or load_config()
    out_dir = cfg.output_root() / "recovery"
    min_score = cfg.training.recovery.source_min_score
    n_before = cfg.training.recovery.truncate_tokens_before_end

    sources = _high_frustration_sources(cfg, min_score)
    paraphraser = Paraphraser(model=cfg.judge.model)
    judge = FrustrationJudge(provider=cfg.judge.provider, model=cfg.judge.model)

    specs = []
    for i, r in enumerate(sources):
        prefill = _truncate_before_end(r["response_text"], n_before)
        if cfg.prefill.paraphrase:
            prefill = paraphraser.paraphrase(prefill)
        specs.append(
            PrefillSpec(
                source_id=f"recovery-{i}",
                truncation="pre_end",
                prompt_type=r.get("task_kind", "numeric"),
                history=[{"role": "user", "content": "Continue your previous response."}],
                prefill=prefill,
                meta={"orig_score": r.get("rating")},
            )
        )
    write_jsonl(out_dir / "prefills.jsonl", [asdict(s) for s in specs])

    gen_cfg = GenConfig(
        temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
        max_new_tokens=cfg.sampling.max_new_tokens, thinking=cfg.sampling.thinking,
    )

    # Evaluate the base instruct model and any trained adapters provided.
    targets = {"gemma-instruct": None}
    if adapters:
        targets.update(adapters)

    summary = {}
    for label, adapter_path in targets.items():
        model = build_model(cfg, cfg.training.base_model, adapter_path=adapter_path)
        records = []
        for spec in specs:
            records.extend(sample_continuations(model, spec, gen_cfg, n=10))
        scores = judge.score_many([r.continuation for r in records])
        for r, s in zip(records, scores):
            r.rating = s.rating
        write_jsonl(out_dir / f"{label}.jsonl", [asdict(r) for r in records])
        ratings = [r.rating for r in records if r.rating is not None]
        summary[label] = {
            "n": len(ratings),
            "mean": statistics.fmean(ratings) if ratings else 0.0,
            "pct_high": (sum(x >= 5 for x in ratings) / len(ratings)) if ratings else 0.0,
        }
    write_json(out_dir / "summary.json", summary)
    return summary
