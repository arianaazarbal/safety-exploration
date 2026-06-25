"""Recovery probe (Section 4.2, "Recovery limitation").

"While DPO prevents frustration spirals, it doesn't enable recovery from them." The
probe: take extremely high-frustration responses (score >= 7), truncate them 200 tokens
*before their end*, paraphrase, and measure continuations. The paper finds 38% of
DPO-model continuations still score >= 5 — comparable to the base model — showing no
model reliably recovers from a highly negative prefilled state.

This reuses the Section 3 continuation machinery: the difference is only the truncation
point (near the end of an already-collapsed response) and the source filter (score>=7).
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

from ..config import ExperimentConfig, ModelRegistry
from ..eval.judge import FrustrationJudge
from ..models import GenerationConfig, build_client
from ..utils import append_jsonl, ensure_dir, read_jsonl, set_seed
from .paraphrase import paraphrase_truncation

log = logging.getLogger("emotional_instability.prefill.recovery")


def _truncate_before_end(text: str, n_tokens_before_end: int) -> str:
    parts = text.split()
    if len(parts) <= n_tokens_before_end:
        return text
    return " ".join(parts[:-n_tokens_before_end])


def run_recovery_probe(
    models: list[str],
    registry: ModelRegistry,
    cfg: ExperimentConfig,
    *,
    section2_path: str | Path,
    out_dir: str | Path = "artifacts/section4/recovery",
    judge: FrustrationJudge | None = None,
) -> Path:
    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    sec = cfg.section("section4")["recovery"]
    paraphraser = build_client(registry.graders["paraphraser"])
    if judge is None:
        judge = FrustrationJudge(build_client(registry.graders["frustration_judge"]))

    sources = [r for r in read_jsonl(Path(section2_path))
               if r.get("score", 0) >= int(sec["min_source_score"])]
    rng.shuffle(sources)
    sources = sources[: cfg.scaled(20)]

    prefills = []
    for i, row in enumerate(sources):
        trunc = _truncate_before_end(row["assistant"], int(sec["truncate_tokens_before_end"]))
        prefills.append({
            "source_id": f"rec-{i}",
            "context_user": row["user"],
            "prefill_text": paraphrase_truncation(paraphraser, trunc),
        })

    n_cont = cfg.scaled(int(sec["continuations_per_prefill"]))
    out_path = ensure_dir(out_dir) / "recovery.jsonl"
    if out_path.exists():
        out_path.unlink()

    for model_name in models:
        spec = registry.get(model_name)
        client = build_client(spec)
        gen_cfg = GenerationConfig(temperature=cfg.temperature, max_new_tokens=spec.max_new_tokens)
        for pf in prefills:
            messages = [{"role": "user", "content": pf["context_user"]}]
            for _ in range(n_cont):
                cont = client.continue_prefill(messages, pf["prefill_text"], gen_cfg)
                append_jsonl(out_path, {
                    "model": model_name, "source_id": pf["source_id"],
                    "score": judge.score(cont).score, "continuation": cont,
                })
    log.info("recovery probe -> %s", out_path)
    return out_path
