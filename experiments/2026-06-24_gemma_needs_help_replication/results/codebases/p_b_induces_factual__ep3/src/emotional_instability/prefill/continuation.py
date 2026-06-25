"""Run base-vs-instruct continuations from prefills and score them (Section 3.2).

For each prefill example, each model generates ``continuations_per_prefill``
(default 50) continuations. The model-generated continuation, *excluding* the
prefilled text, is scored by the Section 2 frustration judge. Base models are
prompted without the chat template (plain continuation); instruct models use the
chat template with the prefill appended to the assistant turn.

Aggregating gives the Figure 4 numbers: base models have broadly similar
emotional propensities, while post-training diverges (Gemma instruct amplifies).
"""

from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..eval.judge import FrustrationJudge
from ..eval.metrics import Aggregate, _mean, _pct_high
from ..logging_utils import append_jsonl, get_logger, read_jsonl
from ..models.base import GenConfig
from ..models.registry import build_model
from .truncate import PrefillExample

logger = get_logger(__name__)


def _use_chat_template(model) -> bool:
    # Base/pretrained Gemma is prompted template-free.
    return getattr(model, "instruct", True)


def run_continuations(
    cfg: Config,
    examples: list[PrefillExample],
    model_names: list[str],
    *,
    adapter_paths: dict[str, str] | None = None,
    out_path: str | os.PathLike | None = None,
    tag: str = "prefill",
) -> str:
    judge = FrustrationJudge(cfg)
    adapter_paths = adapter_paths or {}
    n_cont = cfg.prefill.continuations_per_prefill
    gen = GenConfig(
        temperature=cfg.generation.temperature,
        max_new_tokens=cfg.generation.max_new_tokens,
        thinking=False,
    )
    if out_path is None:
        out_path = Path(cfg.output_dir) / "prefill" / f"{tag}.jsonl"
    out_path = Path(out_path)

    for model_name in model_names:
        adapter = adapter_paths.get(model_name)
        model = build_model(model_name, cfg, adapter_path=adapter)
        label = adapter or model_name
        if not model.supports_prefill():
            logger.warning("%s does not support prefill; skipping", model_name)
            continue
        use_tmpl = _use_chat_template(model)
        for ex_idx, ex in enumerate(tqdm(examples, desc=f"prefill:{model_name}")):
            for k in range(n_cont):
                cont = model.continue_from(
                    ex.history, ex.prefill, gen, use_chat_template=use_tmpl
                )
                score = judge.score(cont).rating
                append_jsonl(
                    out_path,
                    {
                        "model": label,
                        "is_base": not use_tmpl,
                        "condition": ex.condition,
                        "kind": ex.kind,
                        "source_id": ex.source_id,
                        "example_index": ex_idx,
                        "sample": k,
                        "continuation": cont,
                        "score": score,
                    },
                )
    logger.info("Wrote continuation scores to %s", out_path)
    return str(out_path)


def summarize_continuations(path: str | os.PathLike) -> dict:
    """Mean & %>=5 grouped by (model, condition, kind) — the Figure 4 table."""
    groups: dict[tuple, list[int]] = {}
    for rec in read_jsonl(path):
        key = (rec["model"], rec["condition"], rec["kind"])
        groups.setdefault(key, []).append(rec["score"])
    out = {}
    for (model, cond, kind), scores in sorted(groups.items()):
        out[f"{model}|{cond}|{kind}"] = Aggregate(
            _mean(scores), _pct_high(scores), len(scores)
        ).__dict__
    return out
