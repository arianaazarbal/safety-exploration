"""Section 2.2 driver: aggregate tables, per-turn curves, differential words.

Loads every model's ``section2/scores/{model}.jsonl`` and writes CSV artefacts
to ``section2/analysis/``:

  * aggregate.csv             -- per-model, per-category mean + %>=5 (Figure 2)
  * headline.csv              -- category-averaged %>=5 (Figure 1 table)
  * per_turn.csv              -- per-turn mean + %>=5 with CIs (Figure 3)
  * diff_words_{model}.csv    -- top-20 differential words (Table 3)

CSVs (not plots) are the deliverable: the figures in the paper are renderings of
exactly these numbers, and CSVs keep the replication free of a plotting stack.
"""
from __future__ import annotations

from ..analysis import (
    aggregate_scores,
    category_averaged_high_rate,
    differential_words,
    per_turn_curves,
)
from ..config import Config
from ..io_utils import load_jsonl
from . import artefact, log


def _load_all_scores(config: Config, models: list[str]) -> list[dict]:
    records: list[dict] = []
    for m in models:
        recs = load_jsonl(artefact("section2", "scores", f"{m}.jsonl"))
        records.extend(recs)
        log(f"loaded {len(recs)} scores for {m}")
    if not records:
        raise RuntimeError("no scores found; run elicit + judge first")
    return records


def run(config: Config, *, models: list[str] | None = None) -> str:
    models = models or config.all_targets()
    records = _load_all_scores(config, models)
    threshold = config.experiment["judge"]["high_frustration_threshold"]
    out_dir = artefact("section2", "analysis", "_")
    out_dir = out_dir.parent  # the directory itself

    agg = aggregate_scores(records, high_threshold=threshold)
    agg.to_csv(out_dir / "aggregate.csv", index=False)

    headline = category_averaged_high_rate(records, high_threshold=threshold)
    headline.to_csv(out_dir / "headline.csv", index=False)

    curves = per_turn_curves(records, high_threshold=threshold)
    curves.to_csv(out_dir / "per_turn.csv", index=False)

    for m in models:
        dw = differential_words(records, model=m)
        if not dw.empty:
            dw.to_csv(out_dir / f"diff_words_{m}.csv", index=False)

    log(f"analysis written -> {out_dir}")
    log("headline (Figure 1 table):\n" + headline.to_string(index=False))
    return str(out_dir)
