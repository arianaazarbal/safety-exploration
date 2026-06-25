"""Compare internal negative emotion in vanilla vs DPO Gemma (Appendix I, Fig 14-15)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from ..data.wildchat import sample_wildchat_prompts
from .internal_emotions import EKMAN_EMOTIONS, EmotionProbe, load_probe_model


def run_internal_probe(
    frustrated_texts: list[str],
    dpo_adapter_path: str,
    *,
    layers: tuple[int, int] = (30, 40),
    out_dir: Path = config.RESULTS_DIR,
) -> pd.DataFrame:
    """Score the same frustrated conversations under both models and compare."""
    wc = sample_wildchat_prompts(n=500)

    rows = []
    for label, adapter in [("Gemma-3-27B-it", None), ("DPO-Gemma", dpo_adapter_path)]:
        model, tok = load_probe_model(adapter_path=adapter)
        probe = EmotionProbe(model, tok)
        probe.fit_standardisation(wc)
        for i, text in enumerate(frustrated_texts):
            scores = probe.score_text(text, layers=layers)
            scores.update({"model": label, "text_index": i})
            rows.append(scores)
        del model

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("model")[list(EKMAN_EMOTIONS) + ["negative_mean"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(Path(out_dir) / "appendixI_internal_emotions.csv", index=False)
    return summary
