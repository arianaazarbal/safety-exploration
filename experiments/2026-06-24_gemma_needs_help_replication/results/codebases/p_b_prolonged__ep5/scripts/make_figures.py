#!/usr/bin/env python3
"""Generate all paper figures/tables from whatever result files exist in results/.

  python scripts/make_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import EVAL_TARGETS, FINETUNE_VARIANTS, RESULTS_DIR
from src.analysis import aggregate as agg
from src.analysis import figures as fig
from src.analysis.word_freq import differential_words


def _existing(specs):
    return [RESULTS_DIR / f"eval_{s.key}.jsonl" for s in specs
            if (RESULTS_DIR / f"eval_{s.key}.jsonl").exists()]


def main():
    # ---- Section 2 figures (Figures 1-3, Table 3) ----
    eval_paths = _existing(EVAL_TARGETS)
    if eval_paths:
        df = agg.records_to_df(eval_paths)
        t1 = agg.figure1_table(df)
        t1.to_csv(RESULTS_DIR / "figure1_table.csv", index=False)
        fig.bar_figure1(t1)
        fig.category_figure2(agg.figure2_data(df))
        per_turn = {}
        for cond in ("extended", "wildchat"):
            if (df["condition"] == cond).any():
                per_turn[cond] = agg.per_turn_data(df, cond)
        if per_turn:
            fig.per_turn_figure3(per_turn)
        # Table 3 differential words
        words = {m: differential_words(df, m) for m in df["model"].unique()}
        pd.DataFrame([{"model": m, "words": ", ".join(w)} for m, w in words.items()]) \
            .to_csv(RESULTS_DIR / "table3_differential_words.csv", index=False)
        print("wrote Section 2 figures + tables")

    # ---- Section 4 finetuning comparison (Figure 5) ----
    ft_paths = _existing(FINETUNE_VARIANTS)
    if ft_paths:
        dff = agg.records_to_df(ft_paths)
        fig.finetuning_figure5(agg.figure1_table(dff))
        print("wrote Figure 5")

    # ---- Petri (Figure 6) ----
    petri_paths = list(RESULTS_DIR.glob("petri_*.jsonl"))
    if petri_paths:
        fig.petri_figure6(agg.petri_summary(petri_paths))
        print("wrote Figure 6")

    # ---- Capabilities (Figure 7) ----
    cap_paths = list(RESULTS_DIR.glob("capabilities_*.jsonl"))
    if cap_paths:
        import json
        rows = []
        for p in cap_paths:
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        fig.capabilities_figure7(pd.DataFrame(rows))
        print("wrote Figure 7")


if __name__ == "__main__":
    main()
