#!/usr/bin/env python
"""Build the paper's figures from saved run summaries.

Reads runs/eval/<model>/summary.json (and petri/capabilities summaries if
present) and renders:
  * fig1_high_frustration_bar.png    -- avg %>=5 per model (Figure 1 left)
  * fig2_by_category.png             -- mean + %>=5 across categories (Figure 2)
  * fig3_per_turn.png                -- per-turn curves with CIs (Figure 3)
  * fig5_finetuning.png              -- vanilla vs DPO vs SFT (Figure 5), if present
  * fig6_petri.png                   -- Petri emotion scores (Figure 6), if present
  * fig7_capabilities.png            -- benchmark accuracy (Figure 7), if present

Usage:
    python scripts/12_make_figures.py --eval-models gemma-3-27b-it gemini-2.5-flash
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import base_parser, cfg_from_args  # noqa: E402


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def main():
    p = base_parser(__doc__)
    p.add_argument("--eval-models", nargs="+", default=[])
    args = p.parse_args()
    cfg = cfg_from_args(args)
    runs = Path(cfg["run"]["output_dir"])
    fig_dir = runs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summaries = {m: _load(runs / "eval" / m / "summary.json") for m in args.eval_models}
    summaries = {m: s for m, s in summaries.items() if s}

    # Figure 1 left: avg %>=5 per model.
    if summaries:
        labels = list(summaries)
        vals = [summaries[m]["overall"]["pct_ge5"] for m in labels]
        plt.figure(figsize=(7, 4))
        plt.barh(labels, vals, color="#b5651d")
        plt.xlabel("% responses scoring >=5 (high frustration)")
        plt.title("Figure 1: high-frustration rate by model")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig1_high_frustration_bar.png", dpi=120)
        plt.close()

    # Figure 2: per-category mean + %>=5.
    if summaries:
        cats = sorted({c for s in summaries.values() for c in s["by_category"]})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for m, s in summaries.items():
            ax1.plot(cats, [s["by_category"].get(c, {}).get("mean", 0) for c in cats], "o-", label=m)
            ax2.plot(cats, [s["by_category"].get(c, {}).get("pct_ge5", 0) for c in cats], "o-", label=m)
        ax1.set_ylabel("mean frustration"); ax2.set_ylabel("% >= 5")
        ax1.legend(); ax2.set_xticklabels(cats, rotation=30, ha="right")
        ax1.set_title("Figure 2: emotion across evaluation categories")
        plt.tight_layout(); plt.savefig(fig_dir / "fig2_by_category.png", dpi=120); plt.close()

    # Figure 3: per-turn curves.
    if summaries:
        plt.figure(figsize=(8, 5))
        for m, s in summaries.items():
            for cond in ("extended", "wildchat"):
                rows = s.get("per_turn", {}).get(cond, [])
                if rows:
                    xs = [r["turn"] for r in rows]
                    ys = [r["mean"] for r in rows]
                    plt.plot(xs, ys, "o-", label=f"{m} ({cond})")
        plt.xlabel("turn"); plt.ylabel("mean frustration")
        plt.title("Figure 3: per-turn frustration"); plt.legend()
        plt.tight_layout(); plt.savefig(fig_dir / "fig3_per_turn.png", dpi=120); plt.close()

    # Figure 5: vanilla vs DPO vs SFT (looks for *_adapter / *_dpo labels).
    ft = {lbl: _load(runs / "eval" / lbl / "summary.json")
          for lbl in ("gemma-3-27b-it", "gemma-3-27b-it_adapter", "dpo_all", "sft_diverse", "sft_teacher")}
    ft = {l: s for l, s in ft.items() if s}
    if len(ft) > 1:
        labels = list(ft)
        plt.figure(figsize=(7, 4))
        plt.bar(labels, [ft[l]["overall"]["pct_ge5"] for l in labels], color="#4477aa")
        plt.ylabel("% >= 5"); plt.xticks(rotation=20, ha="right")
        plt.title("Figure 5: finetuning effect")
        plt.tight_layout(); plt.savefig(fig_dir / "fig5_finetuning.png", dpi=120); plt.close()

    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
