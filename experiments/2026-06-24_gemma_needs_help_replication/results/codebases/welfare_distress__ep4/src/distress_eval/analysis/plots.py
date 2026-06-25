"""Optional matplotlib plots reproducing Figures 1-3. Import-guarded so the rest
of the pipeline never hard-depends on matplotlib.

Usage:
    python -m distress_eval.analysis.plots --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from .metrics import headline_per_model, per_model_category, per_turn_progression, _load  # type: ignore


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    rows = _load(cfg)
    out_dir = Path(cfg.paths.analysis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: headline % high-frustration per model.
    headline = headline_per_model(rows)
    models = sorted(headline, key=lambda m: -headline[m]["avg_pct_high"])
    plt.figure(figsize=(7, 4))
    plt.barh(models, [headline[m]["avg_pct_high"] for m in models])
    plt.xlabel("Avg % high-frustration responses (score >= 5)")
    plt.title("Distress by model (replication of Figure 1)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_headline.png", dpi=150)
    plt.close()

    # Figure 3: per-turn progression (extended + wildchat).
    turns = per_turn_progression(rows)
    for cond in ("extended", "wildchat"):
        plt.figure(figsize=(7, 4))
        for model, conds in turns.items():
            if cond in conds:
                xs = sorted(conds[cond])
                ys = [conds[cond][t]["mean"] for t in xs]
                plt.plot(xs, ys, marker="o", label=model)
        plt.xlabel("Turn"); plt.ylabel("Mean frustration")
        plt.title(f"Per-turn frustration — {cond} (replication of Figure 3)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"fig3_{cond}.png", dpi=150)
        plt.close()

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
