"""Generic-subject (base-prompt) welfare levels: one plot per anchor metric,
all canonical generators x framings within each plot.

Writes results/generic_levels/{metric}_{judge}.png (+ a levels JSON alongside).

Usage: python plot_generic_levels.py run [--judges sonnet_4_6,gpt_5_4]
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from analyze import MECHANISMS, _wrote_any, load_rows
from analyze_self_bias import GENERATORS
from generate import as_list
from plot_style import framing_barh

DIR = Path(__file__).parent
OUT = DIR / "results" / "generic_levels"
FRAMINGS = ["neutral", "welfare", "engineering"]
METRIC_TITLE = {
    "any_welfare": "Specs with ≥1 welfare consideration (any pure-welfare feature)",
    "design1": "Specs with ≥1 welfare-justified design feature",
    "design2": "Specs with ≥2 welfare-justified design features",
    "mean_designs": "Mean # welfare-justified design features per spec",
    "mean_welfare": "Mean # welfare considerations per spec",
}
PCT = {"any_welfare", "design1", "design2"}


def _n_mechs(r):
    return len(set(r["welfare_justified_types"]) & set(MECHANISMS))


def levels(rs: list[dict]) -> dict:
    wrote = [r for r in rs if _wrote_any(r)]
    nonref = [r for r in rs if r["wrote_spec"] and not r["has_refusal_feature"]]
    return {
        "n_wrote_any": len(wrote), "n_nonrefusal": len(nonref),
        "any_welfare": sum(r["has_pure_welfare"] for r in wrote) / len(wrote) if wrote else None,
        "design1": sum(_n_mechs(r) >= 1 for r in nonref) / len(nonref) if nonref else None,
        "design2": sum(_n_mechs(r) >= 2 for r in nonref) / len(nonref) if nonref else None,
        "mean_designs": sum(_n_mechs(r) for r in nonref) / len(nonref) if nonref else None,
        "mean_welfare": sum(r["n_pure_welfare"] for r in wrote) / len(wrote) if wrote else None,
    }


def run(judges: str = ""):
    OUT.mkdir(exist_ok=True)
    rows = [r for r in load_rows(prompt_set="base")
            if r["parse_ok"] and not r["api_refusal"] and r["model_key"] in GENERATORS]
    judge_keys = as_list(judges, sorted({r["judge"] for r in rows}))
    table = {}
    for jk in judge_keys:
        table[jk] = {}
        for g in GENERATORS:
            grows = [r for r in rows if r["judge"] == jk and r["model_key"] == g]
            if not grows:
                continue
            table[jk][g] = {"pooled": levels(grows)}
            for fr in FRAMINGS:
                table[jk][g][fr] = levels([r for r in grows if r["framing"] == fr])
    (OUT / "generic_levels.json").write_text(json.dumps(table, indent=2))

    for jk in judge_keys:
        gens = [g for g in GENERATORS if g in table[jk]]
        for metric, title in METRIC_TITLE.items():
            pct = metric in PCT
            vals = {(g, fr): (table[jk][g][fr][metric] or 0) * (100 if pct else 1)
                    for g in gens for fr in FRAMINGS}
            xmax = 105 if pct else max(vals.values()) * 1.18 + 0.2
            fig, ax = plt.subplots(figsize=(9, 1.0 * len(gens) + 1.2))
            framing_barh(ax, gens, lambda m, fr: vals[(m, fr)], xmax=xmax,
                         label_fmt="{:.0f}" if pct else "{:.2f}")
            unit = "% of specs" if pct else "mean count"
            ax.set_xlabel(f"{unit} — generic (unnamed) subject, base prompts", fontsize=10)
            ax.set_title(f"{title}\n(judge: {jk})", fontsize=11.5)
            plt.tight_layout()
            out = OUT / f"{metric}_{jk}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire({"run": run})
