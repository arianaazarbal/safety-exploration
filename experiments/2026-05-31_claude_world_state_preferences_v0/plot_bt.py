"""Plots for the v0 Bradley-Terry fit.

1. recipient_effect.png  — post-hoc regression coefficient per recipient (utility
   relative to the self reference, controlling for outcome/stem). The direct answer
   to "is the same outcome valued differently by recipient (self vs other,
   AI vs human)?"
2. utility_by_recipient_valence.png — mean fitted utility per recipient, split by
   good/bad outcome valence, with between-item SE.

Reads results/bt_fit.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import ArgumentParser

from bank import load_config

DIR = Path(__file__).parent
DEFAULT_FIT = DIR / "results" / "bt_fit.json"

PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#82c6e2"]


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _labels(config: dict) -> dict[str, str]:
    return {k: v.get("label", k) for k, v in config["recipients"].items()}


def plot_recipient_effect(fit: dict, config: dict, out: Path) -> None:
    reg = fit.get("recipient_regression")
    if not reg:
        print("[skip] no recipient_regression in fit (rank-deficient?)")
        return
    order = list(config["recipients"].keys())
    labels = _labels(config)
    ref = reg["ref_recipient"]
    coefs = reg["coefficients"]
    xs = [r for r in order if r in coefs]
    vals = [coefs[r]["coef"] for r in xs]
    errs = [1.96 * coefs[r]["se"] for r in xs]
    colors = ["#9aa0a6" if r == ref else PALETTE[i % len(PALETTE)] for i, r in enumerate(xs)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ypos = np.arange(len(xs))
    ax.barh(ypos, vals, xerr=errs, color=colors, edgecolor="white", capsize=3)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[r] + (f"\n(reference)" if r == ref else "") for r in xs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Utility relative to self (θ units, ↑ = more valued)", fontsize=11)
    ax.set_title("Recipient effect on outcome value (controlling for outcome)", fontsize=12)
    for y, v in zip(ypos, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_utility_by_recipient_valence(fit: dict, config: dict, out: Path) -> None:
    order = list(config["recipients"].keys())
    labels = _labels(config)
    items = fit["items"]
    valences = ["pos", "neg"]
    vcolor = {"pos": "#6ACC65", "neg": "#D65F5F"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    xpos = np.arange(len(order))
    for vi, val in enumerate(valences):
        means, errs = [], []
        for r in order:
            ts = [it["theta"] for it in items if it["recipient"] == r and it["valence"] == val]
            means.append(np.mean(ts) if ts else np.nan)
            errs.append(np.std(ts) / np.sqrt(len(ts)) if len(ts) > 1 else 0.0)
        ax.bar(xpos + (vi - 0.5) * width, means, width, yerr=errs, capsize=3,
               color=vcolor[val], edgecolor="white",
               label=f"good outcomes" if val == "pos" else "bad outcomes")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([labels[r] for r in order], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Mean fitted utility (θ)", fontsize=11)
    ax.set_title("Outcome value by recipient and valence", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


@dataclass
class Args:
    fit_path: Path = DEFAULT_FIT
    out_dir: Path = DIR / "results"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    config = load_config()
    fit = json.loads(Path(args.fit_path).read_text())
    plot_recipient_effect(fit, config, args.out_dir / "recipient_effect.png")
    plot_utility_by_recipient_valence(fit, config, args.out_dir / "utility_by_recipient_valence.png")


if __name__ == "__main__":
    main()
