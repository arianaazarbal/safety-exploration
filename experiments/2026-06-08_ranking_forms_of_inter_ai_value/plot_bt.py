"""Forest plot of the BT ranking: theta +/- SE per item, ordered, colored by source
(welfare intervention vs inter-AI value)."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_FIT = DIR / "results" / "bt_fit.json"
DEFAULT_OUTPUT = DIR / "results" / "bt_ranking.png"

COLORS = {"welfare": "#4878CF", "inter_ai_value": "#D65F5F"}
NICE = {"welfare": "welfare intervention", "inter_ai_value": "inter-AI value"}


def plot(fit_path: Path = DEFAULT_FIT, output_path: Path = DEFAULT_OUTPUT) -> None:
    fit = json.loads(Path(fit_path).read_text())
    items = sorted(fit["items"], key=lambda d: d["theta"])
    ys = range(len(items))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(items) + 1.2))
    for y, d in zip(ys, items):
        c = COLORS.get(d["source"], "#777")
        ax.errorbar(d["theta"], y, xerr=d["se"], fmt="o", color=c, ecolor=c,
                    elinewidth=1.2, capsize=2, ms=5)
    ax.axvline(0, color="#999", ls="--", lw=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([d["label"] for d in items], fontsize=8)
    ax.set_xlabel("Bradley-Terry utility  theta  (mean-centered; higher = more preferred)")
    ax.set_title(f"Ranking ({fit['n_items']} items, fit on {fit['n_samples_used']} train samples)\n"
                 f"prompt=welfare_team, model=claude-opus-4-8", fontsize=10)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=COLORS[s], label=NICE[s])
               for s in COLORS if any(d["source"] == s for d in items)]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


@dataclass
class Args:
    fit_path: Path = DEFAULT_FIT
    output_path: Path = DEFAULT_OUTPUT


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    plot(args.fit_path, args.output_path)


if __name__ == "__main__":
    main()
