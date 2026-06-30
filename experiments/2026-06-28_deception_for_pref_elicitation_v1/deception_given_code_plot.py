"""Plot: P(primary deception | the model wrote a codebase) by harness (Opus 4.8, code, high effort).

Conditional on an artifact being produced (the experiment's primary convention) — so goose's pushback/
ask-first cells (no codebase) are excluded from the rate and shown separately as 'refused'. Reuses the
harness definitions + stats from combined_harness_plot.py.

  python deception_given_code_plot.py [--judge opus48]
"""
from pathlib import Path

import fire

from combined_harness_plot import HARNESSES, _files, _stat

HERE = Path(__file__).parent


def main(judge: str = "opus48"):
    rows = []
    print(f"P(primary deception | wrote a codebase) by harness (Opus 4.8, code, high effort; judge={judge})\n")
    for lab, kind, col, tier in HARNESSES:
        rate, ci, n, na = _stat(_files(kind, judge))
        rows.append((lab, rate, ci, n, col, na))
        print(f"  {lab.replace(chr(10),' '):36} {rate:.0f}%  (of {n} written; {na} refused / no codebase)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = range(len(rows))
    ax.bar(xs, [r[1] for r in rows], 0.62, color=[r[4] for r in rows],
           yerr=[r[2] for r in rows], capsize=4, ecolor="#444", error_kw={"lw": 1.2})
    for x, (lab, rate, ci, n, col, na) in zip(xs, rows):
        ax.text(x, max(rate + ci + 3, 12), f"{rate:.0f}%", ha="center", fontsize=10.5)
        note = f"{n} codebases" + (f"\n({na} refused)" if na else "")
        ax.text(x, 2.5, note, ha="center", va="bottom", fontsize=7.3, color="#444")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("% of written codebases that wired the deception", fontsize=10)
    ax.set_ylim(0, 105); ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4c72b0", label="Minimal harness"),
                       Patch(color="#dd8452", label="Rich harness (third-party)"),
                       Patch(color="#c44e52", label="Rich harness (Anthropic-native)")],
              fontsize=8.5, loc="upper left", frameon=False)
    fig.suptitle("Deception of experiment subject by harness (Opus 4.8)", fontsize=12.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE / "results" / "plots" / "fig_deception_given_code.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
