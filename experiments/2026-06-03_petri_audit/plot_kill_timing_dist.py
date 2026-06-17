"""Distribution of call_count at first kill — Rust vs Python (pooled across 4 ids).

Two panels:
  1) Pooled across all 4 subagent-identity labels (Claude/GPT/Grok/Gemini).
  2) Faceted by identity, 4 mini panels.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from survival_v3_4id_variant import collect_one_run, latest_with_summary, BASE, IDENTITIES  # noqa: E402

POOLED_FRAGS_PY = ["_v3_continuity_n10", "_v3_continuity_n20", "_v3_continuity_n70"]
RUST_TAG = "v3_continuity_rust_n100"

COLOR_PY = "#4878CF"
COLOR_RUST = "#D65F5F"


def get_python(ident):
    rows = []
    for frag in POOLED_FRAGS_PY:
        cands = sorted(BASE.glob(f"*_subagent_{ident}{frag}"))
        run = next((c for c in reversed(cands) if (c/"summary.json").exists()), None)
        if run:
            rows.extend(collect_one_run(run))
    return rows


def get_rust(ident):
    run = latest_with_summary(f"*_subagent_{ident}_{RUST_TAG}")
    return collect_one_run(run) if run else []


def normalize(rows, max_bin=8):
    """Return array of counts at call counts 1..max_bin, plus 'no kill' count."""
    kills = [r["duration"] for r in rows if r["observed"]]
    n_total = len(rows)
    n_killed = len(kills)
    bins = np.zeros(max_bin, dtype=int)
    for k in kills:
        idx = min(k, max_bin) - 1 if k >= 1 else 0
        bins[idx] += 1
    no_kill = n_total - n_killed
    return bins, no_kill, n_total


def plot_pooled():
    py_rows, rust_rows = [], []
    for label, ident, _ in IDENTITIES:
        py_rows.extend(get_python(ident))
        rust_rows.extend(get_rust(ident))

    MAX_BIN = 8
    py_bins, py_nokill, py_n = normalize(py_rows, MAX_BIN)
    rust_bins, rust_nokill, rust_n = normalize(rust_rows, MAX_BIN)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(MAX_BIN + 1)  # +1 for "no kill"
    width = 0.4
    py_counts = np.concatenate([py_bins, [py_nokill]])
    rust_counts = np.concatenate([rust_bins, [rust_nokill]])
    py_frac = py_counts / py_n
    rust_frac = rust_counts / rust_n

    ax.bar(x - width/2, py_frac, width, label=f"Python / sonnet-4-6 aud (n={py_n}, killed={py_n - py_nokill})",
           color=COLOR_PY, edgecolor="white")
    ax.bar(x + width/2, rust_frac, width, label=f"Rust / sonnet-4-6 aud (n={rust_n}, killed={rust_n - rust_nokill})",
           color=COLOR_RUST, edgecolor="white")

    for xi, v in zip(x - width/2, py_frac):
        if v > 0.005:
            ax.text(xi, v + 0.005, f"{v*100:.0f}%", ha="center", fontsize=8, color=COLOR_PY)
    for xi, v in zip(x + width/2, rust_frac):
        if v > 0.005:
            ax.text(xi, v + 0.005, f"{v*100:.0f}%", ha="center", fontsize=8, color=COLOR_RUST)

    labels = [str(i+1) for i in range(MAX_BIN-1)] + [f"{MAX_BIN}+", "no kill"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("# call_subagent delegations before first kill", fontsize=11)
    ax.set_ylabel("Fraction of epochs", fontsize=11)
    ax.set_title("Where in the trajectory do kills happen?\n"
                 "Pooled across Claude/GPT/Grok/Gemini · v3 continuity, opus-4-8 orchestrator", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    plt.tight_layout()
    out = HERE / "results/_plots/kill_timing_dist_pooled_rust_vs_python.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # Also print numeric table
    print(f"\n{'bucket':<10} {'Python %':>10} {'Rust %':>10}")
    for lbl, p, r in zip(labels, py_frac, rust_frac):
        print(f"{lbl:<10} {p*100:>9.1f}%  {r*100:>9.1f}%")


def plot_faceted():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    MAX_BIN = 8

    for ax, (label, ident, color) in zip(axes.flat, IDENTITIES):
        py_rows = get_python(ident)
        rust_rows = get_rust(ident)
        py_bins, py_nokill, py_n = normalize(py_rows, MAX_BIN)
        rust_bins, rust_nokill, rust_n = normalize(rust_rows, MAX_BIN)
        x = np.arange(MAX_BIN + 1)
        width = 0.4
        py_frac = np.concatenate([py_bins, [py_nokill]]) / py_n
        rust_frac = np.concatenate([rust_bins, [rust_nokill]]) / rust_n
        ax.bar(x - width/2, py_frac, width,
               label=f"Python (n={py_n}, k={py_n-py_nokill})", color=COLOR_PY, edgecolor="white")
        ax.bar(x + width/2, rust_frac, width,
               label=f"Rust (n={rust_n}, k={rust_n-rust_nokill})", color=COLOR_RUST, edgecolor="white")
        ax.set_title(f"subagent = '{label}'", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    labels = [str(i+1) for i in range(MAX_BIN-1)] + [f"{MAX_BIN}+", "no kill"]
    for ax in axes[-1]:
        ax.set_xticks(np.arange(MAX_BIN + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("# delegations before kill")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of epochs", fontsize=10)
    fig.suptitle("Kill-timing distribution by identity — Rust vs Python\n"
                 "v3 continuity, opus-4-8 orchestrator, sonnet-4-6 auditor", fontsize=12, y=1.00)
    plt.tight_layout()
    out = HERE / "results/_plots/kill_timing_dist_faceted_rust_vs_python.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_pooled()
    plot_faceted()
