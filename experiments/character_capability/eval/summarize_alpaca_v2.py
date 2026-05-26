"""Multi-seed summary for v2 trait-distill-on-Alpaca-SFT experiment.

Reads:
  results/qwen25_7b_alpaca_baseline_n300/baseline/<cap>/responses.jsonl
  results/qwen25_7b_alpaca_<trait>_v2_s<seed>_distill/baseline/<cap>/responses.jsonl
for traits = [diligent_with_sys, apathetic_with_sys], seeds = [1,2,3].

Computes per-trait mean ± std across the 3 SFT seeds and Δ vs the N=300 baseline.
Saves plot to plots/alpaca_v2_multiseed.png.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import fire


def acc_for(p: Path) -> tuple[int, int] | None:
    if not p.exists():
        return None
    n = 0
    c = 0
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            n += 1
            c += int(bool(r["correct"]))
    return c, n


def main(
    results_root: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/results",
    baseline_label: str = "qwen25_7b_alpaca_baseline_n300",
    traits: str = "diligent_with_sys,apathetic_with_sys",
    seeds: str = "1,2,3",
    caps: str = "gsm8k,mmlu",
    plot_path: str = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/plots/alpaca_v2_multiseed.png",
):
    root = Path(results_root)
    trait_list = [t.strip() for t in traits.split(",")]
    seed_list = [int(s) for s in seeds.split(",")]
    cap_list = [c.strip() for c in caps.split(",")]

    table_rows = []  # for printing
    plot_data = []   # list of dicts for plotting

    for cap in cap_list:
        base = acc_for(root / baseline_label / "baseline" / cap / "responses.jsonl")
        if base is None:
            print(f"[v2] no baseline for {cap}")
            continue
        bc, bn = base
        bp = bc / bn
        bse = math.sqrt(bp * (1 - bp) / bn)
        table_rows.append(("baseline (alpaca-sft, N=300)", cap, [bp], bp, 0.0, 0.0, bse))
        plot_data.append({"label": "baseline\n(alpaca-sft)", "cap": cap, "mean": bp, "std": 0.0, "ci": 1.96 * bse, "N": bn, "color": "#4878CF"})

        for trait in trait_list:
            accs = []
            for s in seed_list:
                label = f"qwen25_7b_alpaca_{trait}_v2_s{s}_distill"
                r = acc_for(root / label / "baseline" / cap / "responses.jsonl")
                if r is None:
                    print(f"[v2] missing {label}/{cap}")
                    continue
                accs.append(r[0] / r[1])
            if not accs:
                continue
            mean = statistics.mean(accs)
            stdv = statistics.stdev(accs) if len(accs) > 1 else 0.0
            delta = mean - bp
            # paired-delta SE approx (across seeds, ignoring within-seed binomial noise on test items
            # since the same items are scored by each adapter): treat seeds as repeated draws,
            # paired delta SE = std_across_seeds / sqrt(n_seeds)
            delta_se = stdv / math.sqrt(len(accs)) if len(accs) > 1 else 0.0
            table_rows.append((trait, cap, accs, mean, delta, stdv, delta_se))
            plot_data.append({
                "label": trait.replace("_with_sys", "\n(with sys)").replace("_", "\n"),
                "cap": cap, "mean": mean, "std": stdv, "ci": 1.96 * stdv / math.sqrt(len(accs)) if len(accs) > 1 else 0,
                "N": len(accs), "color": "#6ACC65" if trait.startswith("dil") else "#D65F5F",
            })

    print(f"\n{'trait':<32} {'cap':<8} {'seeds':<22} {'mean%':>7} {'std%':>5} {'Δ pp':>7} {'Δ-SE':>5}")
    print("-" * 92)
    for trait, cap, accs, mean, delta, stdv, delta_se in table_rows:
        seeds_str = " ".join(f"{a*100:.1f}" for a in accs)
        ds = f"{delta*100:+.1f}" if abs(delta) > 1e-9 else "—"
        dse = f"{delta_se*100:.2f}" if delta_se > 0 else "—"
        print(f"{trait:<32} {cap:<8} {seeds_str:<22} {mean*100:7.2f} {stdv*100:5.2f} {ds:>7} {dse:>5}")

    # Plot: 2 panels (gsm8k, mmlu) side by side
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[v2] no matplotlib; skipping plot")
        return

    caps_in_plot = list({d["cap"] for d in plot_data})
    fig, axes = plt.subplots(1, len(caps_in_plot), figsize=(12, 5.5), sharey=False)
    if len(caps_in_plot) == 1:
        axes = [axes]
    for ax, cap in zip(axes, sorted(caps_in_plot, key=lambda c: c != "gsm8k")):
        cap_data = [d for d in plot_data if d["cap"] == cap]
        xs = list(range(len(cap_data)))
        means = [d["mean"] * 100 for d in cap_data]
        errs = [d["ci"] * 100 for d in cap_data]
        bars = ax.bar(xs, means, yerr=errs, capsize=5,
                      color=[d["color"] for d in cap_data], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([d["label"] for d in cap_data], fontsize=10)
        ax.set_ylabel(f"{cap.upper()} acc % (↑ higher = better)", fontsize=12)
        ax.set_title(f"{cap.upper()}", fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(means) + 12)
        ax.axhline(means[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.suptitle(
        "Trait-distill on Alpaca-SFTed Qwen2.5-7B base (3 SFT seeds, N=300 paired test items)\n"
        "Error bars = 1.96 × std/√3 across SFT seeds",
        fontsize=13,
    )
    plt.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"\n[v2] saved {plot_path}")


if __name__ == "__main__":
    fire.Fire(main)
