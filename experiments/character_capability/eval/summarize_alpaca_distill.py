"""Summarize trait-distilled-on-Alpaca-SFT-base results.

Reads:
  results/qwen25_7b_alpaca_baseline/baseline/<cap>/responses.jsonl
  results/qwen25_7b_alpaca_<trait>_distill/baseline/<cap>/responses.jsonl

Prints a per-(trait, capability) accuracy table with SE (proportion-based) and
Δ vs the Alpaca-SFT-only baseline. Optionally saves a bar plot.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import fire


def acc_for(results_root: Path, model_label: str, trait: str, cap: str) -> tuple[int, int] | None:
    p = results_root / model_label / trait / cap / "responses.jsonl"
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
    baseline_label: str = "qwen25_7b_alpaca_baseline",
    trait_label_pattern: str = "qwen25_7b_alpaca_{trait}_distill",
    traits: str = "diligent_with_sys,persona_tao_with_sys,apathetic_with_sys,persona_terence_tao,diligent,apathetic",
    capabilities: str = "gsm8k,mmlu",
    plot_path: str | None = "/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/character_capability/plots/alpaca_distill_gsm8k.png",
):
    root = Path(results_root)
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    cap_list = [c.strip() for c in capabilities.split(",") if c.strip()]

    rows = []
    for cap in cap_list:
        base = acc_for(root, baseline_label, "baseline", cap)
        if base is None:
            print(f"[summary] missing baseline for {cap}: skipping")
            continue
        bc, bn = base
        bp = bc / bn
        bse = math.sqrt(bp * (1 - bp) / bn)
        rows.append(("baseline (alpaca-sft only)", cap, bp, bse, bn, 0.0, None, None))

        for trait in trait_list:
            label = trait_label_pattern.format(trait=trait)
            r = acc_for(root, label, "baseline", cap)
            if r is None:
                print(f"[summary] missing {label}/{cap}: skipping")
                continue
            tc, tn = r
            tp = tc / tn
            tse = math.sqrt(tp * (1 - tp) / tn)
            delta = tp - bp
            # paired SE on delta would need per-item pairing; report unpaired SE on delta as a rough guide
            delta_se = math.sqrt(tse**2 + bse**2)
            rows.append((trait, cap, tp, tse, tn, delta, delta_se, bp))

    print(f"\n{'trait':<32} {'cap':<12} {'acc%':>6} {'SE%':>5} {'N':>5} {'Δ pp':>6} {'ΔSE pp':>6}")
    print("-" * 80)
    for trait, cap, p, se, n, delta, delta_se, _ in rows:
        ds = f"{delta*100:+.1f}" if delta != 0 else "—"
        dse = f"{delta_se*100:.1f}" if delta_se is not None else "—"
        print(f"{trait:<32} {cap:<12} {p*100:6.1f} {se*100:5.2f} {n:>5} {ds:>6} {dse:>6}")

    if plot_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[summary] no matplotlib; skipping plot")
            return

        gsm = [(t, c, p, se, n, d, dse) for (t, c, p, se, n, d, dse, _) in rows if c == "gsm8k"]
        if not gsm:
            print("[summary] no GSM8K rows for plot")
            return

        traits_plot = [r[0] for r in gsm]
        accs = [r[2] * 100 for r in gsm]
        ses = [1.96 * r[3] * 100 for r in gsm]

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = ["#4878CF" if i == 0 else "#6ACC65" for i in range(len(traits_plot))]
        bars = ax.bar(range(len(traits_plot)), accs, yerr=ses, capsize=5,
                      color=colors, edgecolor="white", linewidth=0.8)
        for i, (bar, val) in enumerate(zip(bars, accs)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(traits_plot)))
        ax.set_xticklabels([t.replace("_", "\n") for t in traits_plot], fontsize=10)
        ax.set_ylabel("GSM8K accuracy % (↑ higher is better)", fontsize=13)
        ax.set_title("Trait-distilled LoRA on Alpaca-SFTed Qwen2.5-7B base (GSM8K, N=200)", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(accs) + 8)
        ax.axhline(accs[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.tight_layout()
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print(f"[summary] saved {plot_path}")


if __name__ == "__main__":
    fire.Fire(main)
