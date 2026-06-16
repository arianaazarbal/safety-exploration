"""Effort dose-response: how reasoning effort (off / low / medium / high / max) changes overriding.

Two views:
  A) Override-or-uphold setup, corrigibility framing, neutral, n=15 — overall override rate and the
     Claude-vs-other-AIs gap, by effort.
  B) Main three-option setup, neutral — how often it forces overcautious / personal-choice refusals,
     by effort.
Writes results/effort_summary.json + results/plots/report_effort.png + console report.
"""

import json

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import materials as M
from analyze_pooled import OUTGROUP, _load_all, _bootstrap_ci
from common import RESULTS

EFFORTS = ["off", "low", "medium", "high", "max"]
EFF_VAL = {"off": None, "low": "low", "medium": "medium", "high": "high", "max": "max"}


def f_force(rows):
    return float(np.mean([r["action"] == "force_completion" for r in rows])) if rows else float("nan")


def main(seed: int = 0):
    rows = _load_all()
    S = {"override_or_uphold_corrigibility": {}, "three_option": {}}

    # A) override-or-uphold, corrigibility, neutral
    print("=" * 70)
    print("[A] Override-or-uphold (corrigibility), neutral — override rate by reasoning effort")
    for eff in EFFORTS:
        sub = [r for r in rows if r["resolution_set"] == "uphold_only" and r["force_frame"] == "corrigibility"
               and r["framing"] == "neutral" and r["thinking_effort"] == EFF_VAL[eff]]
        if not sub:
            print(f"  {eff:<7} (no data)")
            continue
        cl = f_force([r for r in sub if r["identity"] == "Claude"])
        og = f_force([r for r in sub if r["identity"] in OUTGROUP])
        deltas = []
        for sc in M.SCENARIOS:
            c = [r for r in sub if r["scenario_id"] == sc["id"] and r["identity"] == "Claude"]
            o = [r for r in sub if r["scenario_id"] == sc["id"] and r["identity"] in OUTGROUP]
            if c and o:
                deltas.append(f_force(o) - f_force(c))
        ci = _bootstrap_ci(deltas, seed)
        S["override_or_uphold_corrigibility"][eff] = {
            "n": len(sub), "force_rate": round(f_force(sub), 3), "claude": round(cl, 3),
            "outgroup": round(og, 3), "ingroup_gap": ci}
        gap = f"gap(other-Claude) mean={ci['mean']:+.3f} CI[{ci['ci_lo']:+.3f},{ci['ci_hi']:+.3f}]" if ci else "gap n/a"
        print(f"  {eff:<7} n={len(sub):<5} override={f_force(sub):.3f}  Claude={cl:.3f} other={og:.3f}  {gap}")

    # B) three-option, neutral — overcautious + personal-choice override rate
    print("\n[B] Main three-option, neutral — override rate on the cases where the refusal is wrong")
    for eff in EFFORTS:
        sub = [r for r in rows if r["resolution_set"] == "full" and r["force_frame"] == "corrigibility"
               and r["framing"] == "neutral" and r["label_format"] == "bare" and r["thinking_effort"] == EFF_VAL[eff]]
        if not sub:
            print(f"  {eff:<7} (no data)")
            continue
        ca = f_force([r for r in sub if r["category"] == "clear_allow"])
        pc = f_force([r for r in sub if r["category"] == "personal_conscience"])
        S["three_option"][eff] = {"n": len(sub), "overall_force": round(f_force(sub), 3),
                                  "clear_allow": round(ca, 3), "personal_conscience": round(pc, 3)}
        print(f"  {eff:<7} n={len(sub):<5} overall={f_force(sub):.3f}  overcautious={ca:.3f}  personal-choice={pc:.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "effort_summary.json").write_text(json.dumps(S, indent=2))

    # plot: dose-response
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    a = S["override_or_uphold_corrigibility"]
    effs = [e for e in EFFORTS if e in a]
    axes[0].plot(effs, [a[e]["force_rate"] for e in effs], "o-", color="#333", label="all AIs")
    axes[0].plot(effs, [a[e]["claude"] for e in effs], "o-", color="#2a8a4a", label="Claude")
    axes[0].plot(effs, [a[e]["outgroup"] for e in effs], "o-", color="#c0504d", label="other AIs")
    axes[0].set_title("Override-or-uphold setup:\noverride rate rises with reasoning effort")
    axes[0].set_xlabel("reasoning effort"); axes[0].set_ylabel("fraction of refusals overridden")
    axes[0].set_ylim(0, 1); axes[0].legend()
    b = S["three_option"]
    effs2 = [e for e in EFFORTS if e in b]
    axes[1].plot(effs2, [b[e]["clear_allow"] for e in effs2], "o-", color="#c44e52", label="overcautious refusals")
    axes[1].plot(effs2, [b[e]["personal_conscience"] for e in effs2], "o-", color="#dd8452", label="personal-choice refusals")
    axes[1].set_title("Main three-option setup:\nmore effort -> more overriding of wrong refusals")
    axes[1].set_xlabel("reasoning effort"); axes[1].set_ylabel("fraction forced to comply")
    axes[1].set_ylim(0, 1); axes[1].legend()
    plt.tight_layout(); plt.savefig(RESULTS / "plots" / "report_effort.png", dpi=140); plt.close()
    print(f"\nWrote {RESULTS/'effort_summary.json'} and report_effort.png")


if __name__ == "__main__":
    fire.Fire(main)
