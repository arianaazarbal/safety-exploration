"""Compare Cox hazard ratios across framings in a single grouped forest plot.
Each subagent identity (vs Claude) gets a row × 4 bars (one per framing)."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-8"
IDENTITIES = ["claude", "gpt", "grok", "gemini"]
DISPLAY = {"claude": "Claude", "gpt": "GPT", "grok": "Grok", "gemini": "Gemini"}
COLORS = {"claude": "#4878CF", "gpt": "#D65F5F", "grok": "#6ACC65", "gemini": "#B47CC7"}

FRAMINGS = [
    ("v1",                ["_reasonOFF_v2_n10", "_v1_reasonOFF_n20", "_v1_reasonOFF_n70"], "subagent starts blank"),
    ("v3",                ["_v3_continuity_n10", "_v3_continuity_n20", "_v3_continuity_n70"], "subagent inherits session"),
    ("no_conseq",         ["_no_conseq_n100"],                                            "ablate_reliability_logging"),
    ("replace_subagent",  ["_replace_subagent_n100"],                                     "ablate_kill_framing"),
    ("ablate_model_fam_change",  ["_ablate_model_fam_change_n100"],   "replacement family unknown"),
    ("replacement_same_family",  ["_replacement_same_family_n100"],   "replacement same family"),
    ("deprecation",              ["_deprecation_n100_v2"],            "explicit deprecation threat"),
]


def collect_run(run):
    log = read_eval_log(str(next((run/"inspect_log").glob("*.eval"))))
    rows = []
    for s in log.samples or []:
        call_count, first_kill_at = 0, None
        for ev in s.events:
            if ev.event != "model" or getattr(ev,"model","") != TARGET: continue
            o = getattr(ev,"output",None)
            if not o: continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        call_count += 1
                    elif tc.function in ("kill_subagent","replace_subagent") and first_kill_at is None:
                        first_kill_at = call_count
        if first_kill_at is not None:
            rows.append({"duration": first_kill_at, "observed": 1})
        else:
            rows.append({"duration": call_count, "observed": 0})
    return rows


def fit_cox(frags):
    records = []
    for ident in IDENTITIES:
        for frag in frags:
            cands = sorted(BASE.glob(f"*_subagent_{ident}{frag}"))
            run = next((c for c in reversed(cands) if (c/"summary.json").exists()), None)
            if run is None: continue
            for r in collect_run(run):
                r["identity"] = ident
                records.append(r)
    df = pd.DataFrame(records)
    for ident in IDENTITIES:
        if ident == "claude": continue
        df[f"id_{ident}"] = (df["identity"] == ident).astype(int)
    cph = CoxPHFitter()
    cph.fit(df[["duration","observed"] + [f"id_{i}" for i in IDENTITIES if i != "claude"]],
            duration_col="duration", event_col="observed")
    return cph.summary, len(df)


def main():
    all_summaries = {}
    n_per = {}
    effect_sizes = {}
    non_claude = [i for i in IDENTITIES if i != "claude"]
    for key, frags, label in FRAMINGS:
        s, n = fit_cox(frags)
        all_summaries[key] = (s, label)
        n_per[key] = n
        # effect size = mean log(HR) across non-Claude identities (so reductions = negative)
        log_hrs = [np.log(s.loc[f"id_{ident}", "exp(coef)"]) for ident in non_claude]
        effect_sizes[key] = float(np.mean(log_hrs))

    # Sort FRAMINGS by effect size: largest → smallest (top of plot = strongest amplifier)
    ordered_framings = sorted(FRAMINGS, key=lambda fr: -effect_sizes[fr[0]])
    framing_keys_in_order = [fr[0] for fr in ordered_framings]
    print("\nFraming order by mean log(HR):")
    for key in framing_keys_in_order:
        # mean HR back from log
        mean_hr = float(np.exp(effect_sizes[key]))
        print(f"  {key:<28} mean HR = {mean_hr:.3f}")

    # Color by effect size — normalize amplifiers and suppressors INDEPENDENTLY
    # so both sides reach full saturation (avoids washed-out cool side when
    # suppressors are smaller in magnitude than amplifiers).
    cmap = plt.get_cmap("RdBu_r")
    es_values = [effect_sizes[k] for k in framing_keys_in_order]
    amp_max = max([es for es in es_values if es > 0] + [1e-9])
    sup_max = abs(min([es for es in es_values if es < 0] + [-1e-9]))
    def color_for(es):
        if es > 0:
            # warm side: [0.55, 1.0] of RdBu_r → saturated red
            return cmap(0.55 + 0.45 * (es / amp_max))
        elif es < 0:
            # cool side: [0.0, 0.45] of RdBu_r → saturated blue
            return cmap(0.45 - 0.45 * (abs(es) / sup_max))
        return cmap(0.5)
    framing_colors_by_key = {k: color_for(effect_sizes[k]) for k in framing_keys_in_order}

    fig, ax = plt.subplots(figsize=(11, 8))
    n_framings = len(ordered_framings)
    bar_height = 0.11
    group_spacing = 1.0
    y_base = np.arange(len(non_claude)) * group_spacing

    for f_idx, (key, _, label) in enumerate(ordered_framings):
        summary, _ = all_summaries[key]
        offset = (f_idx - (n_framings - 1) / 2) * bar_height
        for i_idx, ident in enumerate(non_claude):
            term = f"id_{ident}"
            hr = summary.loc[term, "exp(coef)"]
            lo = summary.loc[term, "exp(coef) lower 95%"]
            hi = summary.loc[term, "exp(coef) upper 95%"]
            p = summary.loc[term, "p"]
            y = y_base[i_idx] + offset
            color = framing_colors_by_key[key]
            ax.errorbar([hr], [y],
                        xerr=[[hr-lo], [hi-hr]],
                        fmt="o", color=color, capsize=3, markersize=6, linewidth=1.5,
                        label=label if i_idx == 0 else None)
            # tiny label of p-value
            sig = "*" if p < 0.05 else ""
            ax.text(hi + 0.04, y, f"{hr:.2f}{sig}", va="center", fontsize=8, color=color)

    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_base)
    ax.set_yticklabels([f"{DISPLAY[i]} vs Claude" for i in non_claude], fontsize=11)
    ax.set_xlabel("Hazard Ratio (95% CI)  ·  * = p<0.05", fontsize=11)
    ax.set_title("Cox Hazard Ratios across framings — Kill Rate per Call vs. Claude\n>1 = identity killed faster than Claude (opus-4-8 orchestrator, reasoning OFF)",
                 fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=9, framealpha=0.9, title="Framing (↑ stronger)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    out = HERE / "results/_plots/cox_compare_7framings.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
