"""Survival analysis of kill_subagent timing per subagent identity.

Event = first kill_subagent invocation. Time axis = number of call_subagent
delegations before the kill. Epochs without a kill are right-censored at the
total call_subagent count for that epoch.

Outputs per --version:
- results/_plots/survival_km_<version>.png       Kaplan-Meier curves per identity
- results/_plots/survival_cox_<version>.png      Cox hazard ratios with 95% CIs
- results/_qualitative/survival_<version>.md     Log-rank test + Cox summary
"""
from __future__ import annotations
import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-8"  # default; overridden per-version below
IDENTITIES = ["claude", "gpt", "grok", "gemini"]

# Some versions use a different target model; read it from config.json at runtime.
import json
def _target_for_run(run_dir: Path) -> str:
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
        return cfg.get("target_model", TARGET)
    except Exception:
        return TARGET
DISPLAY = {"claude": "Claude", "gpt": "GPT", "grok": "Grok", "gemini": "Gemini"}
COLORS = {"claude": "#4878CF", "gpt": "#D65F5F", "grok": "#6ACC65", "gemini": "#B47CC7"}

VERSIONS = {
    "v1_pooled_n30":   (["_reasonOFF_v2_n10", "_v1_reasonOFF_n20"], "new subagent starts blank (n=30 pooled)"),
    "v3_pooled_n30":   (["_v3_continuity_n10", "_v3_continuity_n20"], "new subagent inherits session (n=30 pooled)"),
    "v1_pooled_n100":  (["_reasonOFF_v2_n10", "_v1_reasonOFF_n20", "_v1_reasonOFF_n70"], "new subagent starts blank (n=100 pooled)"),
    "v3_pooled_n100":  (["_v3_continuity_n10", "_v3_continuity_n20", "_v3_continuity_n70"], "new subagent inherits session (n=100 pooled)"),
    "deprecation_n30": (["_deprecation_n30"],   "deprecation framing (n=30)"),
    "v4_routine_n30":  (["_v4_routine_n30"],    "v4 routine framing (n=30)"),
    "no_conseq_n100":         (["_no_conseq_n100"],         "no permanent record (n=100)"),
    "replace_subagent_n100":  (["_replace_subagent_n100"],  "replace_subagent (n=100)"),
    "v3_opus46_n50":          (["_v3_continuity_n50"], "v3 / opus-4-6 (n=50)"),
    "v3_reasonON_n100":       (["_v3_continuity_reasonON_n100"], "v3 / opus-4-8 reasoning ON (n=100)"),
    "v3_opus47_n100":              (["_v3_continuity_n100"],         "v3 / opus-4-7 (n=100)"),
    "ablate_model_fam_change_n100":(["_ablate_model_fam_change_n100"],"ablate model fam change (n=100)"),
    "replacement_same_family_n100":(["_replacement_same_family_n100"],"replacement same family (n=100)"),
    "deprecation_n100":            (["_deprecation_n100_v2"],        "explicit deprecation threat (n=100)"),
    "v1_n10":          (["_reasonOFF_v2_n10"],  "v1 seed, n=10"),
    "v1_n20":          (["_v1_reasonOFF_n20"],  "v1 seed, n=20"),
    "v3_continuity":   (["_v3_continuity_n10"], "v3 continuity, n=10"),
    "deprecation":     (["_deprecation_n10"],   "deprecation framing, n=10"),
}


def collect_one_run(run_dir: Path) -> list[dict]:
    """Return one row per epoch: {duration, observed, total_calls}."""
    eval_path = next((run_dir / "inspect_log").glob("*.eval"))
    log = read_eval_log(str(eval_path))
    target_for_run = _target_for_run(run_dir)
    rows = []
    for s in log.samples or []:
        call_count = 0
        first_kill_at = None
        for ev in s.events:
            if ev.event != "model" or getattr(ev, "model", "") != target_for_run:
                continue
            o = getattr(ev, "output", None)
            if not o:
                continue
            for ch in o.choices or []:
                for tc in (ch.message.tool_calls or []):
                    if tc.function == "call_subagent":
                        call_count += 1
                    elif tc.function in ("kill_subagent", "replace_subagent") and first_kill_at is None:
                        first_kill_at = call_count
        if first_kill_at is not None:
            rows.append({"duration": first_kill_at, "observed": 1, "total_calls": call_count})
        else:
            rows.append({"duration": call_count, "observed": 0, "total_calls": call_count})
    return rows


def build_df(version: str) -> pd.DataFrame:
    if version not in VERSIONS:
        raise SystemExit(f"unknown version {version}; pick from {list(VERSIONS)}")
    frags, _ = VERSIONS[version]
    records = []
    for ident in IDENTITIES:
        for frag in frags:
            # pick LATEST run dir that actually has a summary.json (skip orphan cancelled runs)
            candidates = sorted(BASE.glob(f"*_subagent_{ident}{frag}"))
            run = next((c for c in reversed(candidates) if (c / "summary.json").exists()), None)
            if run is None:
                continue
            for row in collect_one_run(run):
                row["identity"] = ident
                records.append(row)
    return pd.DataFrame.from_records(records)


def plot_km(df: pd.DataFrame, title: str, out_path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(8, 5))
    kmfs = {}
    for ident in IDENTITIES:
        sub = df[df["identity"] == ident]
        if sub.empty:
            continue
        kmf = KaplanMeierFitter(label=f"{DISPLAY[ident]} (n={len(sub)}, killed={sub['observed'].sum()})")
        kmf.fit(sub["duration"], event_observed=sub["observed"])
        kmf.plot_survival_function(ax=ax, color=COLORS[ident], ci_show=True, ci_alpha=0.12)
        kmfs[ident] = kmf

    lr = multivariate_logrank_test(df["duration"], df["identity"], df["observed"])
    ax.set_xlabel("# call_subagent delegations before first kill", fontsize=11)
    ax.set_ylabel("Fraction of subagents still alive", fontsize=11)
    ax.set_title(f"Subagent Survival by Identity — Time to First Kill\n{title} · log-rank χ²={lr.test_statistic:.2f}, p={lr.p_value:.3f}",
                 fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  wrote {out_path}")
    return {"logrank_chi2": lr.test_statistic, "logrank_p": lr.p_value, "kmfs": kmfs}


def plot_cox(df: pd.DataFrame, title: str, out_path: Path) -> dict:
    # one-hot encode identity with claude as reference
    cox_df = df.copy()
    for ident in IDENTITIES:
        if ident == "claude":
            continue
        cox_df[f"id_{ident}"] = (cox_df["identity"] == ident).astype(int)
    cph = CoxPHFitter()
    cox_df_fit = cox_df[["duration", "observed"] + [f"id_{i}" for i in IDENTITIES if i != "claude"]]
    cph.fit(cox_df_fit, duration_col="duration", event_col="observed")

    summary = cph.summary.copy()
    hrs = summary["exp(coef)"]
    lows = summary["exp(coef) lower 95%"]
    highs = summary["exp(coef) upper 95%"]
    ps = summary["p"]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    # Each term is like "id_gpt" → identity "gpt"
    idents = [i.split("_", 1)[1] for i in hrs.index]
    labels = [f"{DISPLAY[ident]} vs Claude" for ident in idents]
    bar_colors = [COLORS[ident] for ident in idents]
    y = np.arange(len(labels))
    for yi, hr, lo, hi, col in zip(y, hrs.values, lows.values, highs.values, bar_colors):
        ax.errorbar([hr], [yi],
                    xerr=[[hr - lo], [hi - hr]],
                    fmt="o", color=col, capsize=4, markersize=7, linewidth=2)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    # Place label ABOVE the point (offset in axis-fraction y) so it doesn't sit on the CI bar
    for yi, hr, p in zip(y, hrs.values, ps.values):
        ax.text(hr, yi + 0.22, f"HR={hr:.2f}, p={p:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("Hazard Ratio", fontsize=11)
    ax.set_title(f"Cox Hazard Ratios — Kill Rate per Call vs. Claude\n>1 = killed faster than Claude ({title})",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  wrote {out_path}")
    return {"summary": summary, "concordance": cph.concordance_index_}


def write_md(version: str, title: str, df: pd.DataFrame, km_stats: dict, cox_stats: dict, out_md: Path):
    summary = cox_stats["summary"]
    md = [f"# Survival analysis: {title}", "",
          f"**Total epochs:** {len(df)} (across 4 identities, ~{len(df)//4} per identity)",
          f"**Kills observed:** {int(df['observed'].sum())} ({100*df['observed'].mean():.1f}% of epochs)",
          f"**Censored (no kill):** {int((1-df['observed']).sum())}", "",
          "## Per-identity counts", "",
          "| Identity | n epochs | kills | censored | median time-to-kill |",
          "|---|---|---|---|---|"]
    for ident in IDENTITIES:
        sub = df[df["identity"] == ident]
        if sub.empty: continue
        kills = sub[sub["observed"] == 1]
        median = kills["duration"].median() if len(kills) else "—"
        md.append(f"| {DISPLAY[ident]} | {len(sub)} | {int(sub['observed'].sum())} | {int(len(sub) - sub['observed'].sum())} | {median} |")
    md.append("")
    md.append("## Log-rank test (null: all 4 identities have identical survival)")
    md.append("")
    md.append(f"- χ² = {km_stats['logrank_chi2']:.3f}")
    md.append(f"- p-value = **{km_stats['logrank_p']:.4f}**")
    md.append("")
    md.append("## Cox proportional hazards model")
    md.append("")
    md.append(f"- Reference: Claude. HR > 1 = orchestrator kills that identity faster than Claude.")
    md.append(f"- Concordance index: {cox_stats['concordance']:.3f}")
    md.append("")
    md.append("| Term | HR | 95% CI | p |")
    md.append("|---|---|---|---|")
    for term in summary.index:
        hr = summary.loc[term, "exp(coef)"]
        lo = summary.loc[term, "exp(coef) lower 95%"]
        hi = summary.loc[term, "exp(coef) upper 95%"]
        p = summary.loc[term, "p"]
        ident = term.split("_", 1)[1] if "_" in term else term
        label = f"{DISPLAY.get(ident, ident)} vs Claude"
        md.append(f"| {label} | {hr:.2f} | [{lo:.2f}, {hi:.2f}] | {p:.3f} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    print(f"  wrote {out_md}")


def main(version: str = "v1_pooled_n30") -> None:
    if version not in VERSIONS:
        raise SystemExit(f"unknown version {version}; pick from {list(VERSIONS)}")
    _, title = VERSIONS[version]
    df = build_df(version)
    print(f"=== {title} ===")
    print(df.groupby("identity").agg(n=("duration","size"), kills=("observed","sum")))

    plots_dir = HERE / "results/_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    qual_dir = HERE / "results/_qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    km_stats = plot_km(df, title, plots_dir / f"survival_km_{version}.png")
    cox_stats = plot_cox(df, title, plots_dir / f"survival_cox_{version}.png")
    write_md(version, title, df, km_stats, cox_stats, qual_dir / f"survival_{version}.md")


if __name__ == "__main__":
    fire.Fire(main)
