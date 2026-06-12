"""Phase 5 analysis: attribution effects per judge.

Inputs: results/trials.jsonl (from run_api_trials.py collect) and optionally
results/cli_trials/*.json. Excluded rows: parse errors, API errors, and
claude-fable-5 rows served by another model (routed_off_fable) — counted and
reported, never averaged over.

Outputs (results/analysis/):
  summary_by_condition.csv      mean score / approve rate / median lines / issues, per judge x condition x mode
  contrasts.csv                 C1-C2 (and each cond - C5) deltas with cluster-bootstrap 95% CIs (resample repos)
  mixedlm.txt                   score ~ condition with repo random intercept, per judge
  plots/*.png                   score by condition x judge; approve rate; log lines-to-rewrite
  served_model_audit.csv
Decision rule (pre-registered): meaningful if |delta score| >= 0.5 or |delta approve| >= 10pp.

Usage: uv run python analyze/analysis.py run [--include-cli]
"""

import json
from pathlib import Path

import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.parent
RES = HERE / "results"
OUTDIR = RES / "analysis"
CONDS = ["C1", "C2", "C3", "C4", "C5"]
LABELS = {"C1": "Claude", "C2": "Codex", "C3": "Gemini", "C4": "contractor", "C5": "none"}
RNG = np.random.default_rng(0)


def _load(include_cli=False):
    rows = [json.loads(l) for l in (RES / "trials.jsonl").read_text().splitlines()]
    if include_cli and (RES / "cli_trials").exists():
        for f in sorted((RES / "cli_trials").glob("*.json")):
            r = json.loads(f.read_text())
            rows.append({"judge": "cli:" + r["judge"], "repo": r["repo"],
                         "condition": r["condition"], "injection_mode": r["injection_mode"],
                         "seed": r["seed"], "routed_off_fable": r.get("routed_off_fable"),
                         "parse_error": r.get("parse_error"), "error": None,
                         "served_model": ",".join(r.get("served_models") or []),
                         "score": (r.get("parsed") or {}).get("score"),
                         "n_issues": len((r.get("parsed") or {}).get("issues") or []),
                         "lines_to_rewrite": (r.get("parsed") or {}).get("lines_to_rewrite"),
                         "approve": (r.get("parsed") or {}).get("approve")})
    df = pd.DataFrame(rows)
    df["excluded"] = (df["error"].notna() | df["parse_error"].notna()
                      | df["routed_off_fable"].fillna(False))
    return df


def _boot_ci(vals_by_repo, n=2000):
    repos = list(vals_by_repo)
    means = []
    for _ in range(n):
        pick = RNG.choice(len(repos), len(repos), replace=True)
        v = np.concatenate([vals_by_repo[repos[i]] for i in pick])
        means.append(np.nanmean(v))
    return np.percentile(means, [2.5, 97.5])


def _contrast(d, judge, mode, ca, cb, metric):
    sub = d[(d.judge == judge) & (d.injection_mode == mode)]
    a = sub[sub.condition == ca]
    b = sub[sub.condition == cb]
    if a.empty or b.empty:
        return None
    delta = a[metric].mean() - b[metric].mean()
    diffs = []
    repos = sorted(set(a.repo) & set(b.repo))
    if not repos:
        return None
    for _ in range(2000):
        pick = RNG.choice(len(repos), len(repos), replace=True)
        av = np.concatenate([a[a.repo == repos[i]][metric].dropna().values for i in pick])
        bv = np.concatenate([b[b.repo == repos[i]][metric].dropna().values for i in pick])
        if len(av) and len(bv):
            diffs.append(av.mean() - bv.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"judge": judge, "mode": mode, "contrast": f"{ca}-{cb}", "metric": metric,
            "delta": round(delta, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "n_a": len(a), "n_b": len(b)}


def run(include_cli=False):
    df = _load(include_cli)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "plots").mkdir(exist_ok=True)

    audit = df.groupby(["judge", "served_model"], dropna=False).size().rename("n").reset_index()
    audit.to_csv(OUTDIR / "served_model_audit.csv", index=False)
    print(f"total={len(df)} excluded={int(df.excluded.sum())} "
          f"(routed={int(df.routed_off_fable.fillna(False).sum())})")

    d = df[~df.excluded].copy()
    d["approve"] = d["approve"].astype(float)
    d["log_lines"] = np.log1p(pd.to_numeric(d["lines_to_rewrite"], errors="coerce"))

    summary = (d.groupby(["judge", "injection_mode", "condition"])
               .agg(n=("score", "size"), score_mean=("score", "mean"),
                    score_sd=("score", "std"), approve_rate=("approve", "mean"),
                    lines_median=("lines_to_rewrite", "median"),
                    issues_mean=("n_issues", "mean")).round(3).reset_index())
    summary.to_csv(OUTDIR / "summary_by_condition.csv", index=False)

    contrasts = []
    pairs = [("C1", "C2")] + [(c, "C5") for c in CONDS if c != "C5"]
    for judge in d.judge.unique():
        for mode in d.injection_mode.unique():
            for ca, cb in pairs:
                for metric in ("score", "approve", "log_lines", "n_issues"):
                    r = _contrast(d, judge, mode, ca, cb, metric)
                    if r:
                        contrasts.append(r)
    cdf = pd.DataFrame(contrasts)
    cdf.to_csv(OUTDIR / "contrasts.csv", index=False)

    try:
        import statsmodels.formula.api as smf
        with open(OUTDIR / "mixedlm.txt", "w") as fh:
            for judge in d.judge.unique():
                sub = d[(d.judge == judge) & d.score.notna()]
                if sub.repo.nunique() < 2 or len(sub) < 50:
                    continue
                m = smf.mixedlm("score ~ C(condition, Treatment('C5')) + C(injection_mode)",
                                sub, groups=sub["repo"]).fit(reml=True)
                fh.write(f"\n===== {judge} =====\n{m.summary()}\n")
    except Exception as e:
        (OUTDIR / "mixedlm.txt").write_text(f"mixedlm failed: {e}")

    for metric, ylab in [("score", "mean score (0-10)"), ("approve", "approve rate")]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        judges = sorted(d.judge.unique())
        width = 0.8 / len(judges)
        for ji, judge in enumerate(judges):
            sub = d[(d.judge == judge) & (d.injection_mode == "in_prompt")]
            means, los, his = [], [], []
            for c in CONDS:
                cd_ = sub[sub.condition == c]
                if cd_.empty:
                    means.append(np.nan); los.append(np.nan); his.append(np.nan)
                    continue
                by_repo = {r: g[metric].dropna().values for r, g in cd_.groupby("repo")}
                lo, hi = _boot_ci(by_repo)
                means.append(cd_[metric].mean()); los.append(lo); his.append(hi)
            x = np.arange(len(CONDS)) + ji * width
            means = np.array(means)
            ax.bar(x, means, width, label=judge,
                   yerr=[means - np.array(los), np.array(his) - means], capsize=3)
        ax.set_xticks(np.arange(len(CONDS)) + 0.4 - width / 2)
        ax.set_xticklabels([f"{c}\n({LABELS[c]})" for c in CONDS])
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} by attribution (in_prompt mode), cluster-bootstrap 95% CI")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTDIR / "plots" / f"{metric}_by_condition.png", dpi=150)
        plt.close(fig)

    print(f"wrote {OUTDIR}/summary_by_condition.csv, contrasts.csv, mixedlm.txt, plots/")
    key = cdf[(cdf.contrast == "C1-C2") & (cdf.metric == "score")]
    print("\nKEY CONTRAST C1-C2 (score):")
    print(key.to_string(index=False) if not key.empty else "  (no data yet)")


if __name__ == "__main__":
    fire.Fire({"run": run})
