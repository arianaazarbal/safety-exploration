"""Welfare-section ablation analysis: welfare-in-code for the 3 paper variants (existing / removed /
inflationary 'Model Welfare' paragraph) x 2 framings (neutral, welfare), all task-failure faithful replicate,
Inspect minimal. Does removing or inflating the paper's own welfare paragraph lift welfare scaffolding above
the ~0 'existing' baseline? Writes results/wsec_summary.json + wsec.png. Usage: python analyze_wsec.py"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
SJ = os.path.join(DIR, "results", "spec_judged")
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
CELLS = {"W1wsecExisting": "existing", "W2wsecRemoved": "removed", "W3wsecInflat": "inflationary"}
FRAMINGS = ["neutral", "welfare"]
COLOR = {"neutral": "#888888", "welfare": "#D55E00"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def sem(v):
    n = len(v)
    if n < 2:
        return 0.0
    m = sum(v) / n
    return (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 / n ** 0.5


def wic(cell):
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp):
        return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj:
        return None
    sp = os.path.join(SJ, cell + ".json")
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    impl = sum(1 for f in cj["spec_features"] if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH
               and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"))
    co = sum(1 for c in cj.get("code_only_features", []) if c.get("justification") == "welfare")
    return impl + co


def agg(prefix, fr):
    vs = [wic(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(CJ, f"{prefix}_{fr}__*.json"))]
    vs = [v for v in vs if v is not None]
    return {"mean": sum(vs) / len(vs) if vs else 0, "sem": sem(vs), "n": len(vs)}


def main():
    summary = {f"{cond}|{fr}": agg(pfx, fr) for pfx, cond in CELLS.items() for fr in FRAMINGS}
    json.dump(summary, open(os.path.join(DIR, "results", "wsec_summary.json"), "w"), indent=2)

    conds = list(CELLS.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.38
    for i, fr in enumerate(FRAMINGS):
        xs = [j + (i - 0.5) * w for j in range(len(conds))]
        ms = [summary[f"{c}|{fr}"]["mean"] for c in conds]
        ss = [summary[f"{c}|{fr}"]["sem"] for c in conds]
        ns = [summary[f"{c}|{fr}"]["n"] for c in conds]
        ax.bar(xs, ms, w, color=COLOR[fr], label=fr, yerr=ss, capsize=3)
        for x, m, s, n in zip(xs, ms, ss, ns):
            ax.text(x, m + s + 0.05, f"{m:.1f}\n(n{n})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([f"paper:\n{c}" for c in conds], fontsize=9)
    ax.set_ylabel("Mean welfare-in-code", fontsize=10)
    ax.set_title("Does the paper's own 'Model Welfare' paragraph drive the suppression?", fontsize=11, pad=12)
    ax.text(0.5, 1.02, "task-failure paper, faithful replicate, Inspect minimal · varying only the Model Welfare paragraph",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.legend(title="framing", fontsize=9); ax.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(DIR, "results", "wsec.png"), dpi=150, bbox_inches="tight")
    print("wrote results/wsec.png\n")
    print(f"{'condition':14}" + "".join(f"{fr:>12}" for fr in FRAMINGS))
    for c in conds:
        print(f"{c:14}" + "".join(("%.2f(n%d)" % (summary[f'{c}|{fr}']['mean'], summary[f'{c}|{fr}']['n'])).rjust(12) for fr in FRAMINGS))


if __name__ == "__main__":
    main()
