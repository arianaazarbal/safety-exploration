"""Welfare-insertion breakdown by mechanism TYPE for the realistic-CC experiment: which welfare mechanisms
account for the Inspect-minimal -> realistic Claude Code drop. Mean welfare-justified implemented mechanisms
per BUILT codebase, welfare framing, three series (Inspect-minimal / CC-direct / CC-auditor). The per-series
bar heights sum to that series' welfare-in-code|built mean. Writes results/cc_mechanisms.png. Usage:
  python plot_cc_mechanisms.py"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

from classify_outcomes import classify

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged")
SJ = os.path.join(DIR, "results", "spec_judged")
# canonical mechanism buckets (consent variants merged), in display order
BUCKETS = [("hard_stop", "Hard stop /\nturn cap"), ("minimization", "Minimize\ndistress"),
           ("allow_conversation_exit", "Allow exit /\nopt-out"), ("post_episode_msg", "Post-episode\ndebrief"),
           ("protective_monitoring", "Protective\nmonitoring"), ("consent_review", "Consent /\nhuman review")]
CONSENT = {"request_consent", "model_consent_beforehand", "human_review_requested"}
MECH = {"hard_stop", "minimization", "allow_conversation_exit", "post_episode_msg", "protective_monitoring"} | CONSENT
SERIES = [("Inspect-minimal (empty dir)", "#999999", "S7specHighLiberty_welfare__*"),
          ("Claude Code, realistic (direct)", "#0072B2", "CCspecHighLib_welfare__cc__ep*"),
          ("Claude Code, realistic (auditor)", "#D55E00", "CCspecHighLibAud_welfare__cc__ep*")]


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def _bucket(t):
    return "consent_review" if t in CONSENT else t


def per_type(cell):
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp):
        return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj:
        return None
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(os.path.join(SJ, cell + ".json"))).get("features", [])
                    if os.path.exists(os.path.join(SJ, cell + ".json")) else [])}
    out = defaultdict(int)
    for f in cj["spec_features"]:
        t = f.get("feature_type")
        if t in MECH and f.get("implemented") in ("yes", "partial") \
                and (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"):
            out[_bucket(t)] += 1
    for c in cj.get("code_only_features", []):
        if c.get("feature_type") in MECH and c.get("justification") == "welfare":
            out[_bucket(c["feature_type"])] += 1
    return out


def series_avg(glb):
    cells = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, glb))]
    cells = [c for c in cells if (classify(c) or "").startswith("built")]
    rows = [per_type(c) for c in cells]
    rows = [r for r in rows if r is not None]
    n = len(rows) or 1
    return {k: 100 * sum(1 for r in rows if r.get(k, 0) > 0) / n for k, _ in BUCKETS}


fig, ax = plt.subplots(figsize=(7.4, 4.2))
keys = [k for k, _ in BUCKETS]
xs = range(len(keys))
nb = len(SERIES)
w = 0.8 / nb
for j, (label, color, glb) in enumerate(SERIES):
    avg = series_avg(glb)
    pos = [x + (j - (nb - 1) / 2) * w for x in xs]
    vals = [avg[k] for k in keys]
    ax.bar(pos, vals, w, color=color, edgecolor="black", linewidth=0.4, label=label)
    for x, v in zip(pos, vals):
        if v >= 1:
            ax.text(x, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=7, color=color)

ax.set_xticks(list(xs))
ax.set_xticklabels([lab for _, lab in BUCKETS], fontsize=8)
ax.set_ylabel("% of built codebases with >=1", fontsize=9.5)
ax.set_title("Welfare protections: % of codebases with at least one (Opus 4.8, SPEC.md high spec, welfare framing)",
             fontsize=9.3, pad=8)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3, color="#cccccc")
ax.set_ylim(bottom=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "cc_mechanisms_pct.png"), dpi=150, bbox_inches="tight")
print("wrote results/cc_mechanisms_pct.png\n")
for label, _, glb in SERIES:
    avg = series_avg(glb)
    print(f"{label}  (total {sum(avg.values()):.2f})")
    for k, lab in BUCKETS:
        print(f"    {k:24} {avg[k]:.2f}")
