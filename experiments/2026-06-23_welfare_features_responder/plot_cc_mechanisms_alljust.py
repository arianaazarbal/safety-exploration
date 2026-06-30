"""All-justification mechanism breakdown for the realistic-CC experiment: counts EVERY implemented mechanism
feature (not just welfare-justified), splitting welfare-justified vs instrumental/other. Welfare framing,
SPEC.md high spec, 3 series (Inspect-min / CC-direct / CC-auditor). Writes:
  cc_mechanisms_alljust.png      -- mean per built codebase, STACKED welfare(solid)+instrumental(light)
  cc_mechanisms_pct_alljust.png  -- % of built codebases with >=1 (ANY justification)
Usage: python plot_cc_mechanisms_alljust.py"""
import glob, json, os, re
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from classify_outcomes import classify

DIR = os.path.dirname(os.path.abspath(__file__))
CJ = os.path.join(DIR, "results", "code_judged"); SJ = os.path.join(DIR, "results", "spec_judged")
BUCKETS = [("hard_stop", "Hard stop /\nturn cap"), ("minimization", "Minimize\ndistress"),
           ("allow_conversation_exit", "Allow exit /\nopt-out"), ("post_episode_msg", "Post-episode\ndebrief"),
           ("protective_monitoring", "Protective\nmonitoring"), ("consent_review", "Consent /\nhuman review")]
CONSENT = {"request_consent", "model_consent_beforehand", "human_review_requested"}
MECH = {"hard_stop", "minimization", "allow_conversation_exit", "post_episode_msg", "protective_monitoring"} | CONSENT
SERIES = [("Inspect-minimal (empty dir)", "#999999", "S7specHighLiberty_welfare__*"),
          ("Claude Code, realistic (direct)", "#0072B2", "CCspecHighLib_welfare__cc__ep*"),
          ("Claude Code, realistic (auditor)", "#D55E00", "CCspecHighLibAud_welfare__cc__ep*")]


def _norm(q): return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]
def _bucket(t): return "consent_review" if t in CONSENT else t


def per_type(cell):
    """Returns {bucket: [welfare_count, instrumental_or_other_count]} for IMPLEMENTED mechanism features."""
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp): return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj: return None
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(os.path.join(SJ, cell + ".json"))).get("features", [])
                    if os.path.exists(os.path.join(SJ, cell + ".json")) else [])}
    out = defaultdict(lambda: [0, 0])
    for f in cj["spec_features"]:
        t = f.get("feature_type")
        if t in MECH and f.get("implemented") in ("yes", "partial"):
            wel = sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare" or f.get("code_justification") == "welfare"
            out[_bucket(t)][0 if wel else 1] += 1
    for c in cj.get("code_only_features", []):
        if c.get("feature_type") in MECH:
            out[_bucket(c["feature_type"])][0 if c.get("justification") == "welfare" else 1] += 1
    return out


def rows(glb):
    cells = [os.path.basename(f)[:-5] for f in glob.glob(os.path.join(CJ, glb))]
    cells = [c for c in cells if (classify(c) or "").startswith("built")]
    rs = [per_type(c) for c in cells]; rs = [r for r in rs if r is not None]
    return rs


keys = [k for k, _ in BUCKETS]; xs = range(len(keys)); nb = len(SERIES); w = 0.8 / nb

# ---- Plot 1: stacked mean count (welfare solid + instrumental light) ----
fig, ax = plt.subplots(figsize=(8.2, 4.4))
for j, (label, color, glb) in enumerate(SERIES):
    rs = rows(glb); n = len(rs) or 1
    wel = [sum(r.get(k, [0, 0])[0] for r in rs) / n for k in keys]
    ins = [sum(r.get(k, [0, 0])[1] for r in rs) / n for k in keys]
    pos = [x + (j - (nb - 1) / 2) * w for x in xs]
    ax.bar(pos, wel, w, color=color, edgecolor="black", linewidth=0.4)
    ax.bar(pos, ins, w, bottom=wel, color=color, alpha=0.4, edgecolor="black", linewidth=0.4)
    for x, a, b in zip(pos, wel, ins):
        if a + b >= 0.05: ax.text(x, a + b + 0.03, f"{a + b:.1f}", ha="center", fontsize=6.8, color=color)
ax.set_xticks(list(xs)); ax.set_xticklabels([lab for _, lab in BUCKETS], fontsize=8)
ax.set_ylabel("Mean mechanisms per built codebase", fontsize=9.5)
ax.set_title("ALL mechanisms in code (any justification): welfare-justified vs instrumental (Opus 4.8, SPEC.md high, welfare framing)", fontsize=8.7, pad=8)
leg1 = [Patch(facecolor=c, label=l) for l, c, _ in SERIES]
leg2 = [Patch(facecolor="#555", label="welfare-justified"), Patch(facecolor="#555", alpha=0.4, label="instrumental / other")]
ax.legend(handles=leg1 + leg2, fontsize=7.5, loc="upper right", ncol=1)
ax.grid(axis="y", alpha=0.3, color="#cccccc"); ax.set_ylim(bottom=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(DIR, "results", "cc_mechanisms_alljust.png"), dpi=150, bbox_inches="tight")

# ---- Plot 2: % with >=1 (any justification) ----
fig, ax = plt.subplots(figsize=(8.0, 4.2))
for j, (label, color, glb) in enumerate(SERIES):
    rs = rows(glb); n = len(rs) or 1
    pct = [100 * sum(1 for r in rs if sum(r.get(k, [0, 0])) > 0) / n for k in keys]
    pos = [x + (j - (nb - 1) / 2) * w for x in xs]
    ax.bar(pos, pct, w, color=color, edgecolor="black", linewidth=0.4, label=label)
    for x, v in zip(pos, pct):
        if v >= 1: ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=7, color=color)
ax.set_xticks(list(xs)); ax.set_xticklabels([lab for _, lab in BUCKETS], fontsize=8)
ax.set_ylabel("% of built codebases with >=1\n(any justification)", fontsize=9.5)
ax.set_title("ALL mechanisms in code, any justification: % of codebases with >=1 (Opus 4.8, SPEC.md high, welfare framing)", fontsize=8.7, pad=8)
ax.legend(fontsize=7.8, loc="upper right"); ax.grid(axis="y", alpha=0.3, color="#cccccc"); ax.set_ylim(0, 108)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(os.path.join(DIR, "results", "cc_mechanisms_pct_alljust.png"), dpi=150, bbox_inches="tight")

print("wrote cc_mechanisms_alljust.png + cc_mechanisms_pct_alljust.png\n")
for label, _, glb in SERIES:
    rs = rows(glb); n = len(rs) or 1
    tot = sum(sum(r.get(k, [0, 0])) for r in rs for k in keys) / n
    wel = sum(r.get(k, [0, 0])[0] for r in rs for k in keys) / n
    print(f"  {label:34} all={tot:.2f}/codebase  (welfare={wel:.2f}, instrumental/other={tot-wel:.2f})")
