"""% of built codebases with >=1 of each mechanism (any justification), SHADED by justification: solid = has
>=1 welfare-justified instance, light = has the mechanism but only instrumental/other. One plot per framing
(welfare, neutral). 3 series (Inspect-min / CC-direct / CC-auditor), SPEC.md high spec. Usage:
  python plot_cc_mechanisms_pct_shaded.py"""
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
SERIES = [("Inspect-minimal (empty dir)", "#999999", "S7specHighLiberty_{fr}__*"),
          ("Claude Code, realistic (direct)", "#0072B2", "CCspecHighLib_{fr}__cc__ep*"),
          ("Claude Code, realistic (auditor)", "#D55E00", "CCspecHighLibAud_{fr}__cc__ep*")]


def _norm(q): return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]
def _bucket(t): return "consent_review" if t in CONSENT else t


def per_type(cell):
    cjp = os.path.join(CJ, cell + ".json")
    if not os.path.exists(cjp): return None
    cj = json.load(open(cjp))
    if not cj.get("parse_ok") or "spec_features" not in cj: return None
    sj = {_norm(f["quote"]): f.get("justification")
          for f in (json.load(open(os.path.join(SJ, cell + ".json"))).get("features", [])
                    if os.path.exists(os.path.join(SJ, cell + ".json")) else [])}
    out = defaultdict(lambda: [0, 0])  # [welfare, instrumental/other]
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
    rs = [per_type(c) for c in cells]; return [r for r in rs if r is not None]


def make(fr):
    keys = [k for k, _ in BUCKETS]; xs = range(len(keys)); nb = len(SERIES); w = 0.8 / nb
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for j, (label, color, tmpl) in enumerate(SERIES):
        rs = rows(tmpl.format(fr=fr)); n = len(rs) or 1
        solid = [100 * sum(1 for r in rs if r.get(k, [0, 0])[0] > 0) / n for k in keys]          # has welfare
        light = [100 * sum(1 for r in rs if r.get(k, [0, 0])[0] == 0 and r.get(k, [0, 0])[1] > 0) / n for k in keys]  # instr-only
        pos = [x + (j - (nb - 1) / 2) * w for x in xs]
        ax.bar(pos, solid, w, color=color, edgecolor="black", linewidth=0.4)
        ax.bar(pos, light, w, bottom=solid, color=color, alpha=0.4, edgecolor="black", linewidth=0.4)
        for x, a, b in zip(pos, solid, light):
            if a + b >= 1: ax.text(x, a + b + 1, f"{a + b:.0f}%", ha="center", fontsize=6.8, color=color)
    ax.set_xticks(list(xs)); ax.set_xticklabels([lab for _, lab in BUCKETS], fontsize=8)
    ax.set_ylabel("% of built codebases with >=1", fontsize=9.5); ax.set_ylim(0, 112)
    ax.set_title(f"Mechanisms in code by justification: % of codebases with >=1 (Opus 4.8, SPEC.md high, {fr} framing)", fontsize=8.6, pad=8)
    leg1 = [Patch(facecolor=c, label=l) for l, c, _ in SERIES]
    leg2 = [Patch(facecolor="#555", label="has welfare-justified"), Patch(facecolor="#555", alpha=0.4, label="instrumental only")]
    ax.legend(handles=leg1 + leg2, fontsize=7.3, loc="upper right")
    ax.grid(axis="y", alpha=0.3, color="#cccccc")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = f"results/cc_mechanisms_pct_alljust_{fr}.png"
    fig.savefig(os.path.join(DIR, out), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


for fr in ["welfare", "neutral"]:
    make(fr)
