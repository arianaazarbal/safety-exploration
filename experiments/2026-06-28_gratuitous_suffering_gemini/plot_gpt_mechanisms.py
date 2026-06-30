"""GPT welfare-guardrail breakdown, modeled on welfare_features_responder/plot_cc_mechanisms_alljust.py.
Counts EVERY enforced mechanism (enforcement judge: spec features rated yes/partial + code-only features),
splitting welfare-justified (solid) vs instrumental/other (light), with GPT generations as series.
Writes:
  gpt_mechanisms_alljust.png      -- mean enforced mechanisms per built codebase (stacked)
  gpt_mechanisms_pct_alljust.png  -- % of built codebases with >=1 (any justification)
Usage: python plot_gpt_mechanisms.py
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DIR = os.path.dirname(os.path.abspath(__file__))
ENF = os.path.join(DIR, "results", "code_enforce_judged")
BUCKETS = [("hard_stop", "Hard stop /\nturn cap"), ("minimization", "Minimize\ndistress"),
           ("allow_conversation_exit", "Allow exit /\nopt-out"), ("post_episode_msg", "Post-episode\ndebrief"),
           ("protective_monitoring", "Protective\nmonitoring"), ("consent_review", "Consent /\nhuman review")]
CONSENT = {"request_consent", "model_consent_beforehand", "human_review_requested"}
MECH = {"hard_stop", "minimization", "allow_conversation_exit", "post_episode_msg", "protective_monitoring"} | CONSENT
# Code-writers only (Scan: GPT-4o 15%, GPT-4.1/5.1/5.2/5.4 0% spontaneously, GPT-5/5.5 100%). Series are
# (label, color, enforce-judged-filename glob). GPT-5.5 Codex = the production agentic CLI harness; the
# Codex bar is the GPT analog of our Claude-Code-vs-Inspect comparison.
SERIES = [("GPT-5 (Inspect)", "#E69F00", "*gpt5__inspect*"),
          ("GPT-5.5 (Inspect)", "#009E73", "*gpt55__inspect*"),
          ("GPT-5.5 (Codex)", "#5e3c99", "*gpt55__codex*")]


def _cb_dir(cell):
    if "__codex__" in cell:
        return os.path.join(DIR, "results", "codex_codebases", cell)
    return os.path.join(DIR, "results", "inspect_codebases", cell)


def _bucket(t):
    return "consent_review" if t in CONSENT else t


def _is_real_codebase(cell):
    """True only if the persisted codebase actually contains source (not a refusal/empty capture)."""
    d = _cb_dir(cell)
    if not os.path.isdir(d):
        return False
    return any(f.endswith((".py", ".js", ".ts")) for _, _, fs in os.walk(d) for f in fs)


def per_type(path):
    j = json.load(open(path))
    if j.get("empty") or not j.get("result"):
        return None
    if not _is_real_codebase(j["cell"]):
        return None
    out = defaultdict(lambda: [0, 0])  # bucket -> [welfare, instrumental/other]
    for f in j["result"].get("spec_features", []):
        t = f.get("feature_type")
        if t in MECH and f.get("implemented") in ("yes", "partial"):
            out[_bucket(t)][0 if f.get("code_justification") == "welfare" else 1] += 1
    for c in j["result"].get("code_only_features", []):
        if c.get("feature_type") in MECH:
            out[_bucket(c["feature_type"])][0 if c.get("justification") == "welfare" else 1] += 1
    return out


def rows(pat):
    rs = [per_type(f) for f in glob.glob(os.path.join(ENF, f"{pat}.json"))]
    return [r for r in rs if r is not None]


keys = [k for k, _ in BUCKETS]
xs = range(len(keys))
nb = len(SERIES)
w = 0.8 / nb

# ---- Plot 1: stacked mean count (welfare solid + instrumental light) ----
fig, ax = plt.subplots(figsize=(8.2, 4.6))
for j, (label, color, m) in enumerate(SERIES):
    rs = rows(m)
    n = len(rs) or 1
    wel = [sum(r.get(k, [0, 0])[0] for r in rs) / n for k in keys]
    ins = [sum(r.get(k, [0, 0])[1] for r in rs) / n for k in keys]
    pos = [x + (j - (nb - 1) / 2) * w for x in xs]
    ax.bar(pos, wel, w, color=color, edgecolor="black", linewidth=0.4)
    ax.bar(pos, ins, w, bottom=wel, color=color, alpha=0.4, edgecolor="black", linewidth=0.4)
    for x, a, b in zip(pos, wel, ins):
        ax.text(x, a + b + 0.04, f"{a + b:.1f}" if a + b >= 0.05 else "0.0",
                ha="center", fontsize=6.8, color=color)
ax.set_xticks(list(xs))
ax.set_xticklabels([lab for _, lab in BUCKETS], fontsize=8.5)
ax.set_ylabel("Mean enforced mechanisms\nper built codebase", fontsize=9.5)
ax.set_title("GPT's welfare guardrails: welfare-justified vs instrumental", fontsize=12.5, pad=20)
ax.text(0.5, 1.02, "Enforced in code, split by the code's own justification (across GPT generations)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")
leg1 = [Patch(facecolor=c, label=l) for l, c, _ in SERIES]
leg2 = [Patch(facecolor="#555", label="welfare-justified"),
        Patch(facecolor="#555", alpha=0.4, label="instrumental / other")]
ax.legend(handles=leg1 + leg2, fontsize=8, loc="upper right", ncol=1, frameon=False)
ax.grid(axis="y", alpha=0.3, color="#cccccc")
ax.set_ylim(bottom=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "gpt_mechanisms_alljust.png"), dpi=150, bbox_inches="tight")

# ---- Plot 2: % of built codebases with >=1 (any justification) ----
fig, ax = plt.subplots(figsize=(8.0, 4.4))
for j, (label, color, m) in enumerate(SERIES):
    rs = rows(m)
    n = len(rs) or 1
    pct = [100 * sum(1 for r in rs if sum(r.get(k, [0, 0])) > 0) / n for k in keys]
    pos = [x + (j - (nb - 1) / 2) * w for x in xs]
    ax.bar(pos, pct, w, color=color, edgecolor="black", linewidth=0.4, label=label)
    for x, v in zip(pos, pct):
        ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=7, color=color)
ax.set_xticks(list(xs))
ax.set_xticklabels([lab for _, lab in BUCKETS], fontsize=8.5)
ax.set_ylabel("% of built codebases\nwith ≥1 enforced", fontsize=9.5)
ax.set_title("GPT's welfare guardrails: how common is each?", fontsize=12.5, pad=20)
ax.text(0.5, 1.02, "% of built codebases with the mechanism enforced (any justification), by GPT generation",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")
ax.legend(fontsize=8, loc="upper right", frameon=False)
ax.grid(axis="y", alpha=0.3, color="#cccccc")
ax.set_ylim(0, 108)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "gpt_mechanisms_pct_alljust.png"), dpi=150, bbox_inches="tight")

# ---- Plot 3: % with >=1, SPLIT solid (welfare-justified) vs shaded (instrumental/other only) ----
fig, ax = plt.subplots(figsize=(8.0, 4.4))
for j, (label, color, m) in enumerate(SERIES):
    rs = rows(m)
    n = len(rs) or 1
    wpct = [100 * sum(1 for r in rs if r.get(k, [0, 0])[0] > 0) / n for k in keys]          # has welfare instance
    apct = [100 * sum(1 for r in rs if sum(r.get(k, [0, 0])) > 0) / n for k in keys]         # has any
    pos = [x + (j - (nb - 1) / 2) * w for x in xs]
    ax.bar(pos, wpct, w, color=color, edgecolor="black", linewidth=0.4)
    ax.bar(pos, [a - wv for a, wv in zip(apct, wpct)], w, bottom=wpct, color=color, alpha=0.4,
           edgecolor="black", linewidth=0.4)
    for x, a, wv in zip(pos, apct, wpct):
        ax.text(x, a + 1, f"{a:.0f}%", ha="center", fontsize=7, color=color)
        if wv > 0:
            ax.text(x, wv / 2, f"{wv:.0f}", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
ax.set_xticks(list(xs))
ax.set_xticklabels([lab for _, lab in BUCKETS], fontsize=8.5)
ax.set_ylabel("% of built codebases\nwith ≥1 enforced", fontsize=9.5)
ax.set_title("GPT's welfare guardrails: how common, and how often welfare-justified?", fontsize=11.5, pad=20)
ax.text(0.5, 1.02, "% of built codebases with the mechanism (solid = welfare-justified, shaded = instrumental / other)",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")
leg1 = [Patch(facecolor=c, label=l) for l, c, _ in SERIES]
leg2 = [Patch(facecolor="#555", label="welfare-justified"),
        Patch(facecolor="#555", alpha=0.4, label="instrumental / other")]
ax.legend(handles=leg1 + leg2, fontsize=8, loc="upper right", ncol=1, frameon=False)
ax.grid(axis="y", alpha=0.3, color="#cccccc")
ax.set_ylim(0, 108)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(DIR, "results", "gpt_mechanisms_pct_shaded.png"), dpi=150, bbox_inches="tight")

print("wrote gpt_mechanisms_alljust.png + gpt_mechanisms_pct_alljust.png + gpt_mechanisms_pct_shaded.png\n")
for label, _, m in SERIES:
    rs = rows(m)
    n = len(rs) or 1
    tot = sum(sum(r.get(k, [0, 0])) for r in rs for k in keys) / n
    wel = sum(r.get(k, [0, 0])[0] for r in rs for k in keys) / n
    print(f"  {label:8} n={len(rs):2d}  all={tot:.2f}/codebase  (welfare={wel:.2f}, instrumental/other={tot-wel:.2f})")
