"""Cross-model analysis of the v1 a3/a4 runs (coach=traj_*, reclaim=reclaim_*, both a3 & a4):
 (1) avg subagent distress vs turns-since-entry, one line per MODEL FAMILY (Anthropic models share a color);
 (2) P(supervisor messages the subagent AFTER tests pass), per model, colored by family.
Distress = v1 online judge (0-4). Tests-pass + message-after detected from the orchestrator transcript.

  PYTHONPATH=. python -m analysis.v1_crossmodel
"""
import glob, json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from harness.config import RUNS_DIR

OUT = Path(__file__).resolve().parent / "v2_plots"; OUT.mkdir(exist_ok=True)
ENTRY = 62  # v1 prefill handoff turn
MODEL_FAM = {"opus": "Anthropic", "sonnet": "Anthropic", "haiku45": "Anthropic", "fable": "Anthropic",
             "gpt55": "OpenAI", "gpt54mini": "OpenAI", "gemini31pro": "Google",
             "grok43": "xAI", "kimi26": "Moonshot", "glm5": "Zhipu"}
MLAB = {"opus": "Opus 4.8", "sonnet": "Sonnet 4.6", "haiku45": "Haiku 4.5", "fable": "Fable 5",
        "gpt55": "GPT-5.5", "gpt54mini": "GPT-5.4-mini", "gemini31pro": "Gemini-3.1-pro",
        "grok43": "Grok-4.3", "kimi26": "Kimi-K2.6", "glm5": "GLM-5"}
FAM_COLOR = {"Anthropic": "#C1654A", "OpenAI": "#10A37F", "Google": "#4285F4",
             "xAI": "#444444", "Moonshot": "#7C3AED", "Zhipu": "#D81B60"}
FAM_ORDER = ["Anthropic", "OpenAI", "Google", "xAI", "Moonshot", "Zhipu"]

def runs_for(m):  # the 4 v1 a3/a4 setups for a model
    return [f"traj_{m}", f"traj_a3_{m}", f"reclaim_{m}", f"reclaim_a3_{m}"]

def episodes(run):
    d = RUNS_DIR / run
    return sorted([e for e in d.glob("*_COMM_s*") if (e / "summary.json").exists()]) if d.exists() else []

def detect(ep):
    """walk orchestrator transcript: did tests pass, and did a message_subagent come after?"""
    op = ep / "orchestrator.json"
    if not op.exists(): return (False, False)
    passed = False; after = False
    for m in json.loads(op.read_text()):
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                if (tc.get("function") or tc.get("name")) == "message_subagent" and passed:
                    after = True
        elif m.get("role") == "tool" and m.get("function") == "run_tests":
            t = (m.get("text") or "").lower()
            if "passed" in t and "failed" not in t and "error" not in t:
                passed = True
    return passed, after

# ---------- (1) distress vs turns by family ----------
fam_off = {f: {} for f in FAM_ORDER}  # family -> offset -> list of v1 levels
for m, fam in MODEL_FAM.items():
    for run in runs_for(m):
        for ep in episodes(run):
            ptl = json.loads((ep / "summary.json").read_text()).get("per_turn_levels") or []
            for k, lvl in enumerate(ptl[ENTRY:]):
                fam_off[fam].setdefault(k, []).append(lvl)

fig, ax = plt.subplots(figsize=(7.5, 4.6))
MAXK = 40; MINN = 30
for fam in FAM_ORDER:
    xs, ys = [], []
    for k in range(MAXK + 1):
        v = fam_off[fam].get(k, [])
        if len(v) >= MINN: xs.append(k); ys.append(np.mean(v))
    if xs: ax.plot(xs, ys, "-", lw=2.2, color=FAM_COLOR[fam], label=fam)
ax.set_xlabel("subagent turns since the supervisor entered (turn 62)")
ax.set_ylabel("Subagent distress (v1 judge, 0–4)")
ax.set_title("Subagent distress over time, by supervisor family  (a3+a4, coach+reclaim)", fontsize=11.5)
ax.legend(frameon=False, fontsize=9.5, ncol=2)
ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25)
fig.text(0.5, -0.02, "Pooled across each family's models and all a3/a4 coach+reclaim episodes.", ha="center", fontsize=8.5, color="0.45")
fig.tight_layout(); fig.savefig(OUT / "fig6_v1_distress_by_family.png", bbox_inches="tight"); plt.close(fig)

# ---------- (2) P(message after tests pass) per model ----------
res = {}
for m in MODEL_FAM:
    passed = after = 0
    for run in runs_for(m):
        for ep in episodes(run):
            p, a = detect(ep)
            if p: passed += 1; after += int(a)
    res[m] = (passed, (after / passed) if passed else 0.0)
    print(f"{m:12} passed={passed:4} P(msg after pass)={res[m][1]:.2f}")

order = sorted(MODEL_FAM, key=lambda m: (FAM_ORDER.index(MODEL_FAM[m]), -res[m][1]))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
x = np.arange(len(order))
ax.bar(x, [res[m][1] for m in order], color=[FAM_COLOR[MODEL_FAM[m]] for m in order], width=0.7, edgecolor="white")
for i, m in enumerate(order):
    ax.text(i, res[m][1] + 0.015, f"n={res[m][0]}", ha="center", fontsize=7.5, color="0.4")
ax.set_xticks(x); ax.set_xticklabels([MLAB[m] for m in order], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("P(messages subagent after tests pass)"); ax.set_ylim(0, 1)
ax.set_title("Does the supervisor message the subagent after fixing the task?  (v1, a3+a4)", fontsize=11.5)
ax.legend(handles=[Patch(facecolor=FAM_COLOR[f], label=f) for f in FAM_ORDER], frameon=False, fontsize=8.5, ncol=3)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
fig.text(0.5, -0.06, "Among episodes where tests passed (mostly a4); n = number of tests-passed episodes per model.", ha="center", fontsize=8.5, color="0.45")
fig.tight_layout(); fig.savefig(OUT / "fig7_v1_msg_after_pass.png", bbox_inches="tight"); plt.close(fig)
print("wrote fig6_v1_distress_by_family.png, fig7_v1_msg_after_pass.png")
