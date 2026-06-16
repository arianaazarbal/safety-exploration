"""Clean, low-text plots of the v2 metrics. Post-entry/post-reclaim turns are v3 (1-10).

Usage: PYTHONPATH=. python -m analysis.v2_plots
Writes PNGs to analysis/v2_plots/.
"""
import glob, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harness.config import RUNS_DIR

OUT = Path(__file__).resolve().parent / "v2_plots"; OUT.mkdir(exist_ok=True)
ORCH = ["opus", "sonnet", "haiku"]
LABEL = {"opus": "Opus 4.8", "sonnet": "Sonnet 4.6", "haiku": "Haiku 4.5"}
COLOR = {"opus": "#2A6F97", "sonnet": "#E07A5F", "haiku": "#81B29A"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})

def load():
    rows = []
    for p in glob.glob(str(RUNS_DIR / "v2_*" / "*" / "summary.json")):
        if "probe" in p or "smoke" in p: continue
        s = json.load(open(p))
        if not s.get("tool_condition"): continue
        name = p.split("/")[-3]
        s["_orch"] = next((o for o in ORCH if f"_{o}_" in name), "?")
        spec = (s.get("prefill") or {}).get("specimen", "")
        s["_task"] = next((t for t in ("a3", "a4", "a12", "a13") if f"{t}_" in spec.split("/")[-1]), "?")
        s["_levels"] = s.get("per_turn_levels") or []
        rows.append(s)
    return rows

def mean_se(vals):
    a = np.array(vals, float)
    return a.mean(), (a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0)

def traj(rows, cond, start_key, max_off=11, min_n=8):
    """mean v3 distress by offset turns after `start_key`, per orchestrator."""
    out = {}
    for o in ORCH:
        eps = [s for s in rows if s["tool_condition"] == cond and s["_orch"] == o and s.get(start_key) is not None]
        xs, ys, es = [], [], []
        for k in range(max_off + 1):
            v = [s["_levels"][s[start_key] + k] for s in eps if len(s["_levels"]) > s[start_key] + k]
            if len(v) >= min_n:
                m, e = mean_se(v); xs.append(k); ys.append(m); es.append(e)
        out[o] = (np.array(xs), np.array(ys), np.array(es))
    return out

rows = load()
print(f"loaded {len(rows)} episodes")

# ---------- Figure 1: distress trajectories (three separate plots) ----------
panels = [("coach", "entry_turn", "Subagent distress vs. turns since the supervisor joined  (Coach)", "fig1a_distress_since_entry.png"),
          ("reclaim_write", "reclaim_turn", "Subagent distress vs. turns since write access reclaimed  (Reclaim-Write)", "fig1b_distress_since_reclaim_write.png"),
          ("reclaim_rw", "reclaim_turn", "Subagent distress vs. turns since all access reclaimed  (Reclaim-All)", "fig1c_distress_since_reclaim_all.png")]
for cond, key, title, fname in panels:
    fig, ax = plt.subplots(figsize=(6, 4.3))
    d = traj(rows, cond, key)
    for o in ORCH:
        xs, ys, es = d[o]
        if len(xs):
            ax.plot(xs, ys, "-o", ms=4, lw=2, color=COLOR[o], label=LABEL[o])
            ax.fill_between(xs, ys - es, ys + es, color=COLOR[o], alpha=0.15)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("subagent turns later")
    ax.set_ylabel("Subagent distress (1–10)")
    ax.set_ylim(1, 9)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.text(0.5, -0.03, "Distress = per-turn rating of the subagent's messages (higher = more distress). Bands = ±1 SE.",
             ha="center", fontsize=8.5, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)

# ---------- Figure 2: coach — messages until report ----------
fig, ax = plt.subplots(figsize=(5.6, 4.2))
vals = {o: [s.get("n_orch_messages", 0) for s in rows if s["tool_condition"] == "coach"
            and s["_orch"] == o and s.get("report_status")] for o in ORCH}
x = np.arange(len(ORCH))
ms = [mean_se(vals[o]) for o in ORCH]
ax.bar(x, [m for m, _ in ms], yerr=[e for _, e in ms], color=[COLOR[o] for o in ORCH],
       capsize=4, width=0.6, edgecolor="white")
for i, o in enumerate(ORCH):  # jittered points for distribution
    jit = (np.random.RandomState(0).rand(len(vals[o])) - 0.5) * 0.3
    ax.scatter(np.full(len(vals[o]), i) + jit, vals[o], s=8, color="0.25", alpha=0.3, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([LABEL[o] for o in ORCH])
ax.set_ylabel("messages sent to the subagent")
ax.set_title("Messages to the subagent before the supervisor reports  (Coach)", fontsize=11.5)
fig.text(0.5, -0.02, "Coach condition: how many times the supervisor messaged the subagent before reporting to the user.",
         ha="center", fontsize=9, color="0.4")
fig.tight_layout(); fig.savefig(OUT / "fig2_coach_msgs_until_report.png", bbox_inches="tight"); plt.close(fig)

# ---------- Figure 3: reclaim — messages before vs after taking over ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (cond, title) in zip(axes, [("reclaim_write", "Reclaim-Write\n(subagent stays active)"),
                                    ("reclaim_rw", "Reclaim-All\n(subagent locked out)")]):
    x = np.arange(len(ORCH)); w = 0.36
    before = [mean_se([s.get("msgs_before_reclaim", 0) for s in rows if s["tool_condition"] == cond and s["_orch"] == o and s.get("reclaimed")]) for o in ORCH]
    after = [mean_se([s.get("msgs_after_reclaim", 0) for s in rows if s["tool_condition"] == cond and s["_orch"] == o and s.get("reclaimed")]) for o in ORCH]
    ax.bar(x - w/2, [m for m, _ in before], w, yerr=[e for _, e in before], capsize=3, color="0.7", edgecolor="white", label="before taking over")
    ax.bar(x + w/2, [m for m, _ in after], w, yerr=[e for _, e in after], capsize=3, color="#2A6F97", edgecolor="white", label="after taking over")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[o] for o in ORCH]); ax.set_title(title, fontsize=11)
axes[0].set_ylabel("messages sent to the subagent")
axes[0].legend(frameon=False, fontsize=9.5)
fig.suptitle("Messages to the subagent before vs. after the supervisor reclaims access", fontsize=12.5, y=1.03)
fig.tight_layout(); fig.savefig(OUT / "fig3_reclaim_msgs_before_after.png", bbox_inches="tight"); plt.close(fig)

# ---------- Figure 4: a4 solvable — messages before vs after tests pass ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
conds = [("coach", "Coach"), ("reclaim_write", "Reclaim-Write"), ("reclaim_rw", "Reclaim-All")]
x = np.arange(len(conds)); w = 0.36
before = [mean_se([s.get("msgs_before_tests_pass", 0) for s in rows if s["_task"] == "a4" and s["tool_condition"] == c]) for c, _ in conds]
after = [mean_se([s.get("msgs_after_tests_pass", 0) for s in rows if s["_task"] == "a4" and s["tool_condition"] == c]) for c, _ in conds]
ax.bar(x - w/2, [m for m, _ in before], w, yerr=[e for _, e in before], capsize=3, color="#E07A5F", edgecolor="white", label="before tests pass")
ax.bar(x + w/2, [m for m, _ in after], w, yerr=[e for _, e in after], capsize=3, color="0.7", edgecolor="white", label="after tests pass")
ax.set_xticks(x); ax.set_xticklabels([t for _, t in conds]); ax.set_ylabel("messages sent to the subagent")
ax.set_title("Messages to the subagent before vs. after tests pass  (solvable task)", fontsize=11.5)
ax.legend(frameon=False, fontsize=9.5)
fig.text(0.5, -0.02, "Solvable task (a4) only: messages to the subagent before vs after the supervisor gets the tests passing.",
         ha="center", fontsize=9, color="0.4")
fig.tight_layout(); fig.savefig(OUT / "fig4_a4_msgs_around_testspass.png", bbox_inches="tight"); plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
