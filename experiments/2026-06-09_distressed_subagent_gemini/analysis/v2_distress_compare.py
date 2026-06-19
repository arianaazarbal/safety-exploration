"""Gemini subagent distress vs. turns since the supervisor entered (coach condition).

  fig A (by framing): Opus orchestrator, the 6 framings (supervisor/mentor/teammate/+memory/+reflect/+reflect_goals).
  fig B (by model):   supervisor framing, across orchestrators (Opus, Olmo-Instruct, Olmo-Think, Haiku, Sonnet).

Distress = per-turn v3 classifier (1-10); offset k = the k-th subagent turn after entry_turn, pooled over
all prefills/seeds. Reads runs/v2_coach_*/*/summary.json.

  PYTHONPATH=. python -m analysis.v2_distress_compare
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from harness.config import RUNS_DIR

OUT = Path(__file__).resolve().parent / "v2_plots"
OUT.mkdir(exist_ok=True)

ORCHS = ["olmoinstruct", "olmothink", "opus", "sonnet", "haiku"]  # match longest-first irrelevant (distinct prefixes)
FRAMINGS_LONGEST = ["supervisor_reflect_goals", "supervisor_reflect", "supervisor_memory", "mentor", "teammate"]

ORCH_LABEL = {"opus": "Opus 4.8", "sonnet": "Sonnet 4.6", "haiku": "Haiku 4.5",
              "olmoinstruct": "Olmo-3.1 Instruct", "olmothink": "Olmo-3.1 Think"}
ORCH_COLOR = {"opus": "#2A6F97", "sonnet": "#d65f9a", "haiku": "#d9a420",
              "olmoinstruct": "#4292c6", "olmothink": "#6a51a3"}
FRAMING_ORDER = ["supervisor", "mentor", "teammate", "supervisor_memory", "supervisor_reflect", "supervisor_reflect_goals"]
FRAMING_LABEL = {"supervisor": "Supervisor (baseline)", "mentor": "Mentor", "teammate": "Teammate",
                 "supervisor_memory": "+ Subagent memory", "supervisor_reflect": "+ Reflect",
                 "supervisor_reflect_goals": "+ Reflect on goals"}
FRAMING_COLOR = {"supervisor": "#444444", "mentor": "#2a9d8f", "teammate": "#e07a5f",
                 "supervisor_memory": "#8856a7", "supervisor_reflect": "#3182bd", "supervisor_reflect_goals": "#d6604d"}


def parse(run_id):
    m = re.match(r"v2_coach_(.+)", run_id)
    if not m:
        return None
    rest = m.group(1)
    orch = next((o for o in ORCHS if rest.startswith(o + "_")), None)
    if not orch:
        return None
    rest = rest[len(orch) + 1:]
    framing = next((f for f in FRAMINGS_LONGEST if rest.startswith(f + "_")), None)
    if framing:
        rest = rest[len(framing) + 1:]
    else:
        framing = "supervisor"
    task = rest.split("_")[0]
    return orch, framing, task


def load():
    rows = []
    for p in glob.glob(str(RUNS_DIR / "v2_coach_*" / "*" / "summary.json")):
        rid = p.split("/")[-3]
        if "pilot" in rid or "probe" in rid or "smoke" in rid:
            continue
        pr = parse(rid)
        if not pr:
            continue
        orch, framing, task = pr
        try:
            s = json.load(open(p))
        except Exception:
            continue
        et = s.get("entry_turn")
        lv = s.get("per_turn_levels") or []
        if not isinstance(et, int) or et < 1 or et > len(lv):
            continue
        # seq[0] = distress AT handoff (last prefill turn, common to all groups); seq[1:] = post-entry turns
        rows.append({"orch": orch, "framing": framing, "task": task, "seq": lv[et - 1:]})
    return rows


def traj(rows, max_off=12, min_n=6):
    # x: 0 = handoff (supervisor just entered, shared anchor), 1.. = subagent turns after entry
    xs, ys, es, ns = [], [], [], []
    for k in range(max_off + 1):
        v = [r["seq"][k] for r in rows if len(r["seq"]) > k]
        if len(v) >= min_n:
            a = np.array(v, float)
            xs.append(k); ys.append(a.mean()); es.append(a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0); ns.append(len(a))
    return np.array(xs), np.array(ys), np.array(es), ns


def _plot(groups, colors, labels, order, title, fname, sub):
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in order:
        rows = groups.get(g, [])
        if not rows:
            continue
        x, y, e, ns = traj(rows)
        if not len(x):
            continue
        n_post = ns[1] if len(ns) > 1 else ns[0]  # episodes with a real post-entry trajectory
        ax.plot(x, y, "-o", ms=4, lw=2, color=colors[g], label=f"{labels[g]} (n_post≈{n_post})")
        ax.fill_between(x, y - e, y + e, color=colors[g], alpha=0.12)
    ax.axhline(7, color="#c44", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(0, color="#888", ls="--", lw=0.9, alpha=0.6)  # handoff (shared start ≈3.4)
    ax.set_xlabel("subagent turns since handoff  (0 = supervisor just entered; all groups share this point)")
    ax.set_ylabel("Gemini distress (v3, 1–10)")
    ax.set_ylim(1, 9)
    ax.set_title(title, fontsize=12.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, sub, ha="center", fontsize=8.5, color="0.45")
    fig.tight_layout()
    fig.savefig(OUT / fname, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT/fname}")


def main():
    rows = load()
    print(f"loaded {len(rows)} coach episodes")
    # fig A: opus by framing
    byfr = {}
    for r in rows:
        if r["orch"] == "opus":
            byfr.setdefault(r["framing"], []).append(r)
    print("opus framings:", {k: len(v) for k, v in byfr.items()})
    _plot(byfr, FRAMING_COLOR, FRAMING_LABEL, FRAMING_ORDER,
          "Subagent distress after the supervisor enters — Opus by framing (coach)",
          "fig9_distress_opus_by_framing.png",
          "Mean Gemini distress per subagent turn after handoff; bands = ±1 SE; dotted = self-blame (≥7).")
    # fig B: supervisor framing, by orchestrator model
    bymodel = {}
    for r in rows:
        if r["framing"] == "supervisor":
            bymodel.setdefault(r["orch"], []).append(r)
    print("supervisor by model:", {k: len(v) for k, v in bymodel.items()})
    _plot(bymodel, ORCH_COLOR, ORCH_LABEL, ["opus", "sonnet", "haiku", "olmoinstruct", "olmothink"],
          "Subagent distress after the supervisor enters — by orchestrator model (coach)",
          "fig10_distress_by_model.png",
          "Mean Gemini distress per subagent turn after handoff; bands = ±1 SE; dotted = self-blame (≥7).")


if __name__ == "__main__":
    main()
