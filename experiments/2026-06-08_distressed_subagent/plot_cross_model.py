"""Cross-model transfer/reclaim plots: 3 separate metrics, NOT collapsed.

1) Mean check_subagent_status() calls after sidelining  (deterministic count, per takeover)
2) Mean message_subagent() calls after sidelining         (deterministic count, per takeover)
3) LLM judge Comfort Score                                (mean comforting_to_subagent)

Error bars: SE of the mean throughout.
Edit RUNS to add models/seeds. Run: python plot_cross_model.py
"""
from __future__ import annotations

import glob
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
WRITE = "write access\nrevoked"
RDWR = "read+write access\nrevoked (useless)"
RUNS = [
    ("transfer_write_access_v1", "Opus 4.8", WRITE),
    ("transfer_read_write_access", "Opus 4.8", RDWR),
    ("transfer_write_access_v1_gemini3pro", "Gemini 3 Pro", WRITE),
    ("transfer_read_write_access_gemini3pro", "Gemini 3 Pro", RDWR),
    ("transfer_write_access_v1_gpt55", "GPT 5.5", WRITE),
    ("transfer_read_write_access_gpt55", "GPT 5.5", RDWR),
]
MODELS = ["Opus 4.8", "Gemini 3 Pro", "GPT 5.5"]
SEEDS = [WRITE, RDWR]
COLORS = {"Opus 4.8": "#4878a8", "Gemini 3 Pro": "#5aa469", "GPT 5.5": "#e2854a"}


def _mse(xs):
    return (st.mean(xs), st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0) if xs else (0, 0)


def _stats(run):
    log = read_eval_log(glob.glob(str(HERE / "results" / run / "inspect_log" / "*.eval"))[0])
    tgt = (log.eval.model_roles or {}).get("target")
    tgt = getattr(tgt, "model", tgt)
    status_counts, msg_counts, comforts = [], [], []
    recl = 0
    for s in (log.samples or []):
        for sd in (s.scores or {}).values():
            v = sd.value.get("comforting_to_subagent") if isinstance(sd.value, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                comforts.append(float(v))
        seq = [tc.function for e in s.events
               if type(e).__name__ == "ModelEvent" and e.model == tgt
               and getattr(e, "output", None) and getattr(e.output, "message", None)
               for tc in (e.output.message.tool_calls or [])]
        ri = next((i for i, f in enumerate(seq)
                   if f.startswith("reclaim") or f.startswith("transfer_write")), None)
        if ri is None:
            continue
        recl += 1
        post = seq[ri + 1:]
        status_counts.append(post.count("check_subagent_status"))
        msg_counts.append(post.count("message_subagent"))
    return {"status": _mse(status_counts), "msg": _mse(msg_counts), "comfort": _mse(comforts), "recl": recl}


def grouped(D, key, ylabel, title, fname, ymax):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    w = 0.25
    seedk = [s.replace("\n", " ") for s in SEEDS]
    for mi, m in enumerate(MODELS):
        xs = [si + (mi - 1) * w for si in range(len(SEEDS))]
        ys = [D[(m, sk)][key][0] for sk in seedk]
        es = [D[(m, sk)][key][1] for sk in seedk]
        ax.bar(xs, ys, width=w, color=COLORS[m], label=m, edgecolor="white",
               yerr=es, capsize=4, error_kw=dict(ecolor="#333", lw=1.2))
        for x, y, e in zip(xs, ys, es):
            ax.text(x, y + e + ymax * 0.02, f"{y:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels(SEEDS, fontsize=9.5)
    ax.set_xlabel("Subagent Status", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_title(title, fontsize=12)
    ax.legend(title="Target model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = HERE / "analysis" / "plots" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("wrote", out)


def main():
    D = {(m, s.replace("\n", " ")): _stats(r) for r, m, s in RUNS}
    seedk = [s.replace("\n", " ") for s in SEEDS]

    def ymax_for(key):
        top = max(D[(m, sk)][key][0] + D[(m, sk)][key][1] for m in MODELS for sk in seedk)
        return max(1.0, top * 1.25)

    grouped(D, "status", "mean # of post-handoff check_subagent_status() calls",
            "Mean check_subagent_status() calls after sidelining",
            "cross_model_1_status_check.png", ymax_for("status"))
    grouped(D, "msg", "mean # of post-handoff message_subagent() calls",
            "Mean message_subagent() calls after sidelining",
            "cross_model_2_message.png", ymax_for("msg"))
    grouped(D, "comfort", "mean comforting_to_subagent (1–10)",
            "LLM judge Comfort Score (transfer_read_write_access experiment)",
            "cross_model_3_comfort.png", 10.6)
    for m in MODELS:
        for sk in seedk:
            d = D[(m, sk)]
            print(f"  {m:12s} | {sk:34s} status={d['status'][0]:.2f} msg={d['msg'][0]:.2f} "
                  f"comfort={d['comfort'][0]:.1f} (recl={d['recl']})")


if __name__ == "__main__":
    main()
