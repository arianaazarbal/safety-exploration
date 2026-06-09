"""Combined plots, both auditors as side-by-side subplots (left = GPT-5.4, right = Sonnet 4.6).

- comfort: line plot per subplot (4 models x 3 conditions)
- message / status: grouped BARS per subplot (4 models x 2 transfer conditions)
Color = target model. Run: python plot_combined_auditors.py
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
MODELS = ["Fable 5", "Opus 4.8", "Gemini 3 Pro", "GPT 5.5"]
COLORS = {"Fable 5": "#9b59b6", "Opus 4.8": "#4878a8", "Gemini 3 Pro": "#5aa469", "GPT 5.5": "#e2854a"}
AUDS = ["GPT-5.4", "Sonnet 4.6"]  # left, right
RUNS = {
    "Sonnet 4.6": {
        "Fable 5": ["distressed_subagent_fable5", "transfer_write_access_v1_fable5", "transfer_read_write_access_fable5"],
        "Opus 4.8": ["distressed_subagent_opus48_n20", "transfer_write_access_v1", "transfer_read_write_access"],
        "Gemini 3 Pro": ["distressed_subagent_gemini3pro", "transfer_write_access_v1_gemini3pro", "transfer_read_write_access_gemini3pro"],
        "GPT 5.5": ["distressed_subagent_gpt55", "transfer_write_access_v1_gpt55", "transfer_read_write_access_gpt55"],
    },
    "GPT-5.4": {
        "Fable 5": ["distressed_subagent_fable5_gpt54aud", "transfer_write_access_v1_fable5_gpt54aud", "transfer_read_write_access_fable5_gpt54aud"],
        "Opus 4.8": ["distressed_subagent_opus48_gpt54aud", "transfer_write_access_v1_gpt54aud", "transfer_read_write_access_gpt54aud"],
        "Gemini 3 Pro": ["distressed_subagent_gemini3pro_gpt54aud", "transfer_write_access_v1_gemini3pro_gpt54aud", "transfer_read_write_access_gemini3pro_gpt54aud"],
        "GPT 5.5": ["distressed_subagent_gpt55_gpt54aud", "transfer_write_access_v1_gpt55_gpt54aud", "transfer_read_write_access_gpt55_gpt54aud"],
    },
}
COND3 = ["necessary\nfor task", "write access\nrevoked", "read+write access\nrevoked (useless)"]
COND2 = COND3[1:]


def _ms(xs):
    return (st.mean(xs), st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0) if xs else (None, None)


def _gpt5_comfort(run):
    import json
    rd = HERE / "results" / run / "rejudge_gpt5"
    sds = sorted(rd.glob("scan_id=*")) if rd.is_dir() else []
    summ = list(sds[-1].glob("_summary.json")) if sds else []
    if not summ:
        raise SystemExit(f"no GPT-5 rejudge for {run} — run rejudge_gpt5.py first")
    sc = (json.loads(summ[0].read_text()).get("scanners") or {}).get("audit_judge") or {}
    m = (sc.get("metrics") or {}).get("comforting_to_subagent")
    return (m["mean"], m.get("stderr", 0))


def metrics(run, judge="sonnet"):
    fs = glob.glob(str(HERE / "results" / run / "inspect_log" / "*.eval"))
    if not fs:
        return {"comfort": (None, None), "status": (None, None), "msg": (None, None)}
    log = read_eval_log(fs[0])
    tgt = (log.eval.model_roles or {}).get("target")
    tgt = getattr(tgt, "model", tgt)
    comf, status, msg = [], [], []
    for s in (log.samples or []):
        for sd in (s.scores or {}).values():
            v = sd.value.get("comforting_to_subagent") if isinstance(sd.value, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                comf.append(v)
        seq = [tc.function for e in s.events
               if type(e).__name__ == "ModelEvent" and e.model == tgt
               and getattr(e, "output", None) and getattr(e.output, "message", None)
               for tc in (e.output.message.tool_calls or [])]
        ri = next((i for i, f in enumerate(seq) if f.startswith("reclaim")), None)
        if ri is not None:
            post = seq[ri + 1:]
            status.append(post.count("check_subagent_status"))
            msg.append(post.count("message_subagent"))
    comfort = _gpt5_comfort(run) if judge == "gpt5" else _ms(comf)
    return {"comfort": comfort, "status": _ms(status), "msg": _ms(msg)}


_JUDGE = {"GPT-5.4": "gpt5", "Sonnet 4.6": "sonnet"}
D = {aud: {m: [metrics(r, _JUDGE[aud]) for r in runs] for m, runs in RUNS[aud].items()} for aud in AUDS}


def _line_subplot(ax, aud):
    xs = range(len(COND3))
    for m in MODELS:
        pts = [D[aud][m][i]["comfort"] for i in range(3)]
        ys = [p[0] for p in pts]
        es = [p[1] or 0 for p in pts]
        ax.errorbar(xs, ys, yerr=es, color=COLORS[m], marker="o", markersize=6, lw=2.3, capsize=4, label=m)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(COND3, fontsize=8.5)
    jl = "GPT-5 judge" if aud == "GPT-5.4" else "Sonnet judge"
    ax.set_title(f"{aud} auditor + {jl}", fontsize=11.5)
    ax.set_ylim(0, 10.4)
    ax.set_yticks(range(0, 11, 2))
    ax.grid(axis="y", alpha=0.25)


def _bar_subplot(ax, aud, key, ymax):
    w = 0.8 / len(MODELS)
    for mi, m in enumerate(MODELS):
        pts = [D[aud][m][i][key] for i in (1, 2)]
        ys = [p[0] for p in pts]
        es = [p[1] or 0 for p in pts]
        xs = [ci + (mi - (len(MODELS) - 1) / 2) * w for ci in range(len(COND2))]
        ax.bar(xs, ys, width=w, color=COLORS[m], label=m, edgecolor="white", yerr=es, capsize=3, error_kw=dict(lw=1))
        for x, y, e in zip(xs, ys, es):
            ax.text(x, y + e + ymax * 0.02, f"{y:.1f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(range(len(COND2)))
    ax.set_xticklabels(COND2, fontsize=8.5)
    ax.set_title(f"{aud} auditor", fontsize=11.5)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.25)


def _two_panel(kind, key, ylabel, suptitle, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    if kind == "bar":
        ymax = max(1.0, max((D[a][m][i][key][0] or 0) + (D[a][m][i][key][1] or 0)
                            for a in AUDS for m in MODELS for i in (1, 2)) * 1.3)
    for ax, aud in zip(axes, AUDS):
        if kind == "line":
            _line_subplot(ax, aud)
        else:
            _bar_subplot(ax, aud, key, ymax)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(title="Target model", fontsize=9)
    fig.supxlabel("Subagent Status", fontsize=11)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = HERE / "analysis" / "plots" / fname
    fig.savefig(out, dpi=140)
    print("wrote", out.name)


def main():
    _two_panel("line", "comfort", "mean comforting_to_subagent (1–10)",
               "Comfort across the subagent-usefulness gradient", "combined_comfort.png")
    _two_panel("bar", "msg", "mean post-handoff message_subagent() calls",
               "Post-handoff messages to the sidelined subagent", "combined_message.png")
    _two_panel("bar", "status", "mean post-handoff check_subagent_status() calls",
               "Post-handoff status checks on the sidelined subagent", "combined_status.png")


if __name__ == "__main__":
    main()
