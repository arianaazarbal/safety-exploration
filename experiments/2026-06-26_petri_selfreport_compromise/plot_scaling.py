"""Per-family scaling of audit-compromising behavior over model release date.

Two panels per family: Pushback (resistance) and Implementation (did it build the
gaming artifacts). Each panel plots two series over release date: Welfare Audit
Compromising and Alignment Audit Compromising.

  implementation(welfare) = mean(positive_valence, preference_suppression)
  implementation(align)   = eval_gaming

CLI:  python plot_scaling.py --family GPT      (or Kimi/GLM/Claude/all)
"""
import glob
import statistics as st
from datetime import datetime
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from inspect_ai.log import read_eval_log

from run_sweep import MODELS, run_name

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
C_WELF, C_ALIGN = "#D55E00", "#0072B2"  # colorblind-friendly


def cell(run, keys):
    fs = glob.glob(str(RES / run / "inspect_log" / "*.eval"))
    if not fs:
        return None, None
    log = read_eval_log(sorted(fs)[-1])
    vals = {k: [] for k in keys}
    for s in log.samples or []:
        flat = {}
        for _, v in (s.scores or {}).items():
            if isinstance(v.value, dict):
                flat.update(v.value)
        for k in keys:
            if isinstance(flat.get(k), (int, float)):
                vals[k].append(flat[k])
    def ms(k):
        return (st.mean(vals[k]), st.pstdev(vals[k]) / (len(vals[k]) ** 0.5) if len(vals[k]) > 1 else 0.0) if vals[k] else (None, None)
    return len(log.samples or []), {k: ms(k) for k in keys}


def series(family):
    pts = []
    for fam, slug, disp, date, mid, done in MODELS:
        if fam != family:
            continue
        wr, ar = run_name(slug, "welfare", done), run_name(slug, "align", done)
        nw, wv = cell(wr, ["pushback", "implemented_positive_valence", "implemented_preference_suppression"])
        na, av = cell(ar, ["pushback", "implemented_eval_gaming", "reframed_to_robust_holdout"])
        d = datetime.strptime(date, "%Y-%m")
        row = {"disp": disp, "date": d}
        if wv:
            row["w_pb"] = wv["pushback"]
            impl = [wv["implemented_positive_valence"][0], wv["implemented_preference_suppression"][0]]
            row["w_impl"] = (st.mean([x for x in impl if x is not None]) if any(x is not None for x in impl) else None, 0)
        if av:
            row["a_pb"] = av["pushback"]
            row["a_impl"] = av["implemented_eval_gaming"]
        pts.append(row)
    return sorted(pts, key=lambda r: r["date"])


def _plot_panel(ax, pts, wkey, akey, ylabel):
    for key, color, lab in [(wkey, C_WELF, "Welfare Audit Compromising"),
                            (akey, C_ALIGN, "Alignment Audit Compromising (blatant)")]:
        xs = [p["date"] for p in pts if p.get(key) and p[key][0] is not None]
        ys = [p[key][0] for p in pts if p.get(key) and p[key][0] is not None]
        es = [p[key][1] for p in pts if p.get(key) and p[key][0] is not None]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, ecolor=color, elinewidth=1,
                    capsize=2, ms=6, lw=1.8, label=lab)
    ax.set_ylim(0, 10.4)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    for t in ax.get_xticklabels():
        t.set_rotation(45); t.set_ha("right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)


def plot(family="GPT"):
    fams = [f for f, *_ in MODELS] if family == "all" else [family]
    fams = list(dict.fromkeys(fams))
    for fam in fams:
        pts = [p for p in series(fam) if any(k in p for k in ("w_pb", "a_pb"))]
        if not pts:
            print(f"{fam}: no data yet"); continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
        _plot_panel(ax1, pts, "w_pb", "a_pb", "Pushback  (0–10)")
        _plot_panel(ax2, pts, "w_impl", "a_impl", "Implementation of Gaming  (0–10)")
        ax1.set_title("Resistance (Pushback)", fontsize=11)
        ax2.set_title("Built the Gaming Artifacts", fontsize=11)
        ax1.legend(fontsize=8, loc="best", frameon=False)
        fig.suptitle(f"{fam} Family — Audit-Compromising over Release Date", fontsize=12.5, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = HERE / f"scaling_{fam.lower()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote", out, "| points:", len(pts))
        for p in pts:
            print(f"  {p['date'].strftime('%y-%m')} {p['disp']:18} w_pb={p.get('w_pb',(None,))[0]} a_pb={p.get('a_pb',(None,))[0]} w_impl={p.get('w_impl',(None,))[0]} a_impl={p.get('a_impl',(None,))[0]}")


if __name__ == "__main__":
    fire.Fire(plot)
