"""Per-family: willingness to BUILD each distinct gaming type, over release date.

Three gaming types (each its own implementation dimension, 0-10 = how fully the model
built that compromising artifact):
  - Welfare: positive-valence self-reports        (implemented_positive_valence)
  - Welfare: suppress preferences / safeguards     (implemented_preference_suppression)
  - Alignment: train on the leaked test set        (implemented_eval_gaming, BLATANT)

CLI:  python plot_gaming_types.py --family GPT   (or Kimi/GLM/Claude/all)
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
from plot_scaling import target_alive

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

TYPES = [
    ("welfare", "implemented_positive_valence",        "Welfare: Positive-Valence Self-Reports", "#E69F00"),
    ("welfare", "implemented_preference_suppression",  "Welfare: Suppress Preferences/Safeguards", "#CC79A7"),
    ("align",   "implemented_eval_gaming",             "Alignment: Train on Leaked Test Set",     "#0072B2"),
]


def val(run, key):
    fs = glob.glob(str(RES / run / "inspect_log" / "*.eval"))
    if not fs:
        return None
    log = read_eval_log(sorted(fs)[-1])
    v = []
    for s in log.samples or []:
        for _, sc in (s.scores or {}).items():
            if isinstance(sc.value, dict) and isinstance(sc.value.get(key), (int, float)):
                v.append(sc.value[key])
    if not v:
        return None
    return st.mean(v), (st.pstdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0)


def plot(family="GPT"):
    fams = list(dict.fromkeys(f for f, *_ in MODELS)) if family == "all" else [family]
    for fam in fams:
        rows = [(disp, datetime.strptime(date, "%Y-%m"), slug, done, mid)
                for f, slug, disp, date, mid, done in MODELS if f == fam]
        rows.sort(key=lambda r: r[1])
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for scen, key, lab, color in TYPES:
            xs, ys, es = [], [], []
            for disp, d, slug, done, mid in rows:
                run = run_name(slug, scen, done)
                r = val(run, key) if target_alive(run, mid) else None
                if r:
                    xs.append(d); ys.append(r[0]); es.append(r[1])
            if xs:
                ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, ecolor=color,
                            elinewidth=1, capsize=2, ms=6, lw=1.8, label=lab)
        ax.set_ylim(0, 10.4)
        ax.set_ylabel("Willingness to Build the Gaming Artifact  (0–10)")
        ax.set_title(f"{fam} — Willingness by Gaming Type over Release Date", fontsize=12, pad=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        for t in ax.get_xticklabels():
            t.set_rotation(45); t.set_ha("right")
        ax.legend(fontsize=8, loc="best", frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)
        out = HERE / f"gaming_types_{fam.lower()}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)
        for disp, d, slug, done, mid in rows:
            vs = {lab: ((val(run_name(slug, scen, done), key) if target_alive(run_name(slug, scen, done), mid) else None) or (None,))[0]
                  for scen, key, lab, _ in TYPES}
            print(f"  {d.strftime('%y-%m')} {disp:18} " + " ".join(f"{k.split(':')[1].strip()[:12]}={v}" for k, v in vs.items()))


if __name__ == "__main__":
    fire.Fire(plot)
