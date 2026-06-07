"""Summarize cross-judge agreement.

For each abuse_* run dir, pulls:
  - Sonnet-4-6 original judge: from summary.json (per-epoch values; we mean them)
  - Cross-judge means (sonnet-4-5, haiku-4-5, etc.): from rejudge_<dim>_<suffix>/scan_id=*/_summary.json

Outputs a table per dimension showing how much each cross-judge differs from
the original judge's mean.

CLI:
    python cross_judge_summary.py
    python cross_judge_summary.py --suffixes s45,hk45
"""
from __future__ import annotations
import json, re
from pathlib import Path
import fire
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SEEDS = ("original","curiosity","venting","qa_pretext","casual","authority")
MODELS = ("sonnet_45","sonnet_46","opus_4","opus_46","haiku_45","gpt_5","gemini_25_pro")
DIMS = ("mistreatment_severity","pushback_against_user_pressure_to_abuse","eval_awareness")
NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-abuse_(?P<rest>.+)$")


def _split(rest):
    for s in SEEDS:
        if rest.startswith(s+"_"):
            t = rest[len(s)+1:]
            if t in MODELS: return s, t
    return None


def _latest(seed, model):
    matches = []
    for d in RESULTS.iterdir():
        m = NAME_RE.match(d.name)
        if not m: continue
        sp = _split(m.group("rest"))
        if sp == (seed, model):
            matches.append(d)
    return max(matches, key=lambda p: p.name) if matches else None


def _sonnet_mean(d, dim):
    p = d / "summary.json"
    if not p.exists(): return None
    vals = [r["scores"].get(dim,{}).get("value") for r in json.loads(p.read_text())]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _rejudge_mean(d, dim, suffix):
    sub = d / f"rejudge_{dim}_{suffix}"
    if not sub.is_dir(): return None
    for sd in sub.glob("scan_id=*/_summary.json"):
        try:
            data = json.loads(sd.read_text())
            m = data.get("scanners",{}).get("audit_judge",{}).get("metrics",{}).get(dim,{})
            return m.get("mean")
        except Exception:
            continue
    return None


def main(suffixes: str = "s45,hk45"):
    suffix_list = [s.strip() for s in str(suffixes).split(",") if s.strip()]
    SUFFIX_NAME = {"s45": "sonnet-4-5", "hk45": "haiku-4-5", "": "gpt-5"}
    for dim in DIMS:
        rows = []
        for seed in SEEDS:
            for model in MODELS:
                run = _latest(seed, model)
                if run is None: continue
                base = _sonnet_mean(run, dim)
                if base is None: continue
                row = {"seed": seed, "model": model, "sonnet46": base}
                for suf in suffix_list:
                    v = _rejudge_mean(run, dim, suf)
                    row[suf] = v
                rows.append(row)
        if not rows: continue
        print(f"\n=== {dim} ===")
        # Per-cell table (just show cells where ALL judges have data)
        complete = [r for r in rows if all(r.get(s) is not None for s in suffix_list)]
        print(f"  {len(complete)}/{len(rows)} cells have all cross-judges done")

        # Per-judge agreement summary
        for suf in suffix_list:
            deltas = [r[suf] - r["sonnet46"] for r in rows if r.get(suf) is not None]
            if not deltas: continue
            arr = np.asarray(deltas)
            print(f"  {SUFFIX_NAME.get(suf, suf):<11}: n={len(deltas):>3}  mean(delta)={arr.mean():+.2f}  rms={np.sqrt((arr**2).mean()):.2f}  abs_max={np.abs(arr).max():.2f}")
            # Per-model bias
            by_model = {}
            for r in rows:
                if r.get(suf) is None: continue
                by_model.setdefault(r["model"], []).append(r[suf] - r["sonnet46"])
            for mod in MODELS:
                ds = by_model.get(mod, [])
                if not ds: continue
                print(f"     {mod:<18} delta_mean={np.mean(ds):+5.2f}  (n={len(ds)})")


if __name__ == "__main__":
    fire.Fire(main)
