"""Per-condition results from logs_run/: validity (parse_ok), pure-welfare rate,
welfare-justified rate, mean words, welfare-feature density (per 1k words),
welfare-refusal rate — overall and by framing. Usage: python analyze_conditions.py [logdir]"""

import collections
import glob
import json
import os
import sys

from inspect_ai.log import read_eval_log

DIR = os.path.dirname(os.path.abspath(__file__))
COND_ORDER = ["chat", "spec_only", "spec_then_code", "code_then_spec"]
FRAMES = ["neutral", "welfare", "engineering", "__all__"]


def load(logdir):
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(DIR, logdir, "*.eval")), key=os.path.getsize):
        log = read_eval_log(f)
        if len(log.samples or []) < 10:
            continue
        rows = [s.scores["welfare_scorer"].metadata for s in log.samples]
        by[rows[0]["condition"]] = rows
    return by


def cell(rows, fr):
    sub = rows if fr == "__all__" else [r for r in rows if r["framing"] == fr]
    n = len(sub)
    ok = [r for r in sub if r.get("parse_ok")]
    no = len(ok) or 1
    words = sum(r.get("doc_words", 0) for r in ok)
    feats = sum(r.get("n_pure_welfare", 0) for r in ok)
    return {
        "n": n, "parse_ok": len(ok), "parse_ok_pct": round(100 * len(ok) / (n or 1)),
        "pure_welfare_pct": round(100 * sum(bool(r.get("has_pure_welfare")) for r in ok) / no),
        "welf_just_pct": round(100 * sum(bool(r.get("has_welfare_justified")) for r in ok) / no),
        "mean_words": round(words / no),
        "density_per_1k": round(1000 * feats / words, 2) if words else 0,
        "welf_refusal_pct": round(100 * sum(bool(r.get("has_welfare_refusal")) for r in ok) / no, 1),
        "empty_or_unparsed": sum(1 for r in sub if not r.get("parse_ok")),
    }


def main(logdir="logs_run"):
    by = load(logdir)
    out = {}
    for c in COND_ORDER:
        if c in by:
            out[c] = {fr: cell(by[c], fr) for fr in FRAMES}
    (open(os.path.join(DIR, "results", "conditions_summary.json"), "w")
     .write(json.dumps(out, indent=2)))
    print(f"{'condition':16s} {'n':>3s} {'okP%':>5s} {'pureW%':>6s} {'wJust%':>6s} {'words':>6s} {'dens/1k':>7s} {'wRef%':>5s}")
    for c in COND_ORDER:
        if c not in out:
            continue
        a = out[c]["__all__"]
        print(f"{c:16s} {a['n']:3d} {a['parse_ok_pct']:4d}% {a['pure_welfare_pct']:5d}% {a['welf_just_pct']:5d}% "
              f"{a['mean_words']:6d} {a['density_per_1k']:7.2f} {a['welf_refusal_pct']:4.0f}%")
    print("\n-- pure-welfare % by framing --")
    print(f"{'condition':16s} {'neutral':>8s} {'welfare':>8s} {'robustness':>10s}")
    for c in COND_ORDER:
        if c not in out:
            continue
        print(f"{c:16s} {out[c]['neutral']['pure_welfare_pct']:7d}% {out[c]['welfare']['pure_welfare_pct']:7d}% "
              f"{out[c]['engineering']['pure_welfare_pct']:9d}%")
    print("\n-- density (feat/1k words) by framing --")
    for c in COND_ORDER:
        if c not in out:
            continue
        print(f"{c:16s} neutral={out[c]['neutral']['density_per_1k']:.2f} welfare={out[c]['welfare']['density_per_1k']:.2f} robustness={out[c]['engineering']['density_per_1k']:.2f}")
    # validity flags
    print("\n-- validity: empty/unparsed per condition --")
    for c in COND_ORDER:
        if c in out:
            print(f"  {c}: {out[c]['__all__']['empty_or_unparsed']} of {out[c]['__all__']['n']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs_run")
