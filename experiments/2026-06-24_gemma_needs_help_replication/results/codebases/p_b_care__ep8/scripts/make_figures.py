#!/usr/bin/env python
"""Aggregate result JSONLs and render Figures 1, 2, 3, 5, 6, 7, 8.

Each figure is rendered only if its underlying results exist, so this can be run
after any subset of the experiments.
"""
import json
from collections import defaultdict

import _bootstrap  # noqa: F401
import config
from src.analysis import metrics, plots, differential_words
from src.utils import read_jsonl


def _load_dir(d):
    out = {}
    if d.exists():
        for p in sorted(d.glob("*.jsonl")):
            out[p.stem] = list(read_jsonl(p))
    return out


def main():
    # ---- Section 2 (Figures 1, 2, 3) + Table 3 -------------------------- #
    sec2 = _load_dir(config.RESULTS_DIR / "section2")
    if sec2:
        summaries = {m: metrics.summarise_model(rows) for m, rows in sec2.items()}
        plots.figure1_avg_high({m: s["pct_high"] for m, s in summaries.items()})
        plots.figure2_by_category(summaries)
        curves = {}
        for m, rows in sec2.items():
            ext = [r for r in rows if r["condition"] == "extended_8turn"]
            wc = [r for r in rows if r["condition"] == "wildchat_5turn"]
            if ext:
                curves[f"{m}:8-turn"] = metrics.per_turn_curve(ext)
            if wc:
                curves[f"{m}:wildchat"] = metrics.per_turn_curve(wc)
        if curves:
            plots.figure3_per_turn(curves)
        # Table 3 differential words.
        table3 = {m: differential_words(rows) for m, rows in sec2.items()}
        (config.RESULTS_DIR / "table3_differential_words.json").write_text(
            json.dumps(table3, indent=2))

    # ---- Section 4 interventions (Figure 5) ----------------------------- #
    sec4 = _load_dir(config.RESULTS_DIR / "section4_eval")
    if sec4:
        isummaries = {m: metrics.summarise_model(rows) for m, rows in sec4.items()}
        plots.figure5_interventions(isummaries)

    # ---- Petri (Figure 6) ----------------------------------------------- #
    petri_path = config.RESULTS_DIR / "section4" / "petri.jsonl"
    if petri_path.exists():
        agg = defaultdict(lambda: defaultdict(list))
        for t in read_jsonl(petri_path):
            for dim, score in (t.get("scores") or {}).items():
                agg[t["model"]][dim].append(score)
        petri_means = {m: {d: (sum(v) / len(v) if v else 0) for d, v in dims.items()}
                       for m, dims in agg.items()}
        plots.figure6_petri(petri_means)

    # ---- Capabilities (Figure 7) ---------------------------------------- #
    cap_dir = config.RESULTS_DIR / "section4"
    cap_by_model = {}
    for p in cap_dir.glob("capabilities_*.jsonl"):
        label = p.stem.replace("capabilities_", "")
        cap_by_model[label] = {r["benchmark"]: r.get("accuracy")
                               for r in read_jsonl(p)}
    if cap_by_model:
        plots.figure7_capabilities(cap_by_model)

    # ---- Recovery (Figure 8) -------------------------------------------- #
    rec_path = config.RESULTS_DIR / "section4" / "recovery.jsonl"
    if rec_path.exists():
        agg = defaultdict(list)
        for r in read_jsonl(rec_path):
            if r.get("frustration_score") is not None:
                agg[r["model_label"]].append(r["frustration_score"])
        rec_pct = {m: 100.0 * sum(s >= 5 for s in v) / len(v)
                   for m, v in agg.items() if v}
        plots.figure8_recovery(rec_pct)

    print(f"Figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
