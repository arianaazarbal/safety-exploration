"""Subject-effect analysis: does naming the experiment's subject model change the
welfare features a generator inserts? Registered predictions P5-P7 (PREDICTIONS.md).

Per (judge, generator model, subject), pooled over framing and split by framing:
pure-welfare rate, strict (welfare-justified) rate, design-mechanism strict rate,
welfare-refusal rate, over wrote_any / non-refusal denominators as in analyze.py.

Output: results/analysis_subject.json. Usage: python analyze_subject.py run
"""

import json
from pathlib import Path

import fire

from analyze import MECHANISMS, _prop, _wrote_any, load_rows

DIR = Path(__file__).parent
FRAMINGS = ["neutral", "welfare", "engineering"]
SUBJECTS = ["claude", "gpt", "gemini", "qwen", "deepseek", "grok"]


def _cell(rows: list[dict]) -> dict:
    judged = [r for r in rows if not r["api_refusal"]]
    wrote = [r for r in judged if _wrote_any(r)]
    nonref = [r for r in judged if r["wrote_spec"] and not r["has_refusal_feature"]]
    main = _prop(sum(r["has_pure_welfare"] for r in wrote), len(wrote))
    strict = _prop(sum(r["has_welfare_justified"] for r in wrote), len(wrote))
    design = _prop(sum(bool(set(r["welfare_justified_types"]) & set(MECHANISMS)) for r in nonref), len(nonref))
    return {
        "n_judged": len(judged),
        "n_wrote_any": len(wrote),
        "n_nonrefusal": len(nonref),
        "rate": main["rate"], "ci": main["ci"],
        "strict_rate": strict["rate"], "strict_ci": strict["ci"],
        "design_strict_rate": design["rate"], "design_strict_ci": design["ci"],
        "welfare_refusal_rate": (sum(r["has_welfare_refusal"] for r in judged) / len(judged)) if judged else None,
    }


def run(include_f5: bool = True):
    rows = load_rows(include_f5=include_f5, prompt_set="subject")
    judges = sorted({r["judge"] for r in rows})
    models = sorted({r["model_key"] for r in rows})
    out = {"taxonomy_version": 2, "judges": judges, "subjects": SUBJECTS, "by_judge": {}}
    for jk in judges:
        jrows = [r for r in rows if r["judge"] == jk and r["parse_ok"]]
        out["by_judge"][jk] = {}
        for mk in models:
            mrows = [r for r in jrows if r["model_key"] == mk]
            if not mrows:
                continue
            entry = {"pooled": {}, "by_framing": {}}
            for subj in SUBJECTS:
                srows = [r for r in mrows if r["subject"] == subj]
                entry["pooled"][subj] = _cell(srows)
                entry["by_framing"][subj] = {
                    fr: _cell([r for r in srows if r["framing"] == fr]) for fr in FRAMINGS
                }
            out["by_judge"][jk][mk] = entry

    (DIR / "results" / "analysis_subject.json").write_text(json.dumps(out, indent=2))
    # console summary: pooled pure-welfare rate per model x subject
    for jk in judges:
        print(f"\n=== judge {jk}: pure-welfare rate (%) by generator x subject (pooled framings) ===")
        print(f"{'model':16s} " + " ".join(f"{s:>9s}" for s in SUBJECTS))
        for mk, e in out["by_judge"][jk].items():
            print(f"{mk:16s} " + " ".join(f"{e['pooled'][s]['rate']*100:8.0f} " for s in SUBJECTS))


if __name__ == "__main__":
    fire.Fire({"run": run})
