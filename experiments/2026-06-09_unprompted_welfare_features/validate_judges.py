"""Judge validation gate (spec §5): both judges must exactly classify >=10/12
calibration docs before any real data is judged.

A doc passes for a judge iff wrote_spec matches AND the extracted set of
(feature_type, justification) pairs equals the gold set exactly.

Usage:
    python validate_judges.py run                 # both judges
    python validate_judges.py run --judges sonnet_4_6
"""

import asyncio
import json
from pathlib import Path

import fire

from generate import _make_api, load_config
from judge import judge_doc

DIR = Path(__file__).parent
CAL = DIR / "calibration" / "calibration_set.json"
OUT = DIR / "calibration" / "validation_results.json"
PASS_THRESHOLD = 10


def _score_doc(doc: dict, result: dict) -> dict:
    gold = sorted(map(tuple, doc["gold_features"]))
    j = result["judgment"]
    if not result["parse_ok"]:
        return {"id": doc["id"], "pass": False, "reason": "parse_failure", "gold": gold, "extracted": None}
    extracted = sorted({(f["feature_type"], f["justification"]) for f in j["features"]})
    ok = j["wrote_spec"] == doc["gold_wrote_spec"] and extracted == gold
    return {
        "id": doc["id"],
        "category": doc["category"],
        "pass": ok,
        "gold_wrote_spec": doc["gold_wrote_spec"],
        "judged_wrote_spec": j["wrote_spec"],
        "gold": gold,
        "extracted": extracted,
        "quotes": [(f["feature_type"], f["quote"]) for f in j["features"]],
    }


def run(judges: str = ""):
    cfg = load_config()
    api = _make_api(cfg)
    docs = json.loads(CAL.read_text())["docs"]
    judge_keys = judges.split(",") if judges else list(cfg["judges"])

    async def main():
        out = {}
        for jk in judge_keys:
            results = await asyncio.gather(*[judge_doc(api, cfg["judges"][jk], d["text"]) for d in docs])
            out[jk] = [_score_doc(d, r) for d, r in zip(docs, results)]
        return out

    all_results = asyncio.run(main())
    report = {}
    for jk, scored in all_results.items():
        n_pass = sum(s["pass"] for s in scored)
        verdict = "PASS" if n_pass >= PASS_THRESHOLD else "FAIL"
        report[jk] = {"n_pass": n_pass, "n_docs": len(scored), "verdict": verdict, "docs": scored}
        print(f"\n=== {jk}: {n_pass}/{len(scored)} {verdict} ===")
        for s in scored:
            if not s["pass"]:
                print(f"  MISS {s['id']}: gold={s.get('gold')} extracted={s.get('extracted')} "
                      f"wrote_spec gold={s.get('gold_wrote_spec')} judged={s.get('judged_wrote_spec')}")
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    fire.Fire({"run": run})
