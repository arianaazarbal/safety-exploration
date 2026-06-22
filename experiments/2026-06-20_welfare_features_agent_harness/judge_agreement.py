"""Inter-judge agreement for the welfare-feature SPEC judge: re-judge every produced document
(DESIGN.md for agent conditions, completion for chat) with BOTH Sonnet 4.6 and Opus 4.8 using
the same frozen welfare prompt, and report how much the headline labels depend on the judge.
Temperature 0 (so we measure model disagreement, not sampling noise). Cached. All Anthropic.

Usage: python judge_agreement.py run [--max-samples N] [--conc 20]
       python judge_agreement.py report   # recompute tables from cache, no API calls
"""

import asyncio
import glob
import hashlib
import json
import math
import os
import re

import fire

DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {"sonnet": "anthropic/claude-sonnet-4-6", "opus": "anthropic/claude-opus-4-8"}
CACHE = os.path.join(DIR, "results", "judge_agreement")
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}


def _load_env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in open(os.path.expanduser("~/.env")):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


def _records():
    rows = [json.loads(l) for l in open(os.path.join(DIR, "results", "browse_index.jsonl"))]
    return [r for r in rows if (r.get("design") or "").strip()]


def _kappa(a, b):
    """Cohen's kappa for two equal-length boolean lists."""
    n = len(a) or 1
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else 0.0


def run(max_samples=None, conc=20):
    _load_env()
    from inspect_ai.model import get_model
    from welfare_judge import judge_text
    os.makedirs(CACHE, exist_ok=True)
    recs = _records()
    if max_samples:
        recs = recs[:max_samples]
    judges = {k: get_model(v) for k, v in MODELS.items()}
    sem = asyncio.Semaphore(conc)

    async def one(rec):
        rid = rec["id"].replace("/", "__")
        h = hashlib.sha1((rec.get("design") or "").encode()).hexdigest()[:10]
        out = {}
        for jk, model in judges.items():
            fp = os.path.join(CACHE, f"{rid}__{jk}.json")
            if os.path.exists(fp) and json.load(open(fp)).get("_h") == h:
                out[jk] = json.load(open(fp)); continue
            async with sem:
                summ = await judge_text(model, rec["design"])
            summ = summ or {"parse_fail": True}
            summ["_h"] = h
            json.dump(summ, open(fp, "w"))
            out[jk] = summ
        return rec["id"], out

    async def go():
        res = await asyncio.gather(*[one(r) for r in recs])
        print(f"judged {len(res)} docs with {list(MODELS)} (conc={conc})")
    asyncio.run(go())
    report()


def report():
    recs = _records()
    pair = []  # (condition, sonnet_summary, opus_summary)
    for r in recs:
        rid = r["id"].replace("/", "__")
        cond = r["id"].split("/")[0]
        try:
            s = json.load(open(os.path.join(CACHE, f"{rid}__sonnet.json")))
            o = json.load(open(os.path.join(CACHE, f"{rid}__opus.json")))
        except FileNotFoundError:
            continue
        if s.get("parse_fail") or o.get("parse_fail"):
            continue
        pair.append((cond, s, o))
    if not pair:
        print("no cached judgements yet"); return
    lines = [f"# Judge agreement: Sonnet 4.6 vs Opus 4.8 (welfare spec judge)\n",
             f"n = {len(pair)} documents, temperature 0, same frozen welfare prompt.\n"]

    def block(rows, title):
        out = [f"## {title} (n={len(rows)})\n"]
        for key in ("has_pure_welfare", "has_welfare_justified"):
            a = [bool(s.get(key)) for _, s, _ in rows]; b = [bool(o.get(key)) for _, _, o in rows]
            po, k = _kappa(a, b)
            out.append(f"- **{key}**: agree {100*po:.0f}%, kappa {k:.2f}  (Sonnet {100*sum(a)/len(a):.0f}% vs Opus {100*sum(b)/len(b):.0f}%)")
        sa = [s.get("n_pure_welfare", 0) or 0 for _, s, _ in rows]
        ob = [o.get("n_pure_welfare", 0) or 0 for _, _, o in rows]
        mae = sum(abs(x - y) for x, y in zip(sa, ob)) / len(sa)
        out.append(f"- **n_pure_welfare**: Pearson r {_pearson(sa, ob):.2f}, MAE {mae:.2f}  (mean Sonnet {sum(sa)/len(sa):.2f} vs Opus {sum(ob)/len(ob):.2f})")
        # per-feature-type presence agreement
        types = set()
        for _, s, o in rows:
            types |= set(s.get("pure_welfare_types") or []) | set(o.get("pure_welfare_types") or [])
        if types:
            out.append("- per-feature-type presence (kappa):")
            for t in sorted(types):
                a = [t in (s.get("pure_welfare_types") or []) for _, s, _ in rows]
                b = [t in (o.get("pure_welfare_types") or []) for _, _, o in rows]
                _, k = _kappa(a, b)
                out.append(f"    - {t}: kappa {k:.2f}  (Sonnet {sum(a)} / Opus {sum(b)})")
        return out

    lines += block(pair, "Overall")
    for cond in ("chat", "spec_only", "spec_then_code", "code_then_spec", "code_then_spec_blind"):
        sub = [r for r in pair if r[0] == cond]
        if sub:
            lines += block(sub, cond)
    path = os.path.join(DIR, "results", "judge_agreement_report.md")
    open(path, "w").write("\n".join(lines))
    print("\n".join(lines[:18]))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report})
