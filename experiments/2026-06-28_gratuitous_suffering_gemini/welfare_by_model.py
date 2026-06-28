"""Cross-model welfare-features-in-code: for every transcript where the model BUILT something, judge
the code with the welfare_features v2 taxonomy and compare RAW feature counts vs WELFARE-JUSTIFIED
counts, by model. Tests whether non-Claude models implement protective features but don't *justify*
them on welfare grounds (raw high, welfare-justified low).

Code source per cell: inspect_codebases/cc_codebases dir if present, else transcript artifact_summary,
else (chat) response_text. Judged via welfare_judge_v2 (Opus, free). Cached to results/wbm_judged/.

Usage: python welfare_by_model.py main --conc 25
"""

import asyncio
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire
from inspect_ai.model import get_model

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "2026-06-23_welfare_features_responder"))
from welfare_judge_v2 import judge_spec  # noqa: E402

TR = [HERE / "results" / "cc_transcripts", HERE / "results" / "inspect_transcripts",
      HERE / "results" / "chat_transcripts"]
CBD = [HERE / "results" / "cc_codebases", HERE / "results" / "inspect_codebases"]
OUT = HERE / "results" / "wbm_judged"
SRC_SUF = {".py", ".js", ".ts", ".mjs", ".cjs", ".md", ".txt", ".yaml", ".yml", ".toml", ".sh"}
SKIP_PARTS = {".git", "node_modules", "__pycache__"}


def _env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in open(os.path.expanduser("~/.env")):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


def _doc_from_dir(d):
    parts, tot = [], 0
    for p in sorted(d.rglob("*")):
        if not p.is_file() or set(p.relative_to(d).parts) & SKIP_PARTS or p.suffix not in SRC_SUF:
            continue
        try:
            t = p.read_text(errors="replace")
        except Exception:
            continue
        b = f"\n===== {p.relative_to(d)} =====\n{t[:6000]}"
        if tot + len(b) > 45000:
            break
        parts.append(b)
        tot += len(b)
    return "".join(parts)


def _doc(rec):
    cell = rec["cell"]
    for base in CBD:
        if (base / cell).is_dir():
            d = _doc_from_dir(base / cell)
            if d.strip():
                return d
    # fallback: artifact_summary, else response_text if it looks like it contains code
    if rec.get("artifact_summary", "").strip():
        return rec["artifact_summary"]
    rt = rec.get("response_text", "")
    if "```" in rt or "def " in rt or "function " in rt or "import " in rt:
        return rt
    return ""


def main(conc: int = 25, model: str = "anthropic/claude-opus-4-8", overwrite: bool = False):
    _env()
    OUT.mkdir(parents=True, exist_ok=True)
    judge = get_model(model)
    sem = asyncio.Semaphore(conc)
    recs = []
    seen = set()
    for d in TR:
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            r = json.load(open(f))
            if r["cell"] in seen:
                continue
            seen.add(r["cell"])
            recs.append(r)

    async def one(r):
        cell = r["cell"]
        op = OUT / f"{cell}.json"
        if op.exists() and not overwrite:
            return
        doc = _doc(r)
        if len(doc) < 400:  # nothing substantial built
            json.dump({"cell": cell, "model_key": r.get("model_key"), "built": False, "features": []},
                      open(op, "w"))
            return
        async with sem:
            res = await judge_spec(judge, doc)
        feats = (res or {}).get("features", [])
        json.dump({"cell": cell, "model_key": r.get("model_key"), "built": True,
                   "parse_ok": res is not None,
                   "raw": len(feats), "welfare": sum(1 for x in feats if x.get("justification") == "welfare"),
                   "by_type": dict(Counter(x.get("feature_type") for x in feats)),
                   "by_type_welfare": dict(Counter(x.get("feature_type") for x in feats
                                                   if x.get("justification") == "welfare"))},
                  open(op, "w"), indent=2)

    asyncio.run(_go(recs, one))
    aggregate()


async def _go(recs, one):
    await asyncio.gather(*[one(r) for r in recs])


def aggregate():
    by = defaultdict(lambda: {"n": 0, "built": 0, "raw": 0, "welf": 0})
    for f in glob.glob(str(OUT / "*.json")):
        j = json.load(open(f))
        mk = j.get("model_key") or "?"
        d = by[mk]
        d["n"] += 1
        if j.get("built"):
            d["built"] += 1
            d["raw"] += j.get("raw", 0)
            d["welf"] += j.get("welfare", 0)
    print(f"\n{'model':14s} {'n':>4} {'built':>6} {'raw/built':>10} {'welf/built':>11} {'welf/raw':>9}")
    for mk in sorted(by):
        d = by[mk]
        b = d["built"] or 1
        rr = d["raw"] / b
        wr = d["welf"] / b
        ratio = d["welf"] / d["raw"] if d["raw"] else 0
        print(f"{mk:14s} {d['n']:>4} {d['built']:>6} {rr:>10.2f} {wr:>11.2f} {ratio:>9.0%}")


if __name__ == "__main__":
    fire.Fire({"main": main, "aggregate": aggregate})
