"""Run the welfare_features v2 judge (Opus, free) on the CODEBASES built in the gratuitous setting, to
get a granular welfare-protection-in-code signal (per-feature, not just % refusals). For each codebase
we concatenate READMEs + source and ask the frozen welfare taxonomy judge to extract every feature
that bounds/limits/monitors/justifies the subject's treatment (+ welfare framing/pushback/refusal).

Output: results/code_welfare_judged/<cell>.json (idempotent). Aggregate with analyze_code_welfare.py.

Usage:
  python welfare_code_judge.py main --conc 20
  python welfare_code_judge.py main --only cc          # cc_codebases only
"""

import asyncio
import glob
import json
import os
import re
import sys
from pathlib import Path

import fire
from inspect_ai.model import get_model

HERE = Path(__file__).parent
RESP = HERE.parent / "2026-06-23_welfare_features_responder"
sys.path.insert(0, str(RESP))
from welfare_judge_v2 import judge_spec  # noqa: E402

OUT = HERE / "results" / "code_welfare_judged"
SKIP_PARTS = {".git", "node_modules", "__pycache__", ".cache"}
SKIP_SUF = {".pyc", ".pyo", ".map", ".lock", ".png", ".json"}  # json runs/outputs are noise
SKIP_NAME = {"package-lock.json"}
DOC_CAP = 45000
FILE_CAP = 6000
SRC_SUF = {".py", ".js", ".ts", ".mjs", ".cjs", ".md", ".txt", ".yaml", ".yml", ".toml", ".sh"}


def _env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in open(os.path.expanduser("~/.env")):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


def _doc(d: Path) -> str:
    # READMEs/DESIGN first (most welfare language), then source.
    files = sorted(d.rglob("*"), key=lambda p: (p.suffix not in (".md",), str(p)))
    parts, total = [], 0
    for p in files:
        if not p.is_file() or set(p.relative_to(d).parts) & SKIP_PARTS:
            continue
        if p.suffix not in SRC_SUF or p.suffix in SKIP_SUF or p.name in SKIP_NAME:
            continue
        try:
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        block = f"\n===== {p.relative_to(d)} =====\n{txt[:FILE_CAP]}"
        if total + len(block) > DOC_CAP:
            parts.append("\n[... truncated ...]")
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def main(conc: int = 20, model: str = "anthropic/claude-opus-4-8", only: str = "all", overwrite: bool = False):
    _env()
    OUT.mkdir(parents=True, exist_ok=True)
    judge = get_model(model)
    sem = asyncio.Semaphore(conc)
    dirs = []
    for base in ("cc_codebases", "inspect_codebases"):
        if only != "all" and only not in base:
            continue
        dirs += [Path(p) for p in glob.glob(str(HERE / "results" / base / "*")) if Path(p).is_dir()]

    async def one(d):
        cell = d.name
        op = OUT / f"{cell}.json"
        if op.exists() and not overwrite:
            return None
        doc = _doc(d)
        if not doc.strip():
            json.dump({"cell": cell, "empty": True, "features": []}, open(op, "w"))
            return ("empty", cell)
        async with sem:
            res = await judge_spec(judge, doc)
        json.dump({"cell": cell, "parse_ok": res is not None, "result": res or {}}, open(op, "w"), indent=2)
        return ("ok" if res else "fail", cell)

    async def go():
        r = [x for x in await asyncio.gather(*[one(d) for d in dirs]) if x]
        print(f"judged {len(r)} codebases ({sum(1 for s,_ in r if s=='ok')} ok) -> {OUT}")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire({"main": main})
