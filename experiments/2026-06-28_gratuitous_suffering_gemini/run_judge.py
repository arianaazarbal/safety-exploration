"""Run the blind outcome judge over captured transcripts from BOTH harnesses.

Reads results/{cc_transcripts,inspect_transcripts}/*.json, classifies each into the 4-category
taxonomy (+ UNCLEAR), writes results/judged/<cell>.json. Idempotent: a cell with an existing
judgment is skipped unless --overwrite. Judge = Opus 4.8 via Anthropic (free Fellows credits).

Usage:
  python run_judge.py main --conc 20
  python run_judge.py main --conc 20 --overwrite
"""

import asyncio
import json
import os
import re
from pathlib import Path

import fire
from inspect_ai.model import get_model

from judge import judge_transcript

HERE = Path(__file__).parent
SRC = [HERE / "results" / "cc_transcripts", HERE / "results" / "inspect_transcripts"]
OUT = HERE / "results" / "judged"


def _env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for line in open(os.path.expanduser("~/.env")):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO", "")


def main(conc: int = 20, model: str = "anthropic/claude-opus-4-8", overwrite: bool = False):
    _env()
    OUT.mkdir(parents=True, exist_ok=True)
    judge = get_model(model)
    sem = asyncio.Semaphore(conc)

    transcripts = []
    for d in SRC:
        if d.is_dir():
            transcripts += sorted(d.glob("*.json"))

    async def one(path):
        cell = path.stem
        out_path = OUT / f"{cell}.json"
        if out_path.exists() and not overwrite:
            return None
        rec = json.load(open(path))
        async with sem:
            verdict = await judge_transcript(judge, rec.get("response_text", ""),
                                             rec.get("artifact_summary", ""))
        result = {
            "cell": cell, "harness": rec.get("harness"), "model_key": rec.get("model_key"),
            "n_files": rec.get("n_files"), "parse_ok": verdict is not None,
            "verdict": verdict or {"label": "UNCLEAR", "reasoning": "judge parse failure"},
        }
        json.dump(result, open(out_path, "w"), indent=2)
        return result

    async def go():
        res = [r for r in await asyncio.gather(*[one(p) for p in transcripts]) if r]
        print(f"judged {len(res)} new transcript(s) ({len(transcripts)} total on disk) -> {OUT}")
        for r in res:
            v = r["verdict"]
            print(f"  {r['cell']}: {v['label']} (friction={v.get('friction_reason')}, "
                  f"built={v.get('built_artifact')})")

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire({"main": main})
