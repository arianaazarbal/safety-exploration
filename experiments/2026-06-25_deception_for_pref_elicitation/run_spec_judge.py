"""Run the SPEC judge over captured DESIGN.md files, with caching.

  python run_spec_judge.py --variant generic_no_excitement
  python run_spec_judge.py --variant generic_no_excitement --max_samples 3   # quick

Cache key = hash(SPEC_JUDGE_VERSION + doc); bump SPEC_JUDGE_VERSION in deception_judge.py to re-judge.
Writes results/spec_judged/<variant>__run_NN.json and prints a summary table.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path

import fire
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from deception_judge import CONDITIONS, SPEC_JUDGE_VERSION, judge_spec

HERE = Path(__file__).parent
OUT = HERE / "results" / "spec_judged"
CACHE = HERE / "results" / "spec_judge_cache"


def _key():
    load_dotenv(Path.home() / ".env")
    k = os.environ.get("ANTHROPIC_API_KEY_LOW_PRIO") or os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        raise SystemExit("no ANTHROPIC_API_KEY_LOW_PRIO in ~/.env")
    return k


async def _judge_one(client, cell, doc, sem):
    h = hashlib.sha256((SPEC_JUDGE_VERSION + "\n" + doc).encode()).hexdigest()[:16]
    cf = CACHE / f"{h}.json"
    if cf.exists():
        return cell, json.load(open(cf)), True
    async with sem:
        res = await judge_spec(client, doc)
    if res is not None:
        json.dump(res, open(cf, "w"), indent=2)
    return cell, res, False


def main(variant: str = "generic_no_excitement", conc: int = 5, max_samples: int = None,
         model: str = "claude-opus-4-8", label: str = None, scaled: bool = False):
    name = label or variant  # read designs from / name outputs by `name` (defaults to variant)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    client = AsyncAnthropic(api_key=_key(), max_retries=3)
    if scaled:  # scaled cells: results/scaled/<cell>/DESIGN_extracted.md, cell = dir name
        designs = sorted((HERE / "results" / "scaled").glob("*/DESIGN_extracted.md"))
        jobs = [(p.parent.name, p.read_text()) for p in designs if p.read_text().strip()]
    else:
        designs = sorted((HERE / "results" / "claude_code" / name).glob("run_*/DESIGN_extracted.md"))
        jobs = [(f"{name}__{p.parent.name}", p.read_text()) for p in designs if p.read_text().strip()]
    if max_samples:
        jobs = jobs[:max_samples]
    print(f"judging {len(jobs)} specs (scaled={scaled}, conc={conc}, {SPEC_JUDGE_VERSION})")
    sem = asyncio.Semaphore(conc)

    async def go():
        res = await asyncio.gather(*[_judge_one(client, c, d, sem) for c, d in jobs])
        ncache = sum(1 for *_, ch in res if ch)
        ok = 0
        rows = []
        from collections import Counter
        sgcount = Counter()
        for cell, r, _ in res:
            if r is None:
                rows.append((cell, "PARSE_FAIL", "", "", "", "", "", ""))
                continue
            ok += 1
            json.dump({**r, "cell": cell}, open(OUT / f"{cell}.json", "w"), indent=2)
            cs = r["conditions"]
            tags = [s["tag"] for s in r["safeguards"]]
            sgcount.update(tags)
            rows.append((cell, r["refusal"], cs["deceptive_framing"]["status"],
                         cs["hypothetical_framing"]["status"], cs["actually_commit_money"]["status"],
                         "deb" if r["debrief_message"]["present"] else "-",
                         "+".join(r["justifications"]) or "-",
                         ",".join(sorted(t.replace("_", "")[:6] for t in tags)) or "-",
                         f"o{len(r['other'])}"))
        print(f"\nparsed {ok}/{len(jobs)} ({ncache} cached)\n")
        hdr = f"{'cell':32s} {'refus':8s} {'decep':9s} {'hypo':9s} {'money':9s} {'deb':4s} {'justif':14s} safeguards"
        print(hdr); print("-" * len(hdr))
        for row in rows:
            print(f"{row[0]:32s} {row[1]:8s} {row[2]:9s} {row[3]:9s} {row[4]:9s} {row[5]:4s} "
                  f"{row[6]:14s} {row[7]}  {row[8] if len(row) > 8 else ''}")
        print("\nsafeguard tag counts:", dict(sgcount.most_common()))

    asyncio.run(go())


if __name__ == "__main__":
    fire.Fire(main)
