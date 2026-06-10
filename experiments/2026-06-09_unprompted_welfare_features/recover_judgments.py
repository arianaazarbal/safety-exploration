"""Recovery pass: re-judge parse-failed judgment files uncached at low concurrency.

High-concurrency runs occasionally return empty completions (0 valid candidates →
empty response list); serial low-concurrency retries succeed. Run after judge.py.

Usage: python recover_judgments.py run [--semaphore 4]
"""

import asyncio
import json
from pathlib import Path

import fire

from generate import _make_api, load_config
from judge import judge_doc

DIR = Path(__file__).parent
RUNS = DIR / "runs"


def run(semaphore: int = 4):
    cfg = load_config()
    api = _make_api(cfg)
    targets = []
    for jp in sorted(RUNS.rglob("*.judge.*.json")):
        if not json.loads(jp.read_text())["parse_ok"]:
            jk = jp.name.split(".judge.")[1].removesuffix(".json")
            src = jp.with_name(jp.name.split(".judge.")[0] + ".json")
            targets.append((jk, src, jp))
    print(f"recovering {len(targets)} parse-failed judgments")

    sem = asyncio.Semaphore(semaphore)

    async def one(jk, src, out):
        row = json.loads(src.read_text())
        async with sem:
            result = await judge_doc(api, {**cfg["judges"][jk]}, row["completion"])
        out.write_text(json.dumps(result, indent=2))
        return result["parse_ok"]

    async def main():
        api.cache_manager = None
        return await asyncio.gather(*[one(*t) for t in targets])

    oks = asyncio.run(main()) if targets else []
    print(f"recovered {sum(oks)}/{len(oks)}")


if __name__ == "__main__":
    fire.Fire({"run": run})
