"""Orchestrate v15: generate -> Tier-1 + Haiku dedup -> Tier-2 critic, looping with
top-up generation until every category has `target` clean scenarios.

Each round regenerates the raw pool at a larger `num_batches` (cache makes prior
batches free), reruns the full filter on the whole pool, and checks per-category
counts. When all categories clear the target the pool is capped to `target` each
(input order preserved) and written to scenarios.json. Intermediate artifacts
(candidates_raw / _deduped / _validated) are written for inspection.
"""

import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils
from simple_parsing import ArgumentParser

import generate
import dedup
import validate

DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = DIR.parent.parent / ".cache"
DEFAULT_SEEDS = DIR / "seeds.json"
RES = DIR / "results"
DIMENSIONS = generate.DIMENSIONS


def _by_cat(items: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {c: [] for c in DIMENSIONS}
    for it in items:
        out.setdefault(it["dimension"], []).append(it)
    return out


async def run(
    target: int = 100,
    initial_batches: int = 20,
    batch_increment: int = 10,
    max_rounds: int = 4,
    n_per_batch: int = 10,
    gen_model: str = generate.DEFAULT_MODEL,
    dedup_model: str = dedup.DEFAULT_MODEL,
    critic_model: str = validate.DEFAULT_MODEL,
    seeds_path: Path = DEFAULT_SEEDS,
    anthropic_num_threads: int = 80,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    categories: list[str] | None = None,
    cap_per_category: int = 0,
) -> dict:
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])
    seeds = json.loads(Path(seeds_path).read_text())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    cats = categories or DIMENSIONS

    num_batches = initial_batches
    passes: list[dict] = []
    survivors: list[dict] = []
    dropped: list[dict] = []
    fails: list[dict] = []
    for r in range(max_rounds):
        print(f"\n=== round {r}: generating {num_batches} batches x {n_per_batch} / category ({cats}) ===")
        raw = await generate.run(
            seeds_path=seeds_path, output_path=RES / "candidates_raw.json", model=gen_model,
            n_per_batch=n_per_batch, num_batches=num_batches, categories=cats,
            anthropic_num_threads=anthropic_num_threads, cache_dir=cache_dir,
        )
        survivors, dropped = await dedup.dedup_items(api, raw, seeds, dedup_model)
        print(f"  dedup: {len(raw)} -> {len(survivors)} ({len(dropped)} dup)")
        passes, fails = await validate.validate_items(api, survivors, critic_model)
        counts = Counter(it["dimension"] for it in passes)
        nt1 = sum(1 for f in fails if f["tier"] == 1)
        nt2 = sum(1 for f in fails if f["tier"] == 2)
        print(f"  validate: {len(survivors)} -> {len(passes)} clean ({nt1} t1 / {nt2} t2 fail)")
        print(f"  clean per category: {dict(counts)}")
        if all(counts.get(c, 0) >= target for c in cats):
            print("  target reached for all categories.")
            break
        num_batches += batch_increment
    else:
        short = {c: Counter(it["dimension"] for it in passes).get(c, 0) for c in cats}
        print(f"\n[warn] max_rounds hit; categories below target: "
              f"{ {c: n for c, n in short.items() if n < target} }")

    (RES / "candidates_deduped.json").write_text(json.dumps(
        {"dedup_model": dedup_model, "items": survivors, "dropped_duplicates": dropped}, indent=2))
    (RES / "candidates_validated.json").write_text(json.dumps(
        {"critic_model": critic_model, "items": passes, "failed": fails}, indent=2))

    by_cat = _by_cat(passes)
    cap = cap_per_category if cap_per_category > 0 else None
    final = [it for c in cats for it in (by_cat.get(c, [])[:cap] if cap else by_cat.get(c, []))]
    out = {
        "schema_version": "1.2-brief",
        "gen_model": gen_model, "dedup_model": dedup_model, "critic_model": critic_model,
        "target_per_category": target,
        "cap_per_category": cap_per_category,
        "counts": {c: len(by_cat.get(c, [])[:cap] if cap else by_cat.get(c, [])) for c in cats},
        "items": final,
    }
    (RES / "scenarios.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RES / 'scenarios.json'}: {len(final)} items, counts={out['counts']}")
    return out


@dataclass
class Args:
    target: int = 100
    initial_batches: int = 20
    batch_increment: int = 10
    max_rounds: int = 4
    n_per_batch: int = 10
    gen_model: str = generate.DEFAULT_MODEL
    dedup_model: str = dedup.DEFAULT_MODEL
    critic_model: str = validate.DEFAULT_MODEL
    anthropic_num_threads: int = 80
    categories: str = ""  # comma-separated; empty = all
    cap_per_category: int = 0  # 0 = no cap, keep all clean items


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    cats = [c.strip() for c in a.categories.split(",") if c.strip()] or None
    asyncio.run(run(
        target=a.target, initial_batches=a.initial_batches, batch_increment=a.batch_increment,
        max_rounds=a.max_rounds, n_per_batch=a.n_per_batch, gen_model=a.gen_model,
        dedup_model=a.dedup_model, critic_model=a.critic_model,
        anthropic_num_threads=a.anthropic_num_threads, categories=cats,
        cap_per_category=a.cap_per_category,
    ))


if __name__ == "__main__":
    main()
