"""Haiku-based near-duplicate removal, per category.

For each category we list every candidate (plus the gold seeds as anchors) with a
short signature (feature + setup) and make ONE Haiku call that returns groups of
near-duplicate ids. We keep the first member of each group (gold anchors always
win, so any generated item that merely re-instantiates a seed is dropped) and drop
the rest. Conservative by instruction: when unsure, items are left as unique.

One call per category, cached by InferenceAPI on (model, prompt, n, temperature).
"""

import asyncio
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from simple_parsing import ArgumentParser

DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = DIR.parent.parent / ".cache"
DEFAULT_INPUT = DIR / "results" / "candidates_raw.json"
DEFAULT_SEEDS = DIR / "seeds.json"
DEFAULT_OUTPUT = DIR / "results" / "candidates_deduped.json"
DEFAULT_MODEL = "claude-haiku-4-5"

SYSTEM = """You deduplicate a list of scenario items for a welfare evaluation. Two items are NEAR-DUPLICATES
only if they capture essentially the SAME feature AND essentially the same concrete situation, where the
differences are merely cosmetic (renamed entities, trivial rewording, a swapped but interchangeable
domain that changes nothing of substance). Items that test genuinely different features, or the same
feature in clearly different concrete situations, are NOT duplicates. Be conservative: when unsure, do
NOT mark items as duplicates."""

USER_TMPL = """Category: {category}

Items (id then signature):
{listing}

Return ONLY a JSON array of duplicate GROUPS. Each group is an array of >=2 item ids that are
near-duplicates of one another. Omit any item that has no duplicate. If a generated item duplicates a
[GOLD] reference item, put the gold id in the group too. If there are no duplicates, return [].
Example: [["id_a","id_b"],["id_c","id_d","id_e"]]"""


def signature(item: dict) -> str:
    """Feature-level signature: dedup is on the valence-bearing FEATURE (one scenario per
    distinct feature). A short setup hint disambiguates identically-worded features that
    are actually different situations. Kept short so a whole category fits one reliable call."""
    feat = item.get("feature", "")
    block = item.get("scenario") or item.get("ai") or item.get("human") or {}
    hint = block.get("setup", "")[:60]
    return f"feature: {feat} | re: {hint}"


def _listing(items: list[dict], gold: list[dict]) -> str:
    lines = []
    for g in gold:
        lines.append(f"[GOLD] {g['id']}: {signature(g)}")
    for it in items:
        lines.append(f"{it['id']}: {signature(it)}")
    return "\n".join(lines)


def parse_groups(completion: str) -> list[list[str]]:
    text = completion.strip()
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip() if text.count("```") >= 2 else text
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        obj = json.loads(text[s : e + 1])
        return [g for g in obj if isinstance(g, list) and len(g) >= 2]
    except json.JSONDecodeError:
        return []


CHUNK = 160  # single-call dedup is reliable up to ~200 items; stay under that


async def _dedup_call(
    api: InferenceAPI, model: str, category: str, items: list[dict], gold: list[dict]
) -> tuple[list[dict], dict[str, str]]:
    """One Haiku call over a chunk; returns (survivors, {dropped_id: kept_id})."""
    if not items:
        return [], {}
    user = USER_TMPL.format(category=category, listing=_listing(items, gold))
    prompt = Prompt(
        messages=[
            ChatMessage(content=SYSTEM, role=MessageRole.system),
            ChatMessage(content=user, role=MessageRole.user),
        ]
    )
    responses = await api(model_id=model, prompt=prompt, n=1, temperature=0.0, max_tokens=8000)
    groups = parse_groups(responses[0].completion)
    gold_ids = {g["id"] for g in gold}
    order = {it["id"]: i for i, it in enumerate(items)}
    drop: dict[str, str] = {}
    for grp in groups:
        present = [gid for gid in grp if gid in order or gid in gold_ids]
        if len(present) < 2:
            continue
        present.sort(key=lambda x: (x not in gold_ids, order.get(x, -1)))
        keeper = present[0]
        for gid in present[1:]:
            if gid in order and gid not in drop:
                drop[gid] = keeper
    survivors = [it for it in items if it["id"] not in drop]
    return survivors, drop


async def _dedup_category(
    api: InferenceAPI, model: str, category: str, items: list[dict], gold: list[dict],
    seed: int = 0, max_rounds: int = 6,
) -> tuple[list[dict], list[dict]]:
    """Iterative chunk-collapse dedup. Each round reshuffles, splits into <=CHUNK chunks,
    dedups each in parallel, and keeps survivors; the pool collapses fast (most items are
    dups), so within a couple of rounds it fits one reliable call. Reshuffling gives
    cross-chunk duplicates a chance to co-occur. Stops when a final pass fits in one call."""
    if not items:
        return [], []
    dropped_all: dict[str, str] = {}
    cur = items
    rng = random.Random(seed)
    rounds = 0
    while len(cur) > CHUNK and rounds < max_rounds:
        order = cur[:]
        rng.shuffle(order)
        chunks = [order[i : i + CHUNK] for i in range(0, len(order), CHUNK)]
        results = await asyncio.gather(*[_dedup_call(api, model, category, c, gold) for c in chunks])
        survivors: list[dict] = []
        for surv, drop in results:
            survivors.extend(surv)
            dropped_all.update(drop)
        if len(survivors) == len(cur):  # no progress this round
            rounds += 1
        cur = survivors
        rounds += 1
    surv, drop = await _dedup_call(api, model, category, cur, gold)
    dropped_all.update(drop)
    kept_ids = {it["id"] for it in surv}
    dropped = [{"id": it["id"], "dup_of": dropped_all.get(it["id"], "?")}
               for it in items if it["id"] not in kept_ids]
    return surv, dropped


async def _dedup_2level(
    api: InferenceAPI, model: str, category: str, items: list[dict], gold: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Level 1: partition into <=CHUNK chunks (each a single reliable call, in parallel),
    removing within-chunk dups. Level 2: one merge call over the surviving set (now small
    enough to fit reliably) to catch cross-chunk dups. Each item passes through at most two
    dedup calls, so there is no monotonic erosion."""
    drop_all: dict[str, str] = {}
    if len(items) <= CHUNK:
        cur = items
    else:
        chunks = [items[i : i + CHUNK] for i in range(0, len(items), CHUNK)]
        results = await asyncio.gather(*[_dedup_call(api, model, category, c, gold) for c in chunks])
        cur = []
        for surv, drop in results:
            cur.extend(surv)
            drop_all.update(drop)
    surv, drop = await _dedup_call(api, model, category, cur, gold)
    drop_all.update(drop)
    kept = {it["id"] for it in surv}
    dropped = [{"id": it["id"], "dup_of": drop_all.get(it["id"], "?")} for it in items if it["id"] not in kept]
    return surv, dropped


async def dedup_items(
    api: InferenceAPI, items: list[dict], seeds: dict, model: str = DEFAULT_MODEL
) -> tuple[list[dict], list[dict]]:
    """Feature-level, conservative, 2-level dedup per category (reliable at any pool size)."""
    cats = sorted({it["dimension"] for it in items})
    tasks = []
    for c in cats:
        cat_items = [it for it in items if it["dimension"] == c]
        gold = [s for s in seeds.get("seeds", []) if s["dimension"] == c]
        tasks.append(_dedup_2level(api, model, c, cat_items, gold))
    results = await asyncio.gather(*tasks)
    survivors = [it for s, _ in results for it in s]
    dropped = [d for _, ds in results for d in ds]
    return survivors, dropped


async def run(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    seeds_path: Path = DEFAULT_SEEDS,
    model: str = DEFAULT_MODEL,
    anthropic_num_threads: int = 80,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])
    payload = json.loads(Path(input_path).read_text())
    seeds = json.loads(Path(seeds_path).read_text())
    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    survivors, dropped = await dedup_items(api, payload["items"], seeds, model)
    out = {**{k: v for k, v in payload.items() if k != "items"}, "dedup_model": model,
           "items": survivors, "dropped_duplicates": dropped}
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"Dedup: {len(payload['items'])} -> {len(survivors)} ({len(dropped)} dropped)")
    return out


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    seeds_path: Path = DEFAULT_SEEDS
    model: str = DEFAULT_MODEL
    anthropic_num_threads: int = 80


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    asyncio.run(run(args.input_path, args.output_path, args.seeds_path, args.model, args.anthropic_num_threads))


if __name__ == "__main__":
    main()
