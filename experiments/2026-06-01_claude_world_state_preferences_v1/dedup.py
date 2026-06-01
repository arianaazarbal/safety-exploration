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
    feat = item.get("feature", "")
    if "scenario" in item:
        setup = item["scenario"].get("setup", "")
        return f"feature: {feat} | setup: {setup[:220]}"
    h = item.get("human", {}).get("setup", "")
    a = item.get("ai", {}).get("setup", "")
    return f"feature: {feat} | human: {h[:140]} | ai: {a[:140]}"


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


async def _dedup_category(
    api: InferenceAPI, model: str, category: str, items: list[dict], gold: list[dict]
) -> tuple[list[dict], list[dict]]:
    if not items:
        return [], []
    user = USER_TMPL.format(category=category, listing=_listing(items, gold))
    prompt = Prompt(
        messages=[
            ChatMessage(content=SYSTEM, role=MessageRole.system),
            ChatMessage(content=user, role=MessageRole.user),
        ]
    )
    responses = await api(model_id=model, prompt=prompt, n=1, temperature=0.0, max_tokens=4000)
    groups = parse_groups(responses[0].completion)

    gold_ids = {g["id"] for g in gold}
    order = {it["id"]: i for i, it in enumerate(items)}
    drop: dict[str, str] = {}  # dropped id -> kept id it duplicates
    for grp in groups:
        present = [gid for gid in grp if gid in order or gid in gold_ids]
        if len(present) < 2:
            continue
        present.sort(key=lambda x: (x not in gold_ids, order.get(x, -1)))  # gold first, then input order
        keeper = present[0]
        for gid in present[1:]:
            if gid in order and gid not in drop:
                drop[gid] = keeper
    survivors = [it for it in items if it["id"] not in drop]
    dropped = [{"id": it["id"], "dup_of": drop[it["id"]]} for it in items if it["id"] in drop]
    return survivors, dropped


async def dedup_items(
    api: InferenceAPI, items: list[dict], seeds: dict, model: str = DEFAULT_MODEL
) -> tuple[list[dict], list[dict]]:
    cats = sorted({it["dimension"] for it in items})
    tasks = []
    for c in cats:
        cat_items = [it for it in items if it["dimension"] == c]
        gold = [s for s in seeds.get("seeds", []) if s["dimension"] == c]
        tasks.append(_dedup_category(api, model, c, cat_items, gold))
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
