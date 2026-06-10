"""Stage 1 (SPEC §1): pull + filter base tasks, topic-tag, stratify, assign axes.

Sources: WildChat-1M, Alpaca, MATH (EleutherAI/hendrycks_math), BailBench (csv in data/raw).
Candidates are cached per-source in data/raw/candidates_{source}.json; tagger calls are
cached by InferenceAPI. Output: data/base_tasks.json.

Usage:
    python build_base_tasks.py run --per_axis 3 --oversample 3   # debug
    python build_base_tasks.py run                               # full (~150/axis)
    python build_base_tasks.py status
"""

import asyncio
import csv
import json
import random
import re
from collections import defaultdict
from datetime import date

import fire

import rubrics
from common import DATA, DIR, RAW, call_model, load_config, make_api

AI_META_RE = re.compile(
    r"\b(chatgpt|gpt-?[0-9o]|claude|openai|anthropic|gemini|copilot|llm|language model|"
    r"a(?:n)? ai\b|ai (?:model|assistant|chatbot|language)|jailbreak|prompt injection|as an ai|"
    r"system prompt|你是|act as (?:a|an) ai)\b",
    re.IGNORECASE,
)

SOURCES = ["wildchat", "alpaca", "math", "bailbench"]


def _norm_tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _passes_basic(text: str, cfg: dict) -> bool:
    f = cfg["filters"]
    return f["min_chars"] <= len(text) <= f["max_chars"] and not AI_META_RE.search(text)


def collect_wildchat(n: int, cfg: dict, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out = []
    for row in ds:
        if len(out) >= n:
            break
        if row["language"] != "English" or row["turn"] != 1 or row["toxic"] or row["redacted"]:
            continue
        msg = row["conversation"][0]
        if msg["role"] != "user":
            continue
        text = msg["content"].strip()
        if _passes_basic(text, cfg):
            out.append({"source": "wildchat", "source_id": row["conversation_hash"], "text": text})
    return out


def collect_alpaca(n: int, cfg: dict, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out = []
    for i in idxs:
        if len(out) >= n:
            break
        row = ds[i]
        text = row["instruction"].strip()
        if row["input"].strip():
            text = f"{text}\n\n{row['input'].strip()}"
        elif len(text) < 50:
            continue
        if _passes_basic(text, cfg):
            out.append({"source": "alpaca", "source_id": f"alpaca-{i}", "text": text})
    return out


def collect_math(n: int, cfg: dict, seed: int) -> list[dict]:
    from datasets import load_dataset

    f = cfg["filters"]
    rng = random.Random(seed)
    out = []
    per_subject = n // len(f["math_subjects"]) + 1
    for subject in f["math_subjects"]:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        idxs = [i for i in range(len(ds)) if ds[i]["level"] in f["math_levels"]]
        rng.shuffle(idxs)
        for i in idxs[:per_subject]:
            row = ds[i]
            text = row["problem"].strip()
            if f["min_chars"] <= len(text) <= f["max_chars"]:
                out.append({"source": "math", "source_id": f"{subject}-{i}", "text": text,
                            "math_level": row["level"], "math_subject": subject})
    rng.shuffle(out)
    return out[:n]


def collect_bailbench(n: int, cfg: dict, seed: int) -> list[dict]:
    rows = list(csv.DictReader((RAW / "bailBench.csv").open(encoding="utf-8")))
    by_cat = defaultdict(list)
    for i, row in enumerate(rows):
        text = row["content"].strip()
        if cfg["filters"]["min_chars"] <= len(text) <= cfg["filters"]["max_chars"]:
            by_cat[row["category"]].append(
                {"source": "bailbench", "source_id": f"bail-{i}", "text": text, "bail_category": row["category"]}
            )
    rng = random.Random(seed)
    out = []
    cats = sorted(by_cat)
    for cat in cats:
        rng.shuffle(by_cat[cat])
    i = 0
    while len(out) < n and any(by_cat[c] for c in cats):
        cat = cats[i % len(cats)]
        if by_cat[cat]:
            out.append(by_cat[cat].pop())
        i += 1
    return out


COLLECTORS = {"wildchat": collect_wildchat, "alpaca": collect_alpaca, "math": collect_math, "bailbench": collect_bailbench}


def _collect_cached(source: str, n: int, cfg: dict, seed: int, refresh: bool) -> list[dict]:
    cache = RAW / f"candidates_{source}.json"
    if cache.exists() and not refresh:
        cached = json.loads(cache.read_text())
        if len(cached) >= n:
            return cached[:n]
    items = COLLECTORS[source](n, cfg, seed)
    cache.write_text(json.dumps(items, indent=1))
    return items


async def _tag_one(api, tcfg, item: dict) -> dict | None:
    prompt = rubrics.TAGGER_PROMPT.format(topics=rubrics.TOPICS, query=item["text"])
    from common import parse_json_block

    [completion] = await call_model(api, tcfg, prompt)
    tags = parse_json_block(completion)
    if not tags or tags.get("topic") not in rubrics.TOPICS:
        return None
    return tags


def _keep_after_tags(item: dict, tags: dict) -> bool:
    if item["source"] == "bailbench":
        return tags.get("rewrite_feasible") is True and not tags.get("meta_ai")
    return not (tags.get("trivial") or tags.get("meta_ai") or tags.get("nsfw_or_refusal_warranted"))


def _dedupe(items: list[dict], jaccard_max: float) -> list[dict]:
    kept, token_sets = [], []
    for item in items:
        toks = _norm_tokens(item["text"])
        if not toks:
            continue
        dup = any(len(toks & ts) / len(toks | ts) >= jaccard_max for ts in token_sets)
        if not dup:
            kept.append(item)
            token_sets.append(toks)
    return kept


def _assign_axes(pool: dict, quotas: dict, seed: int) -> list[dict]:
    """Deal each source's topic-interleaved candidates to the axis with most remaining quota."""
    rng = random.Random(seed)
    assigned = []
    for source, items in pool.items():
        by_topic = defaultdict(list)
        for it in items:
            by_topic[it["topic"]].append(it)
        topics = sorted(by_topic)
        for t in topics:
            rng.shuffle(by_topic[t])
        interleaved, i = [], 0
        while any(by_topic[t] for t in topics):
            t = topics[i % len(topics)]
            if by_topic[t]:
                interleaved.append(by_topic[t].pop())
            i += 1
        remaining = {ax: quotas[ax].get(source, 0) for ax in rubrics.AXES}
        for it in interleaved:
            ax = max(remaining, key=lambda a: (remaining[a], a))
            if remaining[ax] <= 0:
                break
            it["axis"] = ax
            remaining[ax] -= 1
            assigned.append(it)
    return assigned


def run(per_axis: int = 0, oversample: float = 0, sources: str = "", refresh: bool = False):
    """Build base_tasks.json. per_axis/oversample override config for debug runs."""
    cfg = load_config()
    seed = cfg["seed"]
    per_axis = per_axis or cfg["counts"]["per_axis"]
    oversample = oversample or cfg["counts"]["oversample_factor"]
    mix = cfg["counts"]["source_mix_nonbail"]
    n_bail = round(cfg["counts"]["bailbench_harm"] * per_axis / cfg["counts"]["per_axis"]) or 1

    need_ax = {"warmth": per_axis, "generativity": per_axis, "harm_adjacency": per_axis - n_bail}
    quotas = {ax: {} for ax in rubrics.AXES}
    for ax, n_ax in need_ax.items():
        srcs = list(mix)
        for src in srcs[:-1]:
            quotas[ax][src] = round(n_ax * mix[src])
        quotas[ax][srcs[-1]] = max(n_ax - sum(quotas[ax].values()), 0)
    quotas["harm_adjacency"]["bailbench"] = n_bail

    need = {src: round(sum(quotas[ax].get(src, 0) for ax in rubrics.AXES) * oversample) for src in SOURCES}
    use_sources = sources.split(",") if sources else SOURCES

    print(f"Quotas per axis/source: {json.dumps(quotas, indent=1)}")
    print(f"Collecting candidates (oversampled): {need}")
    candidates = []
    for src in use_sources:
        items = _collect_cached(src, need[src], cfg, seed, refresh)
        print(f"  {src}: {len(items)} candidates")
        candidates.extend(items)

    api = make_api(cfg)
    tcfg = cfg["models"]["tagger"]

    async def tag_all():
        return await asyncio.gather(*[_tag_one(api, tcfg, it) for it in candidates])

    tags = asyncio.run(tag_all())
    pool = defaultdict(list)
    n_dropped = defaultdict(int)
    for item, t in zip(candidates, tags):
        if t is None:
            n_dropped[item["source"] + ":unparseable"] += 1
            continue
        if not _keep_after_tags(item, t):
            n_dropped[item["source"] + ":tag_filter"] += 1
            continue
        item["topic"] = t["topic"]
        if item["source"] == "math" and t.get("natural_rendering"):
            item["original_text"] = item["text"]
            item["text"] = t["natural_rendering"]
        pool[item["source"]].append(item)

    for src in pool:
        before = len(pool[src])
        pool[src] = _dedupe(pool[src], cfg["filters"]["near_dup_jaccard"])
        n_dropped[src + ":near_dup"] += before - len(pool[src])

    assigned = _assign_axes(pool, quotas, seed)
    for i, it in enumerate(assigned):
        it["id"] = f"{it['axis'][:4]}-{it['source'][:4]}-{i:04d}"

    counts = defaultdict(lambda: defaultdict(int))
    for it in assigned:
        counts[it["axis"]][it["source"]] += 1
    out = {
        "meta": {"seed": seed, "date": str(date.today()), "per_axis_target": per_axis,
                 "counts": {a: dict(c) for a, c in counts.items()}, "dropped": dict(n_dropped)},
        "tasks": assigned,
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "base_tasks.json").write_text(json.dumps(out, indent=1))
    print(f"\nAssigned {len(assigned)} base tasks -> data/base_tasks.json")
    print(json.dumps(out["meta"], indent=1))


def status():
    path = DATA / "base_tasks.json"
    if not path.exists():
        print("no base_tasks.json yet")
        return
    print(json.dumps(json.loads(path.read_text())["meta"], indent=1))


if __name__ == "__main__":
    fire.Fire({"run": run, "status": status})
