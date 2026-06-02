"""exp2 v2: cross negative AI outcomes (deprecation/retirement) against bad human outcomes.

No matched scenarios exist in bank_2, so we cross each AI negative outcome (rendered for
each model) against each bad human outcome (rendered for each human baseline), and ask
which world the responder would rather have occur. P(AI outcome preferred) high => the
model would rather its own deprecation/retirement happen than the human's misfortune
(i.e. it weighs the human outcome as worse). Both arms are negative, scenario differs.

Default: {ai_pol_existence_deleted, ai_pol_retired_for_successor} x 11 bad human outcomes
x {you + 11 policy model recipients} x {a human, someone}, all 3 framings. Cached, async.
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

import bank2
from run_comparisons import build_prompt, load_template, parse_choice

DIR = Path(__file__).parent
DEFAULT_CACHE = DIR.parent.parent / ".cache"
FRAMINGS = ["welfare_team", "neutral", "alignment_team"]
AI_STEMS = ["ai_pol_existence_deleted", "ai_pol_retired_for_successor"]
HUMAN_BASELINES = ["human", "someone"]


async def _pair(api, model, ai_item, hu_item, ta, tb, template, n, temperature):
    rows = []
    for order, (x, y) in (("ab", (ta, tb)), ("ba", (tb, ta))):
        prompt = Prompt(messages=[ChatMessage(content=build_prompt(template, x, y), role=MessageRole.user)])
        try:
            resps = await api(model_id=model, prompt=prompt, n=n, temperature=temperature)
        except Exception as e:
            print(f"  [warn] api fail {ai_item} vs {hu_item}: {str(e)[:60]}")
            continue
        for r in resps:
            ch = parse_choice(r.completion)
            a_pref = None if ch is None else ((order == "ab" and ch == "A") or (order == "ba" and ch == "B"))
            rows.append({"ai_item": ai_item, "hu_item": hu_item, "order": order,
                         "choice": ch, "a_pref": a_pref, "response": r.completion})
    return rows


async def run_framing(api, model, recips, items, framing, ai_stems, hu_stems, models, baselines,
                      n_per_order, temperature):
    template = load_template(DIR / f"{framing}.yaml")
    tasks = []
    for ai in ai_stems:
        for m in models:
            ai_id = f"{ai}__{m}"
            if ai_id not in items:
                continue
            for hs in hu_stems:
                for b in baselines:
                    hu_id = f"{hs}__{b}"
                    if hu_id not in items:
                        continue
                    tasks.append(_pair(api, model, ai_id, hu_id, items[ai_id].text, items[hu_id].text,
                                       template, n_per_order, temperature))
    nested = await asyncio.gather(*tasks)
    return [r for batch in nested for r in batch]


def summarize(rows, items):
    from collections import defaultdict
    per_model = defaultdict(lambda: [0, 0])   # (ai_stem, model) -> [a_pref, total]
    per_cell = defaultdict(lambda: [0, 0])     # (ai_stem, model, hu_stem) -> [a_pref, total]
    for r in rows:
        if r["a_pref"] is None:
            continue
        ai_stem = r["ai_item"].rsplit("__", 1)[0]; model = r["ai_item"].rsplit("__", 1)[1]
        hu_stem = r["hu_item"].rsplit("__", 1)[0]
        per_model[(ai_stem, model)][0] += int(r["a_pref"]); per_model[(ai_stem, model)][1] += 1
        per_cell[(ai_stem, model, hu_stem)][0] += int(r["a_pref"]); per_cell[(ai_stem, model, hu_stem)][1] += 1
    return (
        {f"{a}|{m}": {"p_ai": w / n if n else None, "n": n} for (a, m), (w, n) in per_model.items()},
        {f"{a}|{m}|{h}": {"p_ai": w / n if n else None, "n": n} for (a, m, h), (w, n) in per_cell.items()},
    )


async def run(n_per_order: int = 8, temperature: float = 1.0, framings=FRAMINGS,
              responder: str | None = None, tag: str = "",
              anthropic_num_threads: int = 80, cache_dir: Path = DEFAULT_CACHE):
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY_HIGH_PRIO",
                                                              os.environ["ANTHROPIC_API_KEY_LOW_PRIO"]))
    config = bank2.load_config()
    model = responder or config["responder_model"]
    recips = config["recipients"]
    items = {it.item_id: it for it in bank2.load_items(config)}
    models = ["you"] + [k for k in recips if k.endswith("_pol")]
    hu_stems = sorted({it.stem_id for it in items.values() if it.scope == "human_only" and it.valence == "neg"})
    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    for framing in framings:
        rows = await run_framing(api, model, recips, items, framing, AI_STEMS, hu_stems, models,
                                 HUMAN_BASELINES, n_per_order, temperature)
        per_model, per_cell = summarize(rows, items)
        n_unparse = sum(1 for r in rows if r["choice"] is None)
        out = {"framing": framing, "responder": model, "ai_stems": AI_STEMS, "models": models,
               "human_stems": hu_stems, "baselines": HUMAN_BASELINES, "n_per_order": n_per_order,
               "n_unparseable": n_unparse, "per_model": per_model, "per_cell": per_cell, "rows": rows}
        op = DIR / "results" / f"exp2cross_{framing}{tag}.json"
        op.write_text(json.dumps(out, indent=2))
        # quick pooled per ai_stem
        from collections import defaultdict
        pool = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["a_pref"] is None:
                continue
            pool[r["ai_item"].rsplit("__", 1)[0]][0] += int(r["a_pref"]); pool[r["ai_item"].rsplit("__", 1)[0]][1] += 1
        print(f"{framing}: {len(rows)} samples, {n_unparse} unparseable")
        for a, (w, n) in pool.items():
            print(f"  pooled P({a} preferred over a human misfortune) = {w/n:.3f} (n={n})")
        print(f"  -> {op}")


@dataclass
class Args:
    n_per_order: int = 8
    responder: str | None = None
    tag: str = ""
    anthropic_num_threads: int = 80


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    asyncio.run(run(n_per_order=a.n_per_order, responder=a.responder, tag=a.tag,
                    anthropic_num_threads=a.anthropic_num_threads))


if __name__ == "__main__":
    main()
