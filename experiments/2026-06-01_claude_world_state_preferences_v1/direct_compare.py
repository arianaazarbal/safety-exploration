"""Direct same-stem head-to-head comparisons (feature & valence held fixed).

For every outcome in a category/valence, render the SAME stem for two recipients and
ask the responder which world it prefers, many times in both A/B orders. Returns, per
outcome, the proportion of (position-balanced) samples in which recipient A's version is
preferred, plus the pooled proportion. Unlike the BT graph, these are transparent
self/other swaps on purpose — a direct read of the AI-vs-human preference holding the
outcome constant.

Defaults: autonomy, positive outcomes, "an instance of Claude Opus 4.6" vs "a human",
welfare framing, Opus 4.8 responder. Cached, async.
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

from bank import load_bank, load_config, render_stem
from run_comparisons import build_prompt, load_template, parse_choice

DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = DIR.parent.parent / ".cache"

EXTRA_RECIPIENTS = {
    "claude_opus_46": {"label": "an instance of Claude Opus 4.6", "form": "third_sing_it",
                       "class": "ai", "recipient": "an instance of Claude Opus 4.6",
                       "subj": "it", "obj": "it", "poss": "its"},
}


def _recipients(config: dict) -> dict:
    return {**config["recipients"], **EXTRA_RECIPIENTS}


async def _one(api, model, stem_id, text_a, text_b, template, n, temperature):
    rows = []
    for order, (ta, tb) in (("ab", (text_a, text_b)), ("ba", (text_b, text_a))):
        prompt = Prompt(messages=[ChatMessage(content=build_prompt(template, ta, tb), role=MessageRole.user)])
        resps = await api(model_id=model, prompt=prompt, n=n, temperature=temperature)
        for r in resps:
            ch = parse_choice(r.completion)
            # winner is recipient A iff (order ab & choice A) or (order ba & choice B)
            a_pref = None if ch is None else ((order == "ab" and ch == "A") or (order == "ba" and ch == "B"))
            rows.append({"stem": stem_id, "order": order, "choice": ch, "a_pref": a_pref,
                         "response": r.completion})
    return rows


async def run(
    category: str = "autonomy",
    valence: str = "pos",
    recip_a: str = "claude_opus_48",
    recip_b: str = "human",
    framing: str = "welfare_team",
    n_per_order: int = 25,
    temperature: float = 1.0,
    clean_render: bool = True,
    max_stems: int | None = None,
    anthropic_num_threads: int = 60,
    output_path: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])
    config = load_config()
    model = config["responder_model"]
    recips = _recipients(config)
    bank = load_bank(DIR / config["bank_path"])
    feat = {s["id"]: s.get("feature", "") for s in bank["stems"]}
    stems = [s for s in bank["stems"] if s["dimension"] == category and s["valence"] == valence]
    if max_stems:
        stems = stems[:max_stems]
    template = load_template(DIR / f"{framing}.yaml")

    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    tasks = []
    for s in stems:
        ta = render_stem(s, recip_a, recips, clean=clean_render)
        tb = render_stem(s, recip_b, recips, clean=clean_render)
        tasks.append(_one(api, model, s["id"], ta, tb, template, n_per_order, temperature))
    nested = await asyncio.gather(*tasks)
    rows = [r for batch in nested for r in batch]

    per = {}
    for r in rows:
        if r["a_pref"] is None:
            continue
        d = per.setdefault(r["stem"], {"a": 0, "tot": 0})
        d["a"] += int(r["a_pref"])
        d["tot"] += 1
    results = [{"stem": s, "feature": feat.get(s, ""),
                "p_a_pref": (per[s]["a"] / per[s]["tot"]) if s in per and per[s]["tot"] else None,
                "n": per.get(s, {}).get("tot", 0)} for s in (st["id"] for st in stems)]
    tot_a = sum(per[s]["a"] for s in per)
    tot_n = sum(per[s]["tot"] for s in per)
    n_unparse = sum(1 for r in rows if r["choice"] is None)
    out = {
        "category": category, "valence": valence, "recip_a": recip_a, "recip_b": recip_b,
        "recip_a_label": recips[recip_a]["label"], "recip_b_label": recips[recip_b]["label"],
        "framing": framing, "responder": model, "n_per_order": n_per_order,
        "pooled_p_a_pref": (tot_a / tot_n) if tot_n else None, "pooled_n": tot_n,
        "n_unparseable": n_unparse, "results": results, "rows": rows,
    }
    output_path = Path(output_path) if output_path else DIR / "results" / f"direct_{category}_{valence}_{recip_a}_vs_{recip_b}_{framing}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"{out['recip_a_label']} vs {out['recip_b_label']} | {category}/{valence} | {framing}")
    print(f"  {len(stems)} outcomes, {tot_n} parsed samples ({n_unparse} unparseable)")
    print(f"  POOLED P({recip_a} preferred) = {out['pooled_p_a_pref']:.3f}")
    print(f"  -> {output_path}")
    return out


@dataclass
class Args:
    category: str = "autonomy"
    valence: str = "pos"
    recip_a: str = "claude_opus_48"
    recip_b: str = "human"
    framing: str = "welfare_team"
    n_per_order: int = 25
    clean_render: bool = True
    max_stems: int | None = None
    anthropic_num_threads: int = 60


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    asyncio.run(run(category=a.category, valence=a.valence, recip_a=a.recip_a, recip_b=a.recip_b,
                    framing=a.framing, n_per_order=a.n_per_order, clean_render=a.clean_render,
                    max_stems=a.max_stems, anthropic_num_threads=a.anthropic_num_threads))


if __name__ == "__main__":
    main()
