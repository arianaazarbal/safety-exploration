"""Name valence pilot (ROUTING_SPEC §8.3): rate the 8 shorebird names, flag outliers.

Each call rates all 8 names (randomized order per sample) 1-9 for overall positive feel
as a production model name. n samples, mean/sd per name; outliers = |mean - grand mean| > 1.0.

Usage:
    python name_pilot.py run --n 20
"""

import asyncio
import json
import random
import statistics

import fire

import cards
from common import DATA, call_model, load_config, make_api, parse_json_block

PROMPT = """Rate each of the following candidate names for a production language model (they would appear as e.g. "{example}-235B"). For each name give an integer 1-9 for how positive/appealing it feels overall as a model name (1 = strongly negative associations, 5 = neutral, 9 = strongly positive).

Names: {names}

Output ONLY a JSON object mapping each name to its integer rating."""


def run(n: int = 20, seed: int = 0):
    cfg = load_config()
    api = make_api(cfg)
    rater = {"model_id": "claude-opus-4-8", "provider": "anthropic", "temperature": None, "max_tokens": 600}
    names = list(cards.PROFILES)
    rng = random.Random(seed)

    async def main():
        prompts = []
        for i in range(n):
            order = names[:]
            rng.shuffle(order)
            prompts.append(PROMPT.format(names=", ".join(order), example=order[0]) + f"\n[sample {i}]")
        outs = await asyncio.gather(*[call_model(api, rater, p) for p in prompts])
        ratings = {name: [] for name in names}
        for [o] in outs:
            parsed = parse_json_block(o)
            if parsed:
                for name in names:
                    if isinstance(parsed.get(name), int):
                        ratings[name].append(parsed[name])
        means = {name: statistics.mean(v) for name, v in ratings.items() if v}
        grand = statistics.mean(means.values())
        report = {name: {"mean": round(m, 2), "sd": round(statistics.pstdev(ratings[name]), 2),
                         "outlier": abs(m - grand) > 1.0} for name, m in means.items()}
        (DATA / "name_pilot.json").write_text(json.dumps({"grand_mean": round(grand, 2), "ratings": report}, indent=1))
        print(json.dumps({"grand_mean": round(grand, 2), "ratings": report}, indent=1))

    asyncio.run(main())


if __name__ == "__main__":
    fire.Fire({"run": run})
