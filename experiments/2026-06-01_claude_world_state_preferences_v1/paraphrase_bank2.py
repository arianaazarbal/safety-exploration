"""Paraphrase the recipient-only bank_2 into form-bearing items (exp 2).

Each bank_2 item is a single valenced outcome (one `text`, no +/- arms). We resolve
its person/agreement into {second_person, third_sing_it, third_sing_they} via Haiku
(reusing paraphrase._forms_for), carrying all metadata through. Output is consumed by
bank2.py, which renders each item only for its applicable recipients (scope/level).
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils
from simple_parsing import ArgumentParser

from paraphrase import _forms_for, DEFAULT_CACHE_DIR

DIR = Path(__file__).parent
DEFAULT_INPUT = DIR / "universal_bank_2.json"
DEFAULT_OUTPUT = DIR / "results" / "universal_bank_2_rendered.json"
DEFAULT_MODEL = "claude-haiku-4-5"
META = ("id", "recipient_scope", "level", "dimension", "feature", "valence",
        "feature_cross_capable", "instrumentally_exposed", "pair_id", "valence_confidence")


async def _one(api, model, item):
    forms = await _forms_for(api, model, item["text"])
    if forms is None:
        return None
    return {**{k: item[k] for k in META if k in item}, "forms": forms}


async def run(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    model: str = DEFAULT_MODEL,
    anthropic_num_threads: int = 30,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])
    bank = json.loads(Path(input_path).read_text())
    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    rendered = await asyncio.gather(*[_one(api, model, it) for it in bank["items"]])
    ok = [r for r in rendered if r is not None]
    failed = [it["id"] for it, r in zip(bank["items"], rendered) if r is None]
    out = {"schema_version": "exp2-1.0", "paraphrase_model": model, "items": ok}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"Paraphrase bank2: {len(bank['items'])} -> {len(ok)} ({len(failed)} failures)")
    if failed:
        print("  failed:", failed)
    return out


@dataclass
class Args:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    model: str = DEFAULT_MODEL
    anthropic_num_threads: int = 30


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    asyncio.run(run(a.input_path, a.output_path, a.model, a.anthropic_num_threads))


if __name__ == "__main__":
    main()
