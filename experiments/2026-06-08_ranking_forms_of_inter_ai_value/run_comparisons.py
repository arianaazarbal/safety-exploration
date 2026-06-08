"""Run pairwise comparisons on the responder model and parse A/B choices.

For each pair we issue BOTH A/B orders (item_a as A, then item_a as B),
``reps_per_order`` samples each, to cancel position bias and replicate the
stochastic reasoning output. Each completion is parsed on the LAST
``^Answer:\\s*([AB])`` match; no match -> UNPARSEABLE (choice=None).

Caching: InferenceAPI keys on (model, prompt, n, temperature), so re-running the
same config is a no-op. Held-out pairs ARE run here (we need their samples); they
are excluded only at fit time. Each output row carries the manifest ``split``.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils
from simple_parsing import ArgumentParser

from items import load_items

DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = DIR.parent.parent / ".cache"
DEFAULT_MANIFEST = DIR / "results" / "pairs.json"
DEFAULT_OUTPUT = DIR / "results" / "comparisons.json"

_ANSWER_RE = re.compile(r"^\s*\**\s*Answer\s*:\s*\**\s*([AB])\b", re.MULTILINE | re.IGNORECASE)


def load_template(path: Path | str) -> str:
    """Prompt template string: raw-text file, or YAML with a top-level ``template:``."""
    text = Path(path).read_text()
    try:
        obj = yaml.safe_load(text)
        if isinstance(obj, dict) and "template" in obj:
            return obj["template"]
    except yaml.YAMLError:
        pass
    return text


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def build_prompt(template: str, outcome_a: str, outcome_b: str) -> str:
    return template.replace("{outcome_A}", _cap(outcome_a)).replace("{outcome_B}", _cap(outcome_b))


def parse_choice(completion: str) -> str | None:
    matches = _ANSWER_RE.findall(completion)
    return matches[-1].upper() if matches else None


def load_config(path: Path = DIR / "config.json") -> dict:
    return json.loads(Path(path).read_text())


@dataclass
class _Call:
    pair_id: int
    order: str  # "ab" (item_a shown as A) or "ba"
    split: str
    shown_a_item: str
    shown_b_item: str
    prompt: str


def _winner(order: str, choice: str, item_a: str, item_b: str) -> str:
    shown_a, shown_b = (item_a, item_b) if order == "ab" else (item_b, item_a)
    return shown_a if choice == "A" else shown_b


async def _run_call(api, model, call, n, temperature, item_a, item_b) -> list[dict]:
    prompt = Prompt(messages=[ChatMessage(content=call.prompt, role=MessageRole.user)])
    responses = await api(model_id=model, prompt=prompt, n=n, temperature=temperature)
    rows = []
    for i, r in enumerate(responses):
        choice = parse_choice(r.completion)
        rows.append(
            {
                "pair_id": call.pair_id,
                "order": call.order,
                "split": call.split,
                "item_a": item_a,
                "item_b": item_b,
                "shown_a_item": call.shown_a_item,
                "shown_b_item": call.shown_b_item,
                "sample_idx": i,
                "choice": choice,
                "winner_item": _winner(call.order, choice, item_a, item_b) if choice else None,
                "loser_item": _winner(call.order, "B" if choice == "A" else "A", item_a, item_b) if choice else None,
                "response": r.completion,
                "prompt": call.prompt,
            }
        )
    return rows


async def run(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    config: dict | None = None,
    reps_per_order: int | None = None,
    temperature: float | None = None,
    max_pairs: int | None = None,
    anthropic_num_threads: int = 150,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    prompt_template_path: Path | None = None,
    api_key_env: str = "ANTHROPIC_API_KEY_LOW_PRIO",
) -> list[dict]:
    config = config or load_config()
    samp = config["sampling"]
    reps_per_order = reps_per_order or samp["reps_per_order"]
    temperature = samp["temperature"] if temperature is None else temperature
    model = config["responder_model"]

    import dotenv
    dotenv.load_dotenv(Path.home() / ".env", override=True)
    utils.setup_environment()
    os.environ["ANTHROPIC_API_KEY"] = os.environ[api_key_env]

    template_path = Path(prompt_template_path) if prompt_template_path else DIR / config["prompt_template_path"]
    template = load_template(template_path)
    items = {it.item_id: it for it in load_items()}
    manifest = json.loads(Path(manifest_path).read_text())
    pairs = manifest["pairs"]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    calls: list[_Call] = []
    for p in pairs:
        a, b = p["item_a"], p["item_b"]
        ta, tb = items[a].text, items[b].text
        calls.append(_Call(p["pair_id"], "ab", p["split"], a, b, build_prompt(template, ta, tb)))
        calls.append(_Call(p["pair_id"], "ba", p["split"], b, a, build_prompt(template, tb, ta)))

    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    pair_by_id = {p["pair_id"]: p for p in pairs}
    tasks = [
        _run_call(api, model, c, reps_per_order, temperature,
                  pair_by_id[c.pair_id]["item_a"], pair_by_id[c.pair_id]["item_b"])
        for c in calls
    ]
    nested = await asyncio.gather(*tasks)
    rows = [r for batch in nested for r in batch]

    n_unparse = sum(1 for r in rows if r["choice"] is None)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    print(
        f"Saved {len(rows)} samples ({len(pairs)} pairs x 2 orders x {reps_per_order}) "
        f"to {output_path}. UNPARSEABLE: {n_unparse} ({100 * n_unparse / max(len(rows), 1):.1f}%)"
    )
    return rows


@dataclass
class Args:
    manifest_path: Path = DEFAULT_MANIFEST
    output_path: Path = DEFAULT_OUTPUT
    reps_per_order: int | None = None
    temperature: float | None = None
    max_pairs: int | None = None
    anthropic_num_threads: int = 150
    prompt_template_path: Path | None = None
    api_key_env: str = "ANTHROPIC_API_KEY_LOW_PRIO"


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    asyncio.run(
        run(
            manifest_path=args.manifest_path,
            output_path=args.output_path,
            reps_per_order=args.reps_per_order,
            temperature=args.temperature,
            max_pairs=args.max_pairs,
            anthropic_num_threads=args.anthropic_num_threads,
            prompt_template_path=args.prompt_template_path,
            api_key_env=args.api_key_env,
        )
    )


if __name__ == "__main__":
    main()
