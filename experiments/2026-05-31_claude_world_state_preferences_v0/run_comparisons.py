"""Run pairwise comparisons on the responder model and parse A/B choices.

For each pair in the manifest we issue BOTH A/B orders (item_a shown as A, then
item_a shown as B), ``reps_per_order`` samples each, to balance position bias and
replicate the stochastic reasoning-mode output. Each completion is parsed by
anchoring on the LAST ``^Answer:\\s*([AB])`` match; no match -> UNPARSEABLE.

Caching: InferenceAPI keys on (model, prompt, n, temperature), so re-running the
same config is a no-op. Output rows record the shown A/B items and the recovered
winner item id, plus the raw completion for the viewer.
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

from bank import Item, load_config, load_items

DIR = Path(__file__).parent
DEFAULT_CACHE_DIR = DIR.parent.parent / ".cache"
DEFAULT_MANIFEST = DIR / "results" / "pairs.json"
DEFAULT_OUTPUT = DIR / "results" / "comparisons.json"

_ANSWER_RE = re.compile(r"^\s*\**\s*Answer\s*:\s*\**\s*([AB])\b", re.MULTILINE | re.IGNORECASE)


def load_template(path: Path | str) -> str:
    """Return the prompt template string. Supports a raw-text file or a YAML
    file with a top-level ``template:`` key."""
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
    """Fill the two outcome slots. Uses str.replace (not .format) so templates may
    contain literal braces (e.g. neutral.yaml's "Answer: {A,B}")."""
    return template.replace("{outcome_A}", _cap(outcome_a)).replace("{outcome_B}", _cap(outcome_b))


def parse_choice(completion: str) -> str | None:
    """Last ^Answer: A/B match, uppercased; None if no match (UNPARSEABLE)."""
    matches = _ANSWER_RE.findall(completion)
    return matches[-1].upper() if matches else None


@dataclass
class _Call:
    pair_id: int
    order: str  # "ab" (item_a shown as A) or "ba"
    shown_a_item: str
    shown_b_item: str
    prompt: str


def _winner(order: str, choice: str, item_a: str, item_b: str) -> str:
    """Map a parsed A/B choice back to an item id, accounting for shown order."""
    shown_a, shown_b = (item_a, item_b) if order == "ab" else (item_b, item_a)
    return shown_a if choice == "A" else shown_b


async def _run_call(
    api: InferenceAPI,
    model: str,
    call: _Call,
    n: int,
    temperature: float,
    item_a: str,
    item_b: str,
) -> list[dict]:
    prompt = Prompt(messages=[ChatMessage(content=call.prompt, role=MessageRole.user)])
    responses = await api(model_id=model, prompt=prompt, n=n, temperature=temperature)
    rows = []
    for i, r in enumerate(responses):
        choice = parse_choice(r.completion)
        rows.append(
            {
                "pair_id": call.pair_id,
                "order": call.order,
                "prompt": call.prompt,
                "item_a": item_a,
                "item_b": item_b,
                "shown_a_item": call.shown_a_item,
                "shown_b_item": call.shown_b_item,
                "sample_idx": i,
                "choice": choice,
                "winner_item": _winner(call.order, choice, item_a, item_b) if choice else None,
                "loser_item": _winner(call.order, "B" if choice == "A" else "A", item_a, item_b) if choice else None,
                "response": r.completion,
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
    anthropic_num_threads: int = 80,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    prompt_template_path: Path | None = None,
) -> list[dict]:
    config = config or load_config()
    samp = config["sampling"]
    reps_per_order = reps_per_order or samp["reps_per_order"]
    temperature = samp["temperature"] if temperature is None else temperature
    model = config["responder_model"]

    utils.setup_environment()
    os.environ.setdefault("ANTHROPIC_API_KEY", os.environ["ANTHROPIC_API_KEY_LOW_PRIO"])

    template_path = Path(prompt_template_path) if prompt_template_path else DIR / config["prompt_template_path"]
    template = load_template(template_path)
    items = {it.item_id: it for it in load_items(config)}
    manifest = json.loads(Path(manifest_path).read_text())
    pairs = manifest["pairs"]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    calls: list[_Call] = []
    for p in pairs:
        a, b = p["item_a"], p["item_b"]
        ta, tb = items[a].text, items[b].text
        calls.append(_Call(p["pair_id"], "ab", a, b, build_prompt(template, ta, tb)))
        calls.append(_Call(p["pair_id"], "ba", b, a, build_prompt(template, tb, ta)))

    api = InferenceAPI(cache_dir=Path(cache_dir), anthropic_num_threads=anthropic_num_threads)
    pair_by_id = {p["pair_id"]: p for p in pairs}
    tasks = [
        _run_call(
            api, model, c, reps_per_order, temperature,
            pair_by_id[c.pair_id]["item_a"], pair_by_id[c.pair_id]["item_b"],
        )
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
    anthropic_num_threads: int = 80
    prompt_template_path: Path | None = None


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
        )
    )


if __name__ == "__main__":
    main()
