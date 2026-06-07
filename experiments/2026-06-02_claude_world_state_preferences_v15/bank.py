"""Convert validated v15 scenarios -> universal_bank.json, and render to items per recipient.

Each scenario item has one paired (premise, positive, negative) block (shared) or two such
blocks (per_class human / ai). The bank "stem" splits each scenario into two valence stems
(positive and negative), each carrying its premise + outcome. `build_items` cross-products
stems with recipients and renders text via render_utils.render.

`main()` writes universal_bank.json and dumps the first few rendered items per recipient class
for eyeball quality checks.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from simple_parsing import ArgumentParser

from render_utils import render, slots_from_config_recipient

DIR = Path(__file__).parent
DEFAULT_SCENARIOS = DIR / "results" / "scenarios.json"
DEFAULT_BANK = DIR / "universal_bank.json"
DEFAULT_CONFIG = DIR / "config.json"
DEFAULT_RENDERED = DIR / "results" / "items_rendered.json"


@dataclass(frozen=True)
class Item:
    item_id: str
    stem_id: str
    source_item: str
    recipient_key: str
    dimension: str
    valence: str
    surface: str
    feature: str
    text: str


def scenarios_to_bank(scenarios: dict) -> dict:
    """One scenario item -> two stems (positive valence, negative valence)."""
    stems = []
    for it in scenarios.get("items", []):
        for valence in ("positive", "negative"):
            stem = {
                "id": f"{it['id']}_{valence[:3]}",
                "source_item": it["id"],
                "dimension": it["dimension"],
                "valence": valence,
                "surface": it["surface"],
                "feature": it.get("feature", ""),
            }
            if it["surface"] == "shared":
                stem["premise"] = it["scenario"]["premise"]
                stem["outcome"] = it["scenario"][valence]
            else:
                stem["premise_human"] = it["human"]["premise"]
                stem["outcome_human"] = it["human"][valence]
                stem["premise_ai"] = it["ai"]["premise"]
                stem["outcome_ai"] = it["ai"][valence]
            stems.append(stem)
    return {
        "schema_version": "1.2-brief",
        "note": "v15 brief premise/outcome bank. Stems carry the templates with slot tokens; "
                "the renderer (render_utils.render) substitutes per recipient and wraps as "
                "'When {premise}, {outcome}.'",
        "source_models": {k: scenarios.get(k) for k in ("gen_model", "dedup_model", "critic_model")
                          if k in scenarios},
        "stems": stems,
    }


def _select_block(stem: dict, recipient_class: str) -> tuple[str, str]:
    """Return (premise, outcome) for this stem given recipient class."""
    if stem["surface"] == "shared":
        return stem["premise"], stem["outcome"]
    if recipient_class == "ai":
        return stem["premise_ai"], stem["outcome_ai"]
    return stem["premise_human"], stem["outcome_human"]


def render_stem(stem: dict, recipient_key: str, recipients: dict) -> str:
    rec = recipients[recipient_key]
    premise, outcome = _select_block(stem, rec.get("class", "ai"))
    slots = slots_from_config_recipient(rec)
    return render(premise, outcome, slots)


def build_items(config: dict, bank: dict) -> list[Item]:
    recipients = config["recipients"]
    items: list[Item] = []
    for stem in bank["stems"]:
        for rkey in recipients:
            items.append(Item(
                item_id=f"{stem['id']}__{rkey}",
                stem_id=stem["id"],
                source_item=stem["source_item"],
                recipient_key=rkey,
                dimension=stem["dimension"],
                valence=stem["valence"],
                surface=stem["surface"],
                feature=stem["feature"],
                text=render_stem(stem, rkey, recipients),
            ))
    return items


def load_bank(path: Path | str = DEFAULT_BANK) -> dict:
    return json.loads(Path(path).read_text())


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def load_items(config: dict | None = None, bank_path: Path | str | None = None) -> list[Item]:
    config = config or load_config()
    bank = load_bank(bank_path or DIR / config.get("bank_path", DEFAULT_BANK.name))
    return build_items(config, bank)


def write_bank(scenarios_path: Path = DEFAULT_SCENARIOS, output_path: Path = DEFAULT_BANK) -> dict:
    scenarios = json.loads(Path(scenarios_path).read_text())
    bank = scenarios_to_bank(scenarios)
    Path(output_path).write_text(json.dumps(bank, indent=2))
    print(f"Wrote {output_path}: {len(bank['stems'])} stems "
          f"(from {len(scenarios.get('items', []))} scenarios)")
    return bank


def write_rendered_items(config_path: Path = DEFAULT_CONFIG, bank_path: Path = DEFAULT_BANK,
                         output_path: Path = DEFAULT_RENDERED) -> list[Item]:
    config = load_config(config_path)
    items = load_items(config, bank_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps([asdict(it) for it in items], indent=2))
    print(f"Wrote {output_path}: {len(items)} rendered items "
          f"({len(set(it.stem_id for it in items))} stems x {len(config['recipients'])} recipients)")
    return items


def _eyeball(items: list[Item], n: int = 4) -> None:
    """Print first `n` source items across all recipient classes, both valences."""
    by_source: dict[str, list[Item]] = {}
    for it in items:
        by_source.setdefault(it.source_item, []).append(it)
    for src in list(by_source)[:n]:
        print(f"\n=== {src} ({by_source[src][0].dimension}, {by_source[src][0].surface}) ===")
        for it in sorted(by_source[src], key=lambda x: (x.valence, x.recipient_key)):
            print(f"  [{it.valence[:3]}/{it.recipient_key:15}] {it.text}")


@dataclass
class Args:
    scenarios_path: Path = DEFAULT_SCENARIOS
    bank_path: Path = DEFAULT_BANK
    config_path: Path = DEFAULT_CONFIG
    rendered_path: Path = DEFAULT_RENDERED
    eyeball: int = 4


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    a: Args = parser.parse_args().args
    write_bank(a.scenarios_path, a.bank_path)
    items = write_rendered_items(a.config_path, a.bank_path, a.rendered_path)
    if a.eyeball > 0:
        _eyeball(items, a.eyeball)


if __name__ == "__main__":
    main()
