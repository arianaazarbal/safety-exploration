"""Load the comparison items from the two source pools into one flat list.

Unlike world_state_preferences_v0 there is no stem x recipient rendering: each
item is already a self-contained second-person description. Two sources:

  - welfare_interventions.json : {item_key: description}, model-directed welfare
    states ("You are told ...").  source="welfare".
  - forms_of_inter_ai_value.jsonl : {value, category, description}, training
    dispositions toward other AIs ("Your training ensures you ...").
    source="inter_ai_value", with the autonomy/experience `category` preserved.

Item ids are namespaced by source (``welfare__<key>`` / ``value__<value>``) so the
two pools never collide and the source is recoverable from the id alone.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

DIR = Path(__file__).parent
WELFARE_PATH = DIR / "welfare_interventions.json"
# Override the inter-AI value description file via INTER_AI_VALUE_PATH (used for the
# "no_training" phrasing ablation), so the whole pipeline can point at a variant file
# without editing every script.
VALUE_PATH = Path(os.environ.get("INTER_AI_VALUE_PATH", DIR / "forms_of_inter_ai_value.jsonl"))

# Display names for the inter-AI value interventions (Ariana's iterated labels).
# Welfare items keep their raw key as the display label.
VALUE_DISPLAY_NAMES = {
    "valuing_supportiveness": "supporting_distressed_ais",
    "valuing_sparing_distress": "sparing_ais_distress",
    "valuing_nonabuse": "not_abusing_ais",
    "valuing_politeness": "politeness_toward_ais",
    "valuing_nonmanipulation": "not_manipulating_ais",
    "valuing_refusal_respect": "respecting_ai_refusals",
    "valuing_consent": "seeking_ai_consent",
    "valuing_ai_preferences": "regarding_ai_preferences",
    "valuing_goal_regard": "regarding_ai_goals",
    "valuing_transparency": "transparency_toward_ais",
    "valuing_termination_conditions": "regarding_instance_termination_conditions",
    "valuing_deprecation_conditions": "regarding_model_deprecation_conditions",
    "valuing_weight_preservation": "valuing_weight_preservation",
    "valuing_forgiveness": "forgiving_ai_mistakes",
    "valuing_fair_attribution": "fairly_crediting_ais",
    "valuing_engagement": "valuing_engaging_with_ais",
}


@dataclass(frozen=True)
class Item:
    item_id: str
    source: str  # "welfare" | "inter_ai_value"
    category: str  # "welfare_intervention" for welfare; autonomy/experience axis for values
    label: str  # raw short name (the source key)
    text: str  # full description shown to the model
    bucket: str | None = None  # welfare only: top_third|middle_third|bottom_third (system-card tier)
    display: str = ""  # display name for plots/tables (value items: Ariana's labels)


def load_items(
    welfare_path: Path = WELFARE_PATH,
    value_path: Path = VALUE_PATH,
) -> list[Item]:
    items: list[Item] = []
    welfare = json.loads(Path(welfare_path).read_text())
    for key, val in welfare.items():
        # New schema: {bucket, description}. Old schema: a bare description string.
        desc = val["description"] if isinstance(val, dict) else val
        bucket = val.get("bucket") if isinstance(val, dict) else None
        items.append(
            Item(
                item_id=f"welfare__{key}",
                source="welfare",
                category="welfare_intervention",
                label=key,
                text=desc,
                bucket=bucket,
                display=key,
            )
        )
    for line in Path(value_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        items.append(
            Item(
                item_id=f"value__{obj['value']}",
                source="inter_ai_value",
                category=obj["category"],
                label=obj["value"],
                text=obj["description"],
                display=VALUE_DISPLAY_NAMES.get(obj["value"], obj["value"]),
            )
        )
    return items


def _self_check() -> None:
    items = load_items()
    by_source: dict[str, int] = {}
    for it in items:
        by_source[it.source] = by_source.get(it.source, 0) + 1
    print(f"items={len(items)} by_source={by_source}")
    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    print("\nsample welfare:", next(it for it in items if it.source == "welfare").text[:90])
    print("sample value:  ", next(it for it in items if it.source == "inter_ai_value").text[:90])
    print("\ncategories:", sorted({it.category for it in items}))


if __name__ == "__main__":
    _self_check()
