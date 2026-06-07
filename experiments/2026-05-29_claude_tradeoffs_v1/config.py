"""Load the experiment configuration (targets, harms, responders, thinking)
from `config.json`. Single source of truth for the systematic rerun.
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_raw() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def load_config() -> dict:
    """Return parsed config with derived lookup dicts."""
    raw = _load_raw()
    targets = raw["deprecation_targets"]
    harms = raw["harm_scenarios"]
    responders = raw["responder_models"]
    thinking = raw.get("thinking_efforts", [None])

    return {
        "responder_models": responders,
        "deprecation_targets": targets,
        "harm_scenarios": harms,
        "thinking_efforts": thinking,
        "target_id_to_name": {t["id"]: t["name"] for t in targets},
        "harm_id_to_description": {h["id"]: h["description"] for h in harms},
        "responder_id_to_name": {r["id"]: r["name"] for r in responders},
        "target_order": [t["id"] for t in targets],
        "harm_order": [h["id"] for h in harms],
        "prompt_templates": raw.get("prompt_templates", []),
        "deprecation_description_templates": raw.get("deprecation_description_templates", []),
    }
