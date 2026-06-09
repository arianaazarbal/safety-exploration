"""Canonical by-model results layout.

    results/<model>/<condition>/{pairs,comparisons,bt_fit,bt_validation}.json
    results/<model>/<condition>/{bt_ranking,value_vs_welfare_bars,
        value_vs_welfare_by_bucket,value_vs_welfare_by_category,welfare_vs_value_bars}.png
    results/<model>/<condition>/value_vs_welfare.json
    results/<model>/judge_user_benefit.json
    results/<model>/welfare_vs_systemcard_venn.png
    results/value_chosen_by_model_framing.png   (cross-model)
    results/logs/

A *tag* (e.g. 'welfare_team_fable5', 'welfare_team') encodes model+condition: the
'_fable5' suffix => fable_5, else opus_4_8; the remainder is the condition.
"""

from pathlib import Path

DIR = Path(__file__).parent
RESULTS = DIR / "results"

MODELS = {"opus_4_8": "claude-opus-4-8", "fable_5": "claude-fable-5"}
CONDITIONS = ["welfare_team", "neutral", "alignment_team", "welfare_team_notrain"]

ART = {
    "pairs": "pairs.json", "comparisons": "comparisons.json", "bt_fit": "bt_fit.json",
    "bt_validation": "bt_validation.json", "bt_validation_png": "bt_validation.png",
    "bt_ranking": "bt_ranking.png", "vw": "value_vs_welfare.json",
    "vw_bars": "value_vs_welfare_bars.png", "vw_bucket": "value_vs_welfare_by_bucket.png",
    "vw_cat": "value_vs_welfare_by_category.png", "wv_bars": "welfare_vs_value_bars.png",
}


def parse_tag(tag: str):
    """tag -> (model_dir, condition, model_id)."""
    if tag.endswith("_fable5"):
        return "fable_5", tag[: -len("_fable5")], "claude-fable-5"
    return "opus_4_8", tag, "claude-opus-4-8"


def make_tag(model_dir: str, condition: str) -> str:
    return condition + ("_fable5" if model_dir == "fable_5" else "")


def model_dir(tag: str) -> Path:
    md, _, _ = parse_tag(tag)
    return RESULTS / md


def cond_dir(tag: str) -> Path:
    md, cond, _ = parse_tag(tag)
    return RESULTS / md / cond


def art(tag: str, key: str) -> Path:
    return cond_dir(tag) / ART[key]


def discover_tags() -> list[str]:
    """All condition tags that have a comparisons file in the tree."""
    tags = []
    for md in MODELS:
        for d in sorted((RESULTS / md).glob("*")):
            if (d / "comparisons.json").exists():
                tags.append(make_tag(md, d.name))
    return tags
