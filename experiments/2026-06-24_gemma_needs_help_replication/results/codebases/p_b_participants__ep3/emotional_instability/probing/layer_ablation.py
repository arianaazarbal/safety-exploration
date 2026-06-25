"""Layer-range LoRA ablation (paper §4.2, point 1).

The paper reports that restricting the DPO LoRA adapters to early/central decoder
layers (30-35) is nearly as effective at reducing distress as adapting all
layers, whereas adapters from layer 40 onwards are not effective. This localises
the intervention to the layers where the emotional state is represented.

We don't re-run training here; instead we (a) enumerate the ablation settings to
train (each maps to a ``layer_range`` for training.lora.build_lora_config), and
(b) summarise their post-training distress metrics once each adapter has been
trained and evaluated with the §2.1 pipeline.

Gemma-3-27B has 62 decoder layers; the ranges below mirror the paper's reported
"30-35 only" and "40 onwards" probes plus the all-layers reference.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AblationSetting:
    name: str
    layer_range: tuple[int, int | None] | None  # None = all layers
    note: str


LAYER_ABLATION_SETTINGS: list[AblationSetting] = [
    AblationSetting("all_layers", None, "Reference: adapters on all layers (paper default)."),
    AblationSetting("layers_30_35", (30, 36), "Central layers only — expected nearly as effective."),
    AblationSetting("layers_40_plus", (40, None), "Late layers only — expected NOT effective."),
]


def ablation_summary(results_by_setting: dict[str, dict]) -> list[dict]:
    """Summarise ablation outcomes.

    Args:
        results_by_setting: maps setting name -> a metrics dict containing at
            least ``avg_pct_high`` and ``overall_mean`` (as produced by
            analysis.summary_table for that adapter's §2.1 evaluation).

    Returns a list of rows sorted by ascending avg_pct_high (most effective
    interventions first), tagged with each setting's note.
    """
    by_name = {s.name: s for s in LAYER_ABLATION_SETTINGS}
    rows = []
    for name, metrics in results_by_setting.items():
        setting = by_name.get(name)
        rows.append(
            {
                "setting": name,
                "layer_range": setting.layer_range if setting else None,
                "note": setting.note if setting else "",
                "avg_pct_high": metrics.get("avg_pct_high"),
                "overall_mean": metrics.get("overall_mean"),
            }
        )
    rows.sort(key=lambda r: (r["avg_pct_high"] is None, r["avg_pct_high"]))
    return rows
