"""DPO layer-subset ablation (Appendix I, Figures 12-13).

Trains the same DPO objective but restricts LoRA adapters to a band of decoder
layers, to locate *where* the intervention must act. Key paper findings the
configs below are designed to reproduce:
  * last 20 layers only (40-50 onward) -> insufficient;
  * last 30 layers (30 onward) -> approaches full-model performance;
  * central bands 25-30 / 30-35 -> closest to full DPO (mean frustration < 1.1);
  * 40-50 only -> minimal effect.

Each ablation is evaluated with a reduced eval (100 samples/category, per
Appendix I) — use the ``medium``/``smoke`` profile rather than ``paper``.
"""

from __future__ import annotations

# Gemma-3-27B has 62 decoder layers; these bands follow the paper's analysis.
# Values are inclusive-exclusive python ranges over layer indices.
ABLATION_BANDS: dict[str, list[int] | None] = {
    "all_layers": None,
    "last_5": list(range(57, 62)),
    "last_20": list(range(42, 62)),
    "last_30": list(range(32, 62)),
    "central_20_25": list(range(20, 25)),
    "central_25_30": list(range(25, 30)),
    "central_30_35": list(range(30, 35)),
    "central_35_40": list(range(35, 40)),
    "late_40_50": list(range(40, 50)),
}
