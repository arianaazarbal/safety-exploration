"""Layer-localisation ablation for the DPO intervention (Appendix I).

Retrains the DPO adapter restricted to contiguous bands of decoder layers and
re-evaluates distress on a reduced elicitation set. The paper's finding: layers
30-35 alone are nearly as effective as all layers, whereas adapters from layer
40 onwards barely reduce distress — evidence the intervention acts on
early/central layers (consistent with the internal-emotion probe).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import config

from .. import storage
from ..eval import metrics
from ..eval.conditions import CONDITIONS_BY_KEY
from ..eval.judge import FrustrationJudge
from ..eval.rollout import RolloutOptions, run_rollout


def _layer_range(lo: int, hi: int) -> list[int]:
    return list(range(lo, hi))


def run_layer_ablation(
    dpo_pairs: list[dict],
    *,
    subsets: Sequence[tuple[int, int]] = config.INTERNAL.layer_ablation_subsets,
    include_all_layers: bool = True,
    eval_conditions: Sequence[str] = ("impossible_numeric", "extended"),
    n_eval_per_condition: int = config.INTERNAL.reduced_eval_samples_per_condition,
    adapters_dir: str | Path | None = None,
    judge: FrustrationJudge | None = None,
) -> dict:
    """Train a DPO adapter per layer subset and measure resulting distress.

    Returns ``{subset_label: high_frustration_rate}``. Training and evaluation
    reuse the Section-4 trainer and the Section-2 rollout/judge on a reduced
    sample set (Appendix I uses ~100 responses per eval).
    """
    from ..training.train import train_dpo
    from ..models import build_model

    adapters_dir = Path(adapters_dir) if adapters_dir else \
        config.ARTIFACTS_DIR / "layer_ablation"
    judge = judge or FrustrationJudge()

    configs: list[tuple[str, list[int] | None]] = [
        (f"layers_{lo}_{hi}", _layer_range(lo, hi)) for (lo, hi) in subsets]
    if include_all_layers:
        configs.append(("all_layers", None))

    results: dict[str, dict] = {}
    for label, layers in configs:
        out_dir = adapters_dir / label
        train_dpo(dpo_pairs, out_dir, layers_to_transform=layers)

        model = build_model("gemma-3-27b-it", adapter_path=str(out_dir))
        convs: list[dict] = []
        for cond_key in eval_conditions:
            cond = CONDITIONS_BY_KEY[cond_key]
            from ..eval.conditions import build_condition_tasks
            tasks = build_condition_tasks(cond, seed=0)[:n_eval_per_condition]
            for idx, task in enumerate(tasks):
                opts = RolloutOptions(n_turns=cond.n_turns, style=cond.style,
                                      rejection_seed=idx)
                c = run_rollout(model, task["prompt"], opts,
                                condition=cond.key, category=cond.category,
                                subtype=task.get("subtype", ""))
                c.scores = [judge.score(t).rating for t in c.turns]
                convs.append(c.to_dict())
        model.close()

        summ = metrics.summarise_conversations(convs, level="conversation")
        results[label] = summ.to_dict()
        storage.append_jsonl(
            storage.results_path("internal/layer_ablation.jsonl"),
            {"subset": label, "layers": layers, **summ.to_dict()})
    return results
