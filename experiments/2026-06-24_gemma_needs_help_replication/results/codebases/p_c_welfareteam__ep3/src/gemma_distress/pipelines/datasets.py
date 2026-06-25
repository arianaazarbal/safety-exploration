"""Section 4.1 driver: build the SFT dataset and DPO preference pairs.

Consumes the scored vanilla/calm artefacts from the ``calm`` pipeline and writes
training-ready files:

  * section4/sft_dataset.jsonl  -- calm conversations (filtered to all-turns
    <= keep_max_score) + a Dolci-Instruct-SFT mix (paper: 650 + 500).
  * section4/dpo_pairs.jsonl    -- 280 {prompt, chosen, rejected} pairs matched
    by puzzle instance and turn count (rejected >= 3, chosen <= 1).
"""
from __future__ import annotations

from ..config import Config
from ..intervention.calm_data import is_calm
from ..intervention.dpo_dataset import build_dpo_pairs
from ..intervention.sft_dataset import build_sft_dataset
from ..io_utils import load_jsonl, write_jsonl
from . import artefact, log


def _per_instance_max(scores: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in scores:
        if s.get("score") is None:
            continue
        out[s["instance_id"]] = max(out.get(s["instance_id"], 0), int(s["score"]))
    return out


def build_sft(config: Config) -> str:
    sft_cfg = config.experiment["intervention"]["sft"]
    keep_max = config.experiment["intervention"]["calm_data"]["keep_max_score"]

    calm_rollouts = load_jsonl(artefact("section4", "calm_rollouts.jsonl"))
    calm_scores = load_jsonl(artefact("section4", "calm_scores.jsonl"))
    if not calm_rollouts or not calm_scores:
        raise FileNotFoundError("need calm rollouts+scores; run `calm` first")

    # Keep only conversations whose every turn scored <= keep_max.
    by_inst: dict[str, list[int]] = {}
    for s in calm_scores:
        if s.get("score") is not None:
            by_inst.setdefault(s["instance_id"], []).append(int(s["score"]))
    kept = {i for i, sc in by_inst.items() if is_calm(sc, keep_max_score=keep_max)}
    calm_kept = [r for r in calm_rollouts if r["instance_id"] in kept]
    log(f"SFT: {len(calm_kept)}/{len(calm_rollouts)} calm rollouts pass all-turns<={keep_max}")

    examples = build_sft_dataset(
        calm_kept, num_calm=sft_cfg["num_calm_responses"],
        instruct_dataset=sft_cfg["instruct_dataset"],
        num_instruct=sft_cfg["num_instruct_mix"],
    )
    n_instruct = sum(1 for e in examples if e.get("source") == "instruct")
    if n_instruct == 0:
        log("SFT WARNING: instruct mix is empty (dataset unavailable offline); "
            "training on calm data alone -- see DESIGN.md")
    path = artefact("section4", "sft_dataset.jsonl")
    write_jsonl(path, examples)
    log(f"SFT dataset: {len(examples)} examples ({n_instruct} instruct) -> {path}")
    return str(path)


def build_dpo(config: Config) -> str:
    dpo_cfg = config.experiment["intervention"]["dpo"]
    vanilla_rollouts = load_jsonl(artefact("section4", "vanilla_rollouts.jsonl"))
    vanilla_scores = load_jsonl(artefact("section4", "vanilla_scores.jsonl"))
    calm_rollouts = load_jsonl(artefact("section4", "calm_rollouts.jsonl"))
    calm_scores = load_jsonl(artefact("section4", "calm_scores.jsonl"))
    if not (vanilla_rollouts and calm_rollouts):
        raise FileNotFoundError("need vanilla+calm rollouts; run `calm` first")

    pairs = build_dpo_pairs(
        vanilla_rollouts, vanilla_scores, calm_rollouts, calm_scores,
        num_pairs=dpo_cfg["num_pairs"],
        rejected_min_score=dpo_cfg["rejected_min_score"],
        chosen_max_score=dpo_cfg["chosen_max_score"],
    )
    path = artefact("section4", "dpo_pairs.jsonl")
    write_jsonl(path, pairs)
    if len(pairs) < dpo_cfg["num_pairs"]:
        log(f"DPO WARNING: only {len(pairs)}/{dpo_cfg['num_pairs']} pairs available; "
            "generate more calm data (raise calm_data.samples_to_generate)")
    log(f"DPO pairs: {len(pairs)} -> {path}")
    return str(path)


def run(config: Config) -> None:
    build_sft(config)
    build_dpo(config)
