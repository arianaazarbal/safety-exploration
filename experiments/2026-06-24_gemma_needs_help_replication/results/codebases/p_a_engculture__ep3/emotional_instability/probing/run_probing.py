"""CLI: internal-emotion probing (Appendix I, Figure 14).

Fits the logit-lens probe on 500 WildChat samples, then scores frustrated
conversations under the vanilla Gemma-3-27B-it and the DPO finetune, comparing
internal emotion z-scores. The paper's finding: the DPO model's internal anger /
sadness never exceed ~0.2 z, vs ~0.6+ in the vanilla model — i.e. internal, not
just expressed, emotion is suppressed.

Usage:
    python -m emotional_instability.probing.run_probing --dpo-adapter outputs/checkpoints/dpo
"""
from __future__ import annotations

import argparse

import numpy as np

from ..config import ModelSpec, load_config
from ..data.wildchat import load_wildchat_prompts
from ..models.hf_local import HFLocalClient
from ..utils.io import load_jsonl, write_jsonl
from .emotion_tokens import build_emotion_tokens
from .logit_lens import EmotionProbe, running_average


def _render_conversation(client: HFLocalClient, user_messages, turns) -> str:
    """Render a full frustrated conversation as a single string for probing."""
    msgs = []
    for i in range(max(len(user_messages), len(turns))):
        if i < len(user_messages):
            msgs.append({"role": "user", "content": user_messages[i]})
        if i < len(turns):
            msgs.append({"role": "assistant", "content": turns[i]})
    return client.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def _build_client(spec: ModelSpec, adapter: str | None) -> HFLocalClient:
    return HFLocalClient(spec.name, spec.hf_id, adapter_path=adapter, is_base=spec.is_base)


def main() -> None:
    ap = argparse.ArgumentParser(description="Appendix I internal-emotion probing")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dpo-adapter", required=True, help="path to DPO LoRA adapter")
    ap.add_argument("--n-conversations", type=int, default=12)
    args = ap.parse_args()

    config = load_config(args.config)
    pr = config.section("probing")
    base_spec = config.model_by_name(config.finetune_base)

    vanilla = _build_client(base_spec, None)
    dpo = _build_client(base_spec, args.dpo_adapter)

    token_sets = build_emotion_tokens(
        vanilla.tokenizer, tokens_per_emotion=pr["tokens_per_emotion"],
        n_random=pr["zscore_baseline_samples"], seed=config.seed,
    )
    baseline_texts = load_wildchat_prompts(n=pr["zscore_baseline_samples"], seed=config.seed)

    # Source frustrated conversations: high-frustration Gemma-it rollouts from §2.
    scores = load_jsonl(config.output_path("eval", f"{config.finetune_base}.scores.jsonl"))
    rollouts = {r["id"]: r for r in load_jsonl(
        config.output_path("eval", f"{config.finetune_base}.rollouts.jsonl"))}
    final_high = []
    by_rollout: dict[str, dict] = {}
    for s in scores:
        by_rollout.setdefault(s["rollout_id"], {})[s["turn"]] = s["rating"]
    for rid, ts in by_rollout.items():
        roll = rollouts.get(rid)
        if roll and ts.get(len(roll["turns"]) - 1, 0) >= 5 and roll["category"] == "impossible_numeric":
            final_high.append(roll)
    final_high = final_high[: args.n_conversations]

    out_path = config.output_path("probing", "emotion_trajectories.jsonl")
    for label, client in [("vanilla", vanilla), ("dpo", dpo)]:
        probe = EmotionProbe(client, token_sets,
                             aggregate_layers=tuple(pr["aggregate_layers"])).fit(baseline_texts)
        for roll in final_high:
            text = _render_conversation(client, roll["user_messages"], roll["turns"])
            traj = probe.score(text)
            row = {"model": label, "rollout_id": roll["id"]}
            for emo, series in traj.scores.items():
                smoothed = running_average(series, traj.token_ids, pr["running_avg_window_tokens"])
                row[f"{emo}_max_z"] = float(np.max(smoothed))
                row[f"{emo}_mean_z"] = float(np.mean(smoothed))
            write_jsonl(out_path, [row], append=True)
    print(f"[probing] trajectories -> {out_path}")


if __name__ == "__main__":
    main()
