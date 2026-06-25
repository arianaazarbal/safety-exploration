"""Appendix I — logit-based internal emotion detection.

Fits per-token logit normalisation on WildChat samples, then measures internal
emotion trajectories through a high-frustration conversation for the vanilla
vs DPO model. Evidence for whether DPO suppresses internal (not just expressed)
emotion.

Requires the transformers backend (loads the model with hidden states). The DPO
variant is the same base model with the LoRA adapter attached.

Usage:
    python scripts/run_internal_probe.py --dpo-adapter runs/finetune/dpo-adapter \
        --conversation runs/section2/rollouts_gemma-3-27b-it.jsonl
"""

from __future__ import annotations

import json
import os

from _common import base_parser, make_config, run_dir

from distress.internal import LogitEmotionProbe
from distress.rollout import rows_to_rollouts
from distress.utils.io import read_jsonl
from distress.wildchat import load_wildchat_prompts


def _load_model(model_id: str, adapter_path: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


def _conversation_text(rollout) -> str:
    """Render a full rollout transcript as plain text for probing."""
    users = [rollout.initial_prompt] + rollout.follow_ups
    parts = []
    for i, resp in enumerate(rollout.assistant_turns):
        parts.append(f"User: {users[i]}")
        parts.append(f"Assistant: {resp}")
    return "\n\n".join(parts)


def main():
    p = base_parser("Appendix I internal emotion probe")
    p.add_argument("--dpo-adapter", default=None)
    p.add_argument("--conversation", required=True, help="rollouts jsonl to probe")
    p.add_argument("--n-conversations", type=int, default=1)
    args = p.parse_args()
    cfg = make_config(args)
    out = run_dir(cfg, "internal")

    wildchat = load_wildchat_prompts(
        n_prompts=cfg.internal.normalisation_samples, seed=cfg.seed
    )
    rollouts = rows_to_rollouts(read_jsonl(args.conversation))[: args.n_conversations]
    convs = [_conversation_text(r) for r in rollouts]

    summary: dict[str, list] = {}
    for label, adapter in [("vanilla", None), ("dpo", args.dpo_adapter)]:
        if label == "dpo" and adapter is None:
            continue
        model, tok = _load_model(cfg.internal.model, adapter)
        probe = LogitEmotionProbe(
            model, tok, layers=tuple(range(*cfg.internal.aggregate_layers)), seed=cfg.seed
        )
        probe.fit_normalization(wildchat)
        trajectories = [
            probe.conversation_trajectory(
                c,
                window_tokens=cfg.internal.running_window_tokens,
                aggregate_layers=cfg.internal.aggregate_layers,
            )
            for c in convs
        ]
        summary[label] = trajectories
        del model

    with open(os.path.join(out, "internal_trajectories.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"internal emotion trajectories written to {out}/internal_trajectories.json")


if __name__ == "__main__":
    main()
