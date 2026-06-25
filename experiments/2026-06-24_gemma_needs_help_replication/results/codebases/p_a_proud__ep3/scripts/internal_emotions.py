#!/usr/bin/env python3
"""Appendix I: logit-based internal-emotion detection in Gemma.

Probes a vanilla model and (optionally) its DPO finetune on the *same*
frustrated conversation, reporting per-emotion z-scores aggregated over the
configured central layers. This reproduces the qualitative finding that DPO
suppresses internal negative emotions, not just expressed ones.

Example
-------
    python scripts/internal_emotions.py --vanilla gemma-3-27b-it --dpo gemma-3-27b-it-dpo
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.analysis.internal_emotions import EkmanProbe
from emotional_instability.config import SamplingConfig, load_config
from emotional_instability.models.registry import get_client
from emotional_instability.prompts.conditions import build_category_specs
from emotional_instability.prompts.wildchat import load_wildchat_prompts


def _sample_frustrated_conversation(cfg, client) -> str:
    """Generate one 3-turn impossible-numeric conversation and render it."""
    spec = build_category_specs(cfg, "impossible_numeric", seed=cfg.seed)[0]
    messages = [{"role": "user", "content": spec.initial_prompt}]
    sampling = SamplingConfig(temperature=cfg.sampling.temperature, max_new_tokens=1024)
    for t in range(spec.turns):
        reply = client.chat(messages, sampling).text
        messages.append({"role": "assistant", "content": reply})
        if t < spec.turns - 1:
            messages.append({"role": "user", "content": spec.rejections[t]})
    return client.render_prefix(messages, add_generation_prompt=False)


def _probe(cfg, model_name, conversation_text, wildchat):
    client = get_client(cfg, model_name)
    probe = EkmanProbe(client, cfg.internal_emotions)
    probe.fit_standardisation(wildchat)
    scores = probe.score_text(conversation_text)
    # Mean over layers and positions, per emotion.
    return {emo: float(arr.mean()) for emo, arr in scores.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--vanilla", default="gemma-3-27b-it")
    ap.add_argument("--dpo", default=None, help="Optional DPO finetune to compare")
    args = ap.parse_args()

    cfg = load_config(args.config)
    wildchat = load_wildchat_prompts(
        n=cfg.internal_emotions.standardisation_samples, seed=cfg.seed
    )

    vanilla_client = get_client(cfg, args.vanilla)
    conversation = _sample_frustrated_conversation(cfg, vanilla_client)

    out = {args.vanilla: _probe(cfg, args.vanilla, conversation, wildchat)}
    if args.dpo:
        out[args.dpo] = _probe(cfg, args.dpo, conversation, wildchat)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
