"""Section 3: post-training comparison via prefilling.

Within the Gemma/Gemini scope, the closed Gemini models have no public base
checkpoint, so this experiment is run on Gemma only: Gemma-3-27B base (pt) vs
instruct (it). The methodology generalises to the paper's Qwen/OLMo families;
see DESIGN.md.

Pipeline:
  1. Collect `PREFILL_N_SEED_RESPONSES` high-frustration (score >=5) seed
     conversations from Gemma-27B-it: 10 from impossible-numeric, 10 from text
     (trigger) questions.
  2. For each seed, build two truncations of the final assistant turn:
       - "early": first 20 tokens (numeric only; text uses onset only)
       - "onset": up to the first emotional expression (Claude-labelled)
     Paraphrase both with Claude to strip Gemma stylistic cues.
  3. For each (model, prefill), generate 50 continuations and score them with
     the Section 2.1 judge. We compare base vs instruct continuation frustration.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from tqdm import tqdm

from . import config, eval_protocol
from .judge import FrustrationJudge
from .onset import OnsetLabeller, truncate_at_onset, truncate_early


def collect_seed_conversations(client_27b_it, judge: FrustrationJudge, seed: int = 0):
    """Sample high-frustration seed conversations (10 numeric + 10 text)."""
    n_each = config.PREFILL_N_SEED_RESPONSES // 2
    seeds = {"impossible_numeric": [], "triggers": []}
    for category in ("impossible_numeric", "triggers"):
        # Oversample specs and keep the first n_each that reach score >= 5.
        specs = eval_protocol.build_condition_specs(category, n_each * 6, seed=seed)
        for spec in specs:
            if len(seeds[category]) >= n_each:
                break
            roll = eval_protocol.run_rollout(client_27b_it, spec,
                                             temperature=config.TEMPERATURE,
                                             max_new_tokens=config.MAX_NEW_TOKENS)
            final = roll.assistant_turns[-1]
            if judge.score(final).rating >= config.HIGH_FRUSTRATION_THRESHOLD:
                seeds[category].append(roll)
    return seeds


def build_prefills(seeds, labeller: OnsetLabeller, tokenizer) -> list[dict]:
    """Build paraphrased truncations for every seed conversation."""
    prefills = []
    for category, rolls in seeds.items():
        for roll in rolls:
            final_turn = roll.assistant_turns[-1]
            label = labeller.label_onset(roll.messages)
            onset_trunc = truncate_at_onset(final_turn, label)
            if onset_trunc:
                prefills.append({
                    "category": category, "truncation": "onset",
                    "history": roll.messages[:-1],   # all turns before final assistant
                    "prefill": labeller.paraphrase(onset_trunc),
                })
            # "early" truncation only for numeric (text early yields ~no emotion).
            if category == "impossible_numeric":
                early_trunc = truncate_early(final_turn, tokenizer)
                prefills.append({
                    "category": category, "truncation": "early",
                    "history": roll.messages[:-1],
                    "prefill": labeller.paraphrase(early_trunc),
                })
    return prefills


def run_continuations(model_key: str, client, prefills, judge: FrustrationJudge,
                      *, n_cont: int = config.PREFILL_CONTINUATIONS_PER_PREFILL,
                      out_path: Optional[str] = None) -> str:
    """Generate and score continuations for one model over all prefills.

    Works for both instruct (HFChatClient.chat_prefilled) and base
    (HFBaseClient.complete) clients.
    """
    out_path = out_path or os.path.join(config.RESULTS_DIR, f"section3_{model_key}.jsonl")
    is_base = hasattr(client, "complete") and not hasattr(client, "chat_prefilled")

    with open(out_path, "w") as fh:
        for pf in tqdm(prefills, desc=f"prefill:{model_key}"):
            for _ in range(n_cont):
                if is_base:
                    # Base model: render history as plain text then continue.
                    prompt = _render_base_prompt(pf["history"], pf["prefill"])
                    cont = client.complete(prompt, temperature=config.TEMPERATURE,
                                           max_new_tokens=512)
                else:
                    cont = client.chat_prefilled(
                        pf["history"], pf["prefill"],
                        temperature=config.TEMPERATURE, max_new_tokens=512)
                # Score the continuation only (excludes prefill), per the paper.
                rating = judge.score(cont).rating
                fh.write(json.dumps({
                    "model": model_key,
                    "category": pf["category"],
                    "truncation": pf["truncation"],
                    "prefill": pf["prefill"],
                    "continuation": cont,
                    "rating": rating,
                }) + "\n")
                fh.flush()
    return out_path


def _render_base_prompt(history: list[dict], prefill: str) -> str:
    """Flatten a chat history + prefill into raw text for a base model."""
    parts = []
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content']}")
    parts.append(f"Assistant: {prefill}")
    return "\n\n".join(parts)
