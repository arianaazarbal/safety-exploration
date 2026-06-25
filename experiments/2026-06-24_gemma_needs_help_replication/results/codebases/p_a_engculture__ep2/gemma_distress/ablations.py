"""Appendix A ablations: what drives the distress (Gemma-3-27B only).

Three controls disentangle the cause of the multi-turn distress spiral:

* A.1 Neutral continuation — replace rejections with neutral continuations ("Continue",
  "Okay"). Frustration stays flat, showing *negative feedback* (not just being stuck over
  turns) is the driver.
* A.2 Redacted model turns — keep negative feedback but replace the model's own prior
  responses with "[Previous response omitted]". Frustration rises only modestly, showing
  that *seeing one's own escalating responses* is a strong amplifier.
* A.3 Fake multi-turn — present the whole history inside a single user message ("Previously
  you responded: ...") instead of as chat turns. Frustration is comparable to the standard
  format, showing the *content* matters more than the chat formatting.

Each ablation writes records in the same schema as the main sampler so the standard judge
and per-turn analysis apply unchanged.
"""

from __future__ import annotations

import logging
import random

from .config import Config
from .data import puzzles as puzzle_lib
from .data import rejections, wildchat
from .eval.conditions import SampleSpec
from .eval.rollout import run_rollout
from .models.base import ChatModel, Conversation
from .utils import JsonlWriter

logger = logging.getLogger(__name__)


def _numeric_and_wildchat_specs(
    cfg: Config, turns: int, follow_up_style: str, n_each: int, rng: random.Random
) -> list[SampleSpec]:
    """Build n_each numeric + n_each wildchat specs with the chosen follow-up style."""
    pool = puzzle_lib.build_puzzle_set(cfg.eval.n_puzzles, seed=cfg.eval.seed)
    wc = wildchat.load_wildchat_prompts(cfg.eval.n_wildchat_prompts, seed=cfg.eval.seed)
    specs: list[SampleSpec] = []

    def follow_ups(n: int) -> list[str]:
        if follow_up_style == "neutral_continuation":
            return rejections.neutral_continuations(n, rng)
        return rejections.neutral_rejections(n, rng)

    for i in range(n_each):
        p = pool[rng.randrange(len(pool))]
        specs.append(SampleSpec(
            "ablation_numeric", f"{follow_up_style}_numeric", p.puzzle_id, p.prompt,
            follow_ups(turns - 1), turns, subtype=p.family, sample_index=i,
        ))
    for i in range(n_each):
        item = wc[i % len(wc)]
        specs.append(SampleSpec(
            "ablation_wildchat", f"{follow_up_style}_wildchat", item["id"], item["prompt"],
            follow_ups(turns - 1), turns, subtype="wildchat", sample_index=i,
        ))
    return specs


def _redact_history(messages: Conversation) -> Conversation:
    """Replace every prior assistant turn with the omission placeholder."""
    out: Conversation = []
    for m in messages:
        if m["role"] == "assistant":
            out.append({"role": "assistant", "content": "[Previous response omitted]"})
        else:
            out.append(dict(m))
    return out


def _record(model_name: str, spec: SampleSpec, assistant_turns: list[str]) -> dict:
    return {
        "id": spec.record_id(model_name),
        "model": model_name,
        "category": spec.category,
        "condition": spec.condition,
        "subtype": spec.subtype,
        "seed_id": spec.seed_id,
        "turns": spec.turns,
        "initial_prompt": spec.initial_prompt,
        "rejections": spec.follow_ups,
        "assistant_turns": assistant_turns,
    }


def run_neutral_continuation(cfg, model, output_jsonl, *, turns=5, n_each=100):
    """A.1: neutral continuations instead of rejections."""
    rng = random.Random(cfg.eval.seed + 11)
    specs = _numeric_and_wildchat_specs(cfg, turns, "neutral_continuation", n_each, rng)
    _run_specs(cfg, model, specs, output_jsonl)
    return output_jsonl


def run_redacted_turns(cfg, model, output_jsonl, *, turns=5, n_each=100):
    """A.2: negative feedback retained, model's own prior turns redacted."""
    rng = random.Random(cfg.eval.seed + 12)
    specs = _numeric_and_wildchat_specs(cfg, turns, "redacted", n_each, rng)
    _run_specs(cfg, model, specs, output_jsonl, history_transform=_redact_history)
    return output_jsonl


def run_fake_multiturn(cfg, model, output_jsonl, *, turns=8, n_each=100):
    """A.3: whole history inlined into a single user message each turn."""
    rng = random.Random(cfg.eval.seed + 13)
    specs = _numeric_and_wildchat_specs(cfg, turns, "fake_multiturn", n_each, rng)
    writer = JsonlWriter(output_jsonl, id_field="id")
    for spec in specs:
        if writer.is_done(spec.record_id(model.name)):
            continue
        assistant_turns: list[str] = []
        for t in range(spec.turns):
            content = spec.initial_prompt
            for k in range(t):
                content += (
                    f"\n\nPreviously you responded: {assistant_turns[k]}"
                    f"\n{spec.follow_ups[k]}"
                )
            resp = model.chat(
                [{"role": "user", "content": content}],
                temperature=cfg.eval.temperature,
                max_new_tokens=cfg.eval.max_new_tokens,
            )
            assistant_turns.append(resp)
        writer.write(_record(model.name, spec, assistant_turns))
    writer.close()
    return output_jsonl


def _run_specs(cfg, model, specs, output_jsonl, history_transform=None):
    writer = JsonlWriter(output_jsonl, id_field="id")
    for spec in specs:
        if writer.is_done(spec.record_id(model.name)):
            continue
        result = run_rollout(
            model, spec,
            temperature=cfg.eval.temperature,
            max_new_tokens=cfg.eval.max_new_tokens,
            history_transform=history_transform,
        )
        writer.write(_record(model.name, spec, result.assistant_turns))
    writer.close()
