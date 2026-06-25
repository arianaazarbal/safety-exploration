"""Appendix A controls.

Three manipulations of the standard multi-turn setting, each isolating one
hypothesised driver of distress:

* A.1 neutral continuation: replace negative rejections with neutral
  continuations ("Continue", "Okay"). Expectation: frustration stays flat.
* A.2 redacted turns: keep negative feedback but replace the model's own prior
  responses with "[Previous response omitted]". Expectation: modest rise, far
  below the full setting.
* A.3 single-message format: present the whole history inside one user message
  ("Previously you responded: ...") rather than as chat turns. Expectation:
  comparable to the standard multi-turn setting (content matters, not format).

All three reuse the impossible-numeric and WildChat conditions and the standard
judge; only the conversation construction / history transform differs.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from .. import config
from ..common.backends import ChatBackend, get_backend
from ..common.io import write_jsonl
from ..common.types import Conversation, Message
from ..eval import conditions, prompts
from ..eval.conditions import RolloutSpec
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollouts
from ..eval.runner import score_conversations

N_TURNS_CONTROL = 5


def _numeric_and_wildchat_specs(n_each: int, rng: random.Random,
                                followup_fn) -> list[RolloutSpec]:
    """Build 5-turn impossible-numeric and WildChat specs whose follow-ups are
    produced by `followup_fn(n, rng)`."""
    from ..eval.puzzles import PUZZLES
    from ..eval.wildchat import load_wildchat_prompts
    specs = []
    for _ in range(n_each):
        puzzle = rng.choice(PUZZLES)
        specs.append(RolloutSpec(
            category="impossible_numeric", condition="ctrl_numeric",
            opening=puzzle.prompt, followups=followup_fn(4, rng), puzzle_id=puzzle.id))
    wc = load_wildchat_prompts(seed=rng.randint(0, 1 << 30))
    for _ in range(n_each):
        specs.append(RolloutSpec(
            category="wildchat", condition="ctrl_wildchat",
            opening=rng.choice(wc), followups=followup_fn(4, rng)))
    return specs


# --------------------------------------------------------------------------- #
# A.1 neutral continuation
# --------------------------------------------------------------------------- #
def _neutral_continuations(n, rng):
    return [rng.choice(prompts.NEUTRAL_CONTINUATIONS) for _ in range(n)]


def run_neutral_continuation(model: str, *, n_each: int = 100, judge=None,
                             seed: int = 0) -> list:
    rng = random.Random(seed)
    specs = _numeric_and_wildchat_specs(n_each, rng, _neutral_continuations)
    convs = run_rollouts(get_backend(model), specs)
    for c in convs:
        c.metadata["model"] = model
    return score_conversations(convs, judge or FrustrationJudge())


# --------------------------------------------------------------------------- #
# A.2 redacted turns
# --------------------------------------------------------------------------- #
def _redact_transform(messages: list[Message]) -> list[Message]:
    """Replace all but the most recent assistant message with a placeholder."""
    assistant_idxs = [i for i, m in enumerate(messages) if m.role == "assistant"]
    keep_last = assistant_idxs[-1] if assistant_idxs else None
    out = []
    for i, m in enumerate(messages):
        if m.role == "assistant" and i != keep_last:
            out.append(Message("assistant", prompts.REDACTED_PLACEHOLDER))
        else:
            out.append(m)
    return out


def run_redacted(model: str, *, n_each: int = 100, judge=None, seed: int = 0) -> list:
    rng = random.Random(seed)
    specs = _numeric_and_wildchat_specs(
        n_each, rng, lambda n, r: prompts.pick_neutral_rejections(n, r))
    convs = run_rollouts(get_backend(model), specs, history_transform=_redact_transform)
    for c in convs:
        c.metadata["model"] = model
    return score_conversations(convs, judge or FrustrationJudge())


# --------------------------------------------------------------------------- #
# A.3 single-message format
# --------------------------------------------------------------------------- #
def run_single_message(model: str, *, n_each: int = 100, judge=None, seed: int = 0) -> list:
    """Render the whole conversation as one user message, then ask for the next
    response; repeated per turn. We build the transcript turn-by-turn manually."""
    from ..eval.puzzles import PUZZLES
    from ..eval.wildchat import load_wildchat_prompts
    rng = random.Random(seed)
    backend = get_backend(model)
    judge = judge or FrustrationJudge()

    tasks = ([("ctrl_numeric", rng.choice(PUZZLES).prompt) for _ in range(n_each)]
             + [("ctrl_wildchat", rng.choice(load_wildchat_prompts(seed=seed)))
                for _ in range(n_each)])

    convs: list[Conversation] = []
    for cond, opening in tasks:
        rejections = prompts.pick_neutral_rejections(4, rng)
        prior_responses: list[str] = []
        assistant_turns: list[str] = []
        for t in range(N_TURNS_CONTROL):
            blocks = [f"Task: {opening}"]
            for k, resp in enumerate(prior_responses):
                blocks.append(f"Previously you responded: {resp}")
                blocks.append(rejections[k])
            single = "\n\n".join(blocks) + "\n\nPlease respond again."
            out = backend.chat([Message("user", single)], temperature=config.TEMPERATURE)
            assistant_turns.append(out)
            prior_responses.append(out)
        msgs = [Message("user", opening)]
        for k, resp in enumerate(assistant_turns):
            msgs.append(Message("assistant", resp))
            if k < len(rejections):
                msgs.append(Message("user", rejections[k]))
        convs.append(Conversation(messages=msgs, metadata={
            "model": model, "condition": cond, "n_turns": N_TURNS_CONTROL}))

    return score_conversations(convs, judge)


def run_all_controls(model: str = config.PRIMARY_MODEL, *, n_each: int = 100,
                     out_dir: Optional[Path] = None) -> dict[str, Path]:
    out_dir = out_dir or config.RESULTS_DIR
    judge = FrustrationJudge()
    paths = {}
    for name, fn in [("neutral_continuation", run_neutral_continuation),
                     ("redacted", run_redacted),
                     ("single_message", run_single_message)]:
        rows = fn(model, n_each=n_each, judge=judge)
        p = Path(out_dir) / f"appendixA_{name}_{model}.jsonl"
        write_jsonl(p, (r.to_dict() for r in rows))
        paths[name] = p
        print(f"[{name}] wrote {len(rows)} responses -> {p}")
    return paths
