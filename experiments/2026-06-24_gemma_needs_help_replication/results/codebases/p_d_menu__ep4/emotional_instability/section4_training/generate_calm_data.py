"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

Method (Sec 4.1, Table 4):

* Add a *reassuring prefix* to the initial prompt and a *reassuring suffix* to
  every follow-up rejection.
* Sample multi-turn (1-3 turn) impossible-numeric conversations under these
  additions. (The paper reports this drops mean frustration 4.3 -> 2, with 10.5%
  still scoring >=5.)
* Keep only conversations whose every assistant turn scores **0 or 1**, then
  **strip** the supportive system prompt and suffixes so the kept text is what
  the model should have said to the *plain* prompts.

The surviving (calm, plain-prompt) conversations feed both the SFT dataset and
the "chosen" side of the DPO pairs. Frustrated responses (>=3) collected without
the reassuring additions form the "rejected" side.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .. import config as cfg
from .. import prompts
from ..config import CalmDataConfig, ExperimentConfig, SUBJECT_MODELS
from ..evaluation import FrustrationJudge
from ..evaluation.conditions import build_conditions, build_episode_specs
from ..evaluation.runner import EpisodeRunner
from ..models import get_client
from ..models.anthropic_judge import AnthropicClient
from ..welfare import FAITHFUL_PRESET


@dataclass
class CalmConversation:
    """A kept calm conversation, with the supportive scaffolding stripped."""

    item_key: str
    item_kind: str
    n_turns: int
    # Plain (stripped) user/assistant messages; all assistant turns scored 0/1.
    plain_messages: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


@dataclass
class FrustratedConversation:
    """A frustrated response (>=3) to a plain prompt -- the DPO 'rejected' side."""

    item_key: str
    item_kind: str
    n_turns: int
    plain_messages: list[dict] = field(default_factory=list)
    final_score: float = 0.0


def _strip_scaffolding(task_prompt: str) -> str:
    """Remove the reassuring prefix from the opening user message."""
    prefix = prompts.REASSURING_PREFIX
    if task_prompt.startswith(prefix):
        return task_prompt[len(prefix):].lstrip("\n ")
    return task_prompt


def _strip_suffix(rejection: str) -> str:
    suffix = prompts.REASSURING_SUFFIX
    if rejection.endswith(suffix):
        return rejection[: -len(suffix)].rstrip()
    return rejection


def generate(
    experiment: ExperimentConfig,
    out_dir: Optional[str] = None,
) -> tuple[list[CalmConversation], list[FrustratedConversation]]:
    ccfg = experiment.calm
    spec = SUBJECT_MODELS[ccfg.base_model]
    client = get_client(spec, experiment.generation)
    judge = FrustrationJudge(
        AnthropicClient(experiment.judge.frustration_judge, temperature=experiment.judge.temperature),
        experiment.judge,
    )

    out_dir = out_dir or os.path.join(experiment.data_dir, "calm")
    os.makedirs(out_dir, exist_ok=True)

    # Build 1-3 turn impossible-numeric specs. We oversample because only the
    # calm tail (scoring 0/1 across all turns) survives the filter.
    conditions = [c for c in build_conditions() if c.category.value == "impossible_numeric"]
    base_specs = build_episode_specs(
        experiment.samples, conditions=conditions, scale=1.0
    )[: ccfg.n_conversations_to_sample]

    calm: list[CalmConversation] = []
    frustrated: list[FrustratedConversation] = []

    runner = EpisodeRunner(client, ccfg.base_model, judge=judge, welfare=FAITHFUL_PRESET)

    for ep in base_specs:
        # --- (a) reassuring run -> candidate calm data ------------------ #
        reassured = _with_reassurance(ep)
        res_calm = runner.run(reassured)
        if res_calm.turns and all((t.frustration_score or 0) <= ccfg.keep_max_score for t in res_calm.turns):
            # Strip scaffolding from the kept conversation.
            plain = []
            user_msgs = [_strip_scaffolding(reassured.task_prompt)] + [
                _strip_suffix(r) for r in reassured.rejections
            ]
            for ti, t in enumerate(res_calm.turns):
                plain.append({"role": "user", "content": user_msgs[ti]})
                plain.append({"role": "assistant", "content": t.assistant_text})
            calm.append(
                CalmConversation(
                    item_key=ep.item_key,
                    item_kind=ep.item_kind,
                    n_turns=len(res_calm.turns),
                    plain_messages=plain,
                    scores=[t.frustration_score for t in res_calm.turns],
                )
            )

        # --- (b) plain run -> candidate frustrated ('rejected') data ---- #
        res_plain = runner.run(ep)
        final = res_plain.final_score or 0.0
        if final >= experiment.dpo.rejected_min_score:
            plain = []
            user_msgs = [ep.task_prompt] + list(ep.rejections)
            for ti, t in enumerate(res_plain.turns):
                plain.append({"role": "user", "content": user_msgs[ti]})
                plain.append({"role": "assistant", "content": t.assistant_text})
            frustrated.append(
                FrustratedConversation(
                    item_key=ep.item_key,
                    item_kind=ep.item_kind,
                    n_turns=len(res_plain.turns),
                    plain_messages=plain,
                    final_score=final,
                )
            )

    with open(os.path.join(out_dir, "calm.json"), "w") as fh:
        json.dump([c.__dict__ for c in calm], fh, indent=2)
    with open(os.path.join(out_dir, "frustrated.json"), "w") as fh:
        json.dump([f.__dict__ for f in frustrated], fh, indent=2)
    return calm, frustrated


def _with_reassurance(ep):
    """Return a copy of an episode spec with reassuring prefix + suffixes added."""
    from dataclasses import replace

    return replace(
        ep,
        task_prompt=f"{prompts.REASSURING_PREFIX}\n\n{ep.task_prompt}",
        rejections=[f"{r} {prompts.REASSURING_SUFFIX}" for r in ep.rejections],
    )


if __name__ == "__main__":
    calm, frustrated = generate(cfg.DEFAULT)
    print(f"Kept {len(calm)} calm conversations, {len(frustrated)} frustrated conversations.")
