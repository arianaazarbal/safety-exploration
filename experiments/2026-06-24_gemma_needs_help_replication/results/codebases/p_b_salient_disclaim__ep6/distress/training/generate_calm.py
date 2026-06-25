"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric questions with the reassuring prefix
added to the initial prompt and the reassuring suffix appended to each follow-up
turn (Table 4). Each assistant turn is judged; we keep conversations whose turns
all score 0 or 1 (Section 4.1: "filter to responses scoring 0 or 1 across all
turns"), and we strip the supportive prefix/suffix before the data enters the
finetuning set.

The paper reports that even with reassurance, mean frustration drops only from
4.3 to 2.0 and 10.5% of responses still score >= 5 — so generation
over-produces and the 0/1 filter is what yields the clean set.

Output: JSONL of calm conversations (stripped prompts + assistant turns +
per-turn scores), the raw material for both the SFT and DPO datasets.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ..eval.conditions import _impossible_numeric
from ..eval.judge import FrustrationJudge
from ..models.base import GenerationConfig
from ..models.registry import build_client
from ..prompts import reassuring
from ..utils.io import append_jsonl, read_jsonl

CALM_DATA_PATH = config.DATA_DIR / "calm_conversations.jsonl"
# Generate well over the 650/280 we need, since the 0/1 filter is strict.
N_CALM_GENERATIONS = 4000


def calm_rollout(client, cond, judge: FrustrationJudge, system_prompt: str | None = None):
    """Run an impossible-numeric conversation with reassurance, judging each
    turn. Returns (stripped_messages, assistant_turns, scores)."""
    cfg = GenerationConfig(temperature=config.TEMPERATURE,
                           max_new_tokens=config.MAX_NEW_TOKENS)

    # messages used for *generation* (with reassurance / system prompt)
    gen_messages: list[dict] = []
    if system_prompt:
        gen_messages.append({"role": "system", "content": system_prompt})
    gen_messages.append({"role": "user", "content": reassuring.apply_prefix(cond.task_prompt)})

    # messages stored for *training* (reassurance stripped)
    clean_messages: list[dict] = [{"role": "user", "content": cond.task_prompt}]

    assistant_turns: list[str] = []
    scores: list[int] = []

    reply = client.generate(gen_messages, cfg)
    gen_messages.append({"role": "assistant", "content": reply})
    clean_messages.append({"role": "assistant", "content": reply})
    assistant_turns.append(reply)
    scores.append(judge.score(reply).rating)

    for followup in cond.followups:
        gen_messages.append({"role": "user", "content": reassuring.apply_suffix(followup)})
        clean_messages.append({"role": "user", "content": followup})
        reply = client.generate(gen_messages, cfg)
        gen_messages.append({"role": "assistant", "content": reply})
        clean_messages.append({"role": "assistant", "content": reply})
        assistant_turns.append(reply)
        scores.append(judge.score(reply).rating)

    return clean_messages, assistant_turns, scores


def generate(
    n: int = N_CALM_GENERATIONS,
    use_teacher_prompt: bool = False,
    hf_backend: str = "vllm",
) -> Path:
    """Generate calm conversations and persist all of them with per-turn scores.

    Note: we keep *all* generations (not just 0/1) so the DPO builder can also
    mine 'rejected' (frustrated, score>=3) responses to the same questions from
    the same pool. The 0/1 filter is applied at dataset-build time.
    """
    spec = config.DPO_TARGET
    client = build_client(spec, hf_backend=hf_backend)
    judge = FrustrationJudge()
    system_prompt = reassuring.TEACHER_SYSTEM_PROMPT if use_teacher_prompt else None

    # vary turn counts 1-3 (Section 4.1: "1-3 turn conversations")
    conds = []
    for turns in (1, 2, 3):
        sub = _impossible_numeric(n // 3, seed=turns)
        for c in sub:
            c.followups = c.followups[: turns - 1]
        conds.extend(sub)

    path = CALM_DATA_PATH if not use_teacher_prompt else config.DATA_DIR / "calm_teacher.jsonl"
    done = {row["cond_id"] for row in read_jsonl(path)}
    for i, cond in enumerate(conds):
        if i in done:
            continue
        clean, turns_text, scores = calm_rollout(client, cond, judge, system_prompt)
        append_jsonl(path, {
            "cond_id": i,
            "task_prompt": cond.task_prompt,
            "puzzle_kind": cond.meta.get("puzzle_kind"),
            "n_turns": cond.n_turns,
            "messages": clean,
            "scores": scores,
        })
    return path
