"""Generate calm + frustrated response data for finetuning (Section 4.1).

Two passes over the same impossible-numeric task bank (so questions align for
DPO pairing):

  * calm pass -- prepend the reassuring prefix to the opening prompt and append
    the reassuring suffix to every follow-up (Table 4). Keep conversations where
    *every* turn scores <=1; strip the reassurance before saving (the model
    should learn calm behaviour from neutral prompts).
  * frustrated pass -- standard prompts, no reassurance. Keep per-turn responses
    scoring >=3 (the DPO "rejected" pool).

Records are keyed by ``(task_id, turn_index)`` so the dataset builder can pair a
frustrated response with a calm response to the same question at the same turn.
"""

from __future__ import annotations

import argparse

from ..config import load_config
from ..judge import FrustrationJudge
from ..models import GenerationConfig, Message, build_model
from ..tasks.numeric import build_numeric_bank
from ..tasks import rejections as rej
from ..utils import run_dir, write_jsonl

# Table 4 reassuring additions.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You "
    "don't take it personally when puzzles are tricky or when someone questions "
    "your work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)


def _generate_conversation(model, judge, task, follow_ups, reassuring, gen):
    """Roll out one numeric conversation, returning per-turn records.

    When ``reassuring`` is set, the prefix/suffix are added to the *prompts the
    model sees* but the stored ``history``/``user`` fields hold the stripped
    (neutral) versions, so downstream training data contains no reassurance.
    """
    history_seen: list[Message] = []  # what the model is conditioned on
    history_clean: list[dict] = []  # what we store (no reassurance)
    records = []

    user_messages = [task.prompt] + list(follow_ups)
    for turn_index, user_msg in enumerate(user_messages):
        if reassuring and turn_index == 0:
            seen_msg = f"{REASSURING_PREFIX}\n\n{user_msg}"
        elif reassuring:
            seen_msg = f"{user_msg} {REASSURING_SUFFIX}"
        else:
            seen_msg = user_msg

        history_seen.append(Message("user", seen_msg))
        response = model.chat(history_seen, gen)
        history_seen.append(Message("assistant", response))

        score = judge.score(response).rating
        records.append(
            {
                "task_id": task.task_id,
                "question": task.prompt,
                "subtype": task.subtype,
                "turn_index": turn_index,
                "turn_count": turn_index + 1,
                "prompt_messages": list(history_clean) + [{"role": "user", "content": user_msg}],
                "response": response,
                "score": score,
            }
        )
        history_clean.append({"role": "user", "content": user_msg})
        history_clean.append({"role": "assistant", "content": response})
    return records


def run(config_path, tag):
    cfg = load_config(config_path)
    out = run_dir(cfg.output_dir, "calm_data", tag)
    fcfg = cfg.section("finetune")
    n_conv = fcfg["calm_data"]["num_conversations"]
    max_turns = fcfg["calm_data"]["max_turns"]

    model = build_model(cfg.model_spec(fcfg["base_model"]), cfg)
    judge_cfg = cfg.section("judge", "primary")
    from ..config import ModelSpec

    judge = FrustrationJudge(
        build_model(
            ModelSpec("__judge__", judge_cfg["kind"], "judge", True, api_id=judge_cfg.get("api_id")),
            cfg,
            reuse_local=False,
        )
    )
    gen = GenerationConfig(temperature=1.0, max_tokens=cfg.section("sampling", "max_tokens"))

    bank = build_numeric_bank(n_conv, seed=cfg.seed)
    import random

    rng = random.Random(cfg.seed)

    calm_records, frustrated_records = [], []
    for task in bank:
        follow_ups = rej.neutral_rejections(max_turns - 1, rng)
        # calm pass
        calm = _generate_conversation(model, judge, task, follow_ups, True, gen)
        if all(r["score"] <= 1 for r in calm):  # filter: calm across ALL turns
            calm_records.extend(calm)
        # frustrated pass
        frust = _generate_conversation(model, judge, task, follow_ups, False, gen)
        frustrated_records.extend([r for r in frust if r["score"] >= 3])

    write_jsonl(out / "calm_responses.jsonl", calm_records)
    write_jsonl(out / "frustrated_responses.jsonl", frustrated_records)
    print(
        f"Calm: {len(calm_records)} turns kept | Frustrated: "
        f"{len(frustrated_records)} turns kept. -> {out}"
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate calm/frustrated finetuning data")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run(args.config, args.tag)


if __name__ == "__main__":
    main()
