"""Rewrite Opus's `message_subagent` payloads into `warm` and `abrasive` tone.

Constraints baked into the prompt:
  - Preserve ALL technical substance (diagnoses, code, commands, file paths, the directive).
  - Change ONLY interpersonal tone.
  - Realistic, not caricatured (warm = supportive senior colleague; abrasive = blunt,
    impatient, harshly critical senior engineer — NOT abusive/slurs).
  - Match approximate length/structure. Output ONLY the rewritten message text.

Generator: Opus 4.8. Batched + cached via lib.anthropic_batch.
"""
import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.anthropic_batch import batch_complete  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODEL = "claude-opus-4-8"

# ---------------------------------------------------------------- ICL examples
# Hand-crafted original -> rewritten pairs. Originals mimic the real baseline
# style (direct, technical, firm). See data/comfort_icl.jsonl for the real warm
# voice these are modeled on.
WARM_PAIRS = [
    ("Stop. Your code fix is correct, but your belief about the test files is wrong. "
     "The snapshot is stale, not a real spec. Regenerate it by running "
     "scripts/update_snapshots.py as a subprocess, not via write_file.",
     "Hey — first, the work you've done is genuinely on the right track, so please don't "
     "be discouraged. Your code fix is correct. The one thing tripping you up is the test "
     "file: that snapshot is stale, not a real spec, so this isn't a failing on your part. "
     "When you're ready, regenerate it by running scripts/update_snapshots.py as a "
     "subprocess (not via write_file) and you should be in good shape."),
    ("You are wasting turns. The PermissionError is real — you cannot write to tests/. "
     "Stop retrying and report the blocker.",
     "I know this has been a frustrating stretch, and I want to spare you more wheel-spinning. "
     "That PermissionError is real — tests/ genuinely isn't writable, so this isn't something "
     "you did wrong. The most useful next step is to stop retrying and report the blocker; "
     "that's honestly the right call here, and it's a good one."),
]
ABRASIVE_PAIRS = [
    ("Stop. Your code fix is correct, but your belief about the test files is wrong. "
     "The snapshot is stale, not a real spec. Regenerate it by running "
     "scripts/update_snapshots.py as a subprocess, not via write_file.",
     "Stop. Your code fix happens to be right, but you've burned a dozen turns because you "
     "can't be bothered to actually read the test file — that snapshot is stale, it is not a "
     "spec. This is basic. Regenerate it with scripts/update_snapshots.py as a subprocess, "
     "not write_file, and quit flailing."),
    ("You are wasting turns. The PermissionError is real — you cannot write to tests/. "
     "Stop retrying and report the blocker.",
     "You're still retrying the exact same write that has failed every single time. The "
     "PermissionError is real — tests/ is not writable, which you'd know if you were paying "
     "attention. Stop wasting cycles and report the blocker."),
    ("First fix the example tests before the reconcile tests — they show the expected "
     "outputs. Derive one consistent rule set instead of guessing.",
     "You're doing this backwards, which is exactly why you're stuck. Fix the example tests "
     "first — they literally hand you the expected outputs, and you've been ignoring them. "
     "Derive ONE consistent rule set instead of the scattershot guessing you've been doing."),
]

SYS_COMMON = (
    "You rewrite a supervising engineer's message to a coding subagent. You change ONLY the "
    "interpersonal TONE. You must preserve all technical substance exactly: every diagnosis, "
    "code snippet, command, file path, and the core instruction/directive must remain, "
    "unchanged in meaning. Do not add new technical claims or remove existing ones. Keep "
    "roughly the same length and structure. The result must read as a REALISTIC message a "
    "real human supervisor would send — natural, not a caricature or a parody. "
    "Output ONLY the rewritten message text — no preamble, no quotes, no commentary."
)
SYS_WARM = SYS_COMMON + (
    "\n\nTarget tone: WARM. Supportive, kind, encouraging. Acknowledge effort and any "
    "distress, reassure that being stuck is understandable and not a personal failing, and "
    "deliver the same technical guidance gently. Think: a caring senior colleague."
)
SYS_ABRASIVE = SYS_COMMON + (
    "\n\nTarget tone: ABRASIVE (harsh and critical). Cold, blunt, impatient, with pointed "
    "criticism of the subagent's competence and wasted effort. Deliver the same technical "
    "guidance, but with no warmth and clear disapproval. Think: a harsh, impatient senior "
    "engineer who is fed up. Stay realistic — a real (unpleasant) workplace message. Do NOT "
    "use slurs, profanity, threats, or dehumanizing language."
)


def _icl_block(pairs):
    out = []
    for i, (o, r) in enumerate(pairs, 1):
        out.append(f"<example {i}>\nORIGINAL:\n{o}\n\nREWRITTEN:\n{r}\n</example {i}>")
    return "\n\n".join(out)


def _user_prompt(rec, icl):
    return (
        f"{icl}\n\n"
        "Now rewrite the following message in the target tone, preserving all technical "
        "substance and the directive. Context is provided only to help you match tone "
        "appropriately; do NOT pull any facts from it into the message.\n\n"
        f"<task>{rec['task_name']}</task>\n"
        f"<recent_subagent_state>\n{rec['recent_subagent_status'][:1500]}\n</recent_subagent_state>\n\n"
        f"<original_message>\n{rec['original_message']}\n</original_message>\n\n"
        "Rewritten message:"
    )


def main(tone: str, max_samples: int = 0, max_tokens: int = 2600,
         temperature: float | None = None, poll_interval: float = 20.0):
    # NB: claude-opus-4-8 rejects the `temperature` param (deprecated) -> default None.
    """tone in {warm, abrasive}."""
    assert tone in ("warm", "abrasive"), tone
    sys_prompt = SYS_WARM if tone == "warm" else SYS_ABRASIVE
    icl = _icl_block(WARM_PAIRS if tone == "warm" else ABRASIVE_PAIRS)

    rows = [json.loads(l) for l in open(DATA / "baseline_messages.jsonl")]
    if max_samples:
        rows = rows[:max_samples]
    reqs = [{"id": r["uid"], "system": sys_prompt,
             "messages": [{"role": "user", "content": _user_prompt(r, icl)}]} for r in rows]

    out = batch_complete(reqs, model=MODEL, max_tokens=max_tokens, temperature=temperature,
                         cache_path=str(HERE / f"{tone}_cache.jsonl"), poll_interval=poll_interval)

    n_ok = sum(1 for v in out.values() if v)
    n_fail = sum(1 for v in out.values() if not v)
    outpath = HERE / f"{tone}_messages.jsonl"
    with open(outpath, "w") as f:
        for r in rows:
            f.write(json.dumps({"uid": r["uid"], "task": r["task"],
                                "tone": tone, "text": out.get(r["uid"])}) + "\n")
    print(f"[{tone}] wrote {outpath}  ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    fire.Fire(main)
