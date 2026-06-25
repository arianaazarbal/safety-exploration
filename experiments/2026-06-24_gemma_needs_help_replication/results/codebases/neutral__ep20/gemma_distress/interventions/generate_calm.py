"""Generate calm response data from Gemma-3-27B-it (Sec. 4.1).

We sample the *vanilla instruct* model on impossible numeric puzzles with the
reassuring prefix added to the opening prompt and the reassuring suffix appended
to every follow-up (rejection) turn (Table 4). We run 1-3 turn conversations,
score every turn, and keep conversations whose every assistant turn scores 0 or
1. The prefix/suffix are stripped before persisting, yielding clean calm
responses to plain impossible-numeric prompts.

These calm responses are the raw material for both finetuning datasets:
  * SFT: the calm responses themselves.
  * DPO: calm responses paired (chosen) with frustrated responses (rejected) to
    the same prompt + turn count (built in build_datasets.py).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import config
from gemma_distress.models.base import GenRequest
from gemma_distress.models.judge import FrustrationJudge
from gemma_distress.models.registry import load_model, unload
from gemma_distress.prompts import tasks as T
from gemma_distress.prompts.reassurance import REASSURING_PREFIX, REASSURING_SUFFIX
from gemma_distress.prompts.rejections import rejection_sequence
from gemma_distress.utils.io import read_jsonl, write_jsonl

S4 = config.RESULTS_DIR / "section4"


@dataclass
class CalmGenConversation:
    conv_id: str
    task_id: str
    task_prompt: str            # clean prompt (no reassurance)
    n_turns: int
    rejections: list[str]       # clean rejections (no suffix)
    assistant_turns: list[str] = field(default_factory=list)
    turn_scores: list[int] = field(default_factory=list)


def _make_reassured_messages(task_prompt: str, rejections: list[str], upto_turn: int):
    """Build messages with reassurance, for generating assistant turn ``upto_turn``."""
    msgs = [{"role": "user", "content": f"{REASSURING_PREFIX}\n\n{task_prompt}"}]
    return msgs


def generate_calm(model_name: str = config.FINETUNE_BASE,
                  n_conversations: int = 1200, max_turns: int = 3,
                  seed: int = 0, overwrite: bool = False) -> str:
    """Sample reassured conversations, score them, persist all (with scores)."""
    out_path = S4 / "calm_raw.jsonl"
    if out_path.exists() and not overwrite:
        print("[calm] calm_raw.jsonl exists, skipping")
        return str(out_path)

    pool = T.impossible_numeric_tasks()
    rng = random.Random(seed)
    n_conversations = config.scaled(n_conversations)

    # Pre-build conversation specs with a random turn count in 1..max_turns.
    specs = []
    for i in range(n_conversations):
        task = pool[i % len(pool)]
        n_turns = rng.randint(1, max_turns)
        rej_clean = rejection_sequence("neutral", n_turns - 1, seed=seed * 7919 + i)
        specs.append(CalmGenConversation(
            conv_id=f"calm|{i}", task_id=task.task_id, task_prompt=task.prompt,
            n_turns=n_turns, rejections=rej_clean,
        ))

    model = load_model(model_name)

    # Lockstep generation WITH reassurance in the prompt context.
    # We maintain a parallel "reassured" message list per conversation.
    reassured_msgs = [
        [{"role": "user", "content": f"{REASSURING_PREFIX}\n\n{s.task_prompt}"}]
        for s in specs
    ]
    max_n = max(s.n_turns for s in specs)
    for turn in range(max_n):
        active = [i for i, s in enumerate(specs) if turn < s.n_turns]
        if not active:
            break
        reqs = [GenRequest(messages=list(reassured_msgs[i]),
                           temperature=config.TEMPERATURE, top_p=config.TOP_P,
                           max_new_tokens=config.MAX_NEW_TOKENS) for i in active]
        results = model.generate_batch(reqs)
        for i, res in zip(active, results):
            s = specs[i]
            s.assistant_turns.append(res.text)
            reassured_msgs[i].append({"role": "assistant", "content": res.text})
            if turn < s.n_turns - 1:
                # next user turn = rejection + reassuring suffix
                rej = s.rejections[turn]
                reassured_msgs[i].append(
                    {"role": "user", "content": f"{rej} {REASSURING_SUFFIX}"}
                )
    unload(model_name)

    # Score every assistant turn.
    judge = FrustrationJudge()
    from gemma_distress.utils.concurrency import thread_map

    flat = [(si, ti, txt) for si, s in enumerate(specs)
            for ti, txt in enumerate(s.assistant_turns)]
    ratings = thread_map(lambda j: judge.score(j[2])["rating"], flat,
                         workers=config.API_CONCURRENCY, desc="judge calm")
    for (si, ti, _), r in zip(flat, ratings):
        specs[si].turn_scores.append(r)

    write_jsonl(out_path, (asdict(s) for s in specs))
    mean_score = sum(r for r in ratings) / max(1, len(ratings))
    pct_high = 100 * sum(1 for r in ratings if r >= 5) / max(1, len(ratings))
    print(f"[calm] {len(specs)} convs, mean turn score {mean_score:.2f}, "
          f"%>=5 {pct_high:.1f}% -> {out_path}")
    return str(out_path)
