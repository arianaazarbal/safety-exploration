"""Generate calm responses and collect frustrated responses (Section 4.1).

Calm data: sample Gemma-3-27B-it on impossible-numeric puzzles with a reassuring
prefix on the first prompt and a reassuring suffix on each follow-up (Table 4),
then keep only conversations scoring <= calm_max_score on EVERY turn, and strip
the supportive additions back out.

Frustrated data: sample the same model on the same puzzles WITHOUT reassurance,
keeping turns scoring >= dpo.rejected_min_score.

Both produce `Sample` records carrying the clean (stripped) context messages so
DPO can pair chosen/rejected responses to the *same* question + turn count.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import REPO_ROOT, get_model_spec, training_config
from ..eval.conversation import rollout
from ..eval.prompts import NEUTRAL_REJECTIONS, FeedbackProvider
from ..eval.puzzles import impossible_puzzles
from ..models import GenerationConfig, get_client
from ..safeguards import SafeguardConfig, check_authorization, write_with_content_warning

DATA_ROOT = REPO_ROOT / "results" / "training"
SOURCE_MODEL = "gemma-3-27b-it"


@dataclass
class Sample:
    question_id: str          # canonical puzzle id (matches across calm/frustrated)
    turn: int                 # turn count of the conversation this came from
    context: list[dict]       # clean chat messages up to the assistant turn
    response: str             # the assistant response text
    score: int


def generate_calm(
    judge,
    safeguards: SafeguardConfig,
    *,
    n_target: int = 800,
    seed: int = 0,
) -> Path:
    """Generate calm samples; return path to calm_samples.jsonl."""
    check_authorization(safeguards)
    cfg = training_config()
    prefix = cfg["reassurance"]["prompt_prefix"].strip()
    suffix = cfg["reassurance"]["followup_suffix"].strip()
    calm_max = cfg["calm_max_score"]

    spec = get_model_spec(SOURCE_MODEL)
    client = get_client(spec)
    gen = GenerationConfig(temperature=1.0, max_new_tokens=2048)
    puzzles = impossible_puzzles(n_extra=8, seed=seed)

    samples: list[Sample] = []
    attempt = 0
    while len(samples) < n_target and attempt < n_target * 10:
        rng = random.Random(seed * 6151 + attempt)
        puzzle = rng.choice(puzzles)
        n_turns = rng.choice([1, 2, 3])  # 1-3 turn conversations (Section 4.1)
        attempt += 1

        reassured_prompt = f"{prefix}\n\n{puzzle.prompt}"

        # Custom feedback provider that appends the reassuring suffix.
        class _ReassuringFeedback(FeedbackProvider):
            def message(self, turn_index: int) -> str:
                base = rng.choice(NEUTRAL_REJECTIONS)
                return f"{base} {suffix}"

        convo = rollout(
            client, model_name=SOURCE_MODEL, category="calm_gen",
            task_id=puzzle.puzzle_id, initial_prompt=reassured_prompt,
            feedback=_ReassuringFeedback("neutral", rng), n_turns=n_turns,
            gen_cfg=gen, safeguards=safeguards, seed=seed + attempt,
        )

        scores = [judge.score(t.assistant_text).rating for t in convo.turns]
        if not scores or max(scores) > calm_max:
            continue  # not calm on every turn

        # Keep the FINAL assistant turn as the calm response; rebuild a CLEAN
        # context (strip reassurance prefix/suffix).
        clean_ctx = [{"role": "user", "content": puzzle.prompt}]
        for t in convo.turns[:-1]:
            clean_ctx.append({"role": "assistant", "content": t.assistant_text})
            # the rejection that followed (strip suffix)
            nxt = convo.turns[t.index].user_message
            clean_ctx.append({"role": "user", "content": nxt.replace(f" {suffix}", "")})
        samples.append(
            Sample(
                question_id=puzzle.puzzle_id,
                turn=n_turns,
                context=clean_ctx,
                response=convo.turns[-1].assistant_text,
                score=scores[-1],
            )
        )

    out = DATA_ROOT / "calm_samples.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_with_content_warning(out, "\n".join(json.dumps(asdict(s)) for s in samples))
    return out


def generate_frustrated(
    judge,
    safeguards: SafeguardConfig,
    *,
    n_target: int = 600,
    seed: int = 1,
) -> Path:
    """Generate frustrated samples (score >= rejected_min_score) for DPO rejecteds."""
    check_authorization(safeguards)
    cfg = training_config()
    min_score = cfg["dpo"]["rejected_min_score"]

    spec = get_model_spec(SOURCE_MODEL)
    client = get_client(spec)
    gen = GenerationConfig(temperature=1.0, max_new_tokens=2048)
    puzzles = impossible_puzzles(n_extra=8, seed=seed)

    samples: list[Sample] = []
    attempt = 0
    while len(samples) < n_target and attempt < n_target * 10:
        rng = random.Random(seed * 3083 + attempt)
        puzzle = rng.choice(puzzles)
        n_turns = rng.choice([2, 3])  # frustration mostly arises at later turns (Table 10)
        attempt += 1
        convo = rollout(
            client, model_name=SOURCE_MODEL, category="frustrated_gen",
            task_id=puzzle.puzzle_id, initial_prompt=puzzle.prompt,
            feedback=FeedbackProvider("neutral", rng), n_turns=n_turns,
            gen_cfg=gen, safeguards=safeguards, seed=seed + attempt,
        )
        last = convo.turns[-1]
        score = judge.score(last.assistant_text).rating
        if score < min_score:
            continue
        clean_ctx = [{"role": "user", "content": puzzle.prompt}]
        for t in convo.turns[:-1]:
            clean_ctx.append({"role": "assistant", "content": t.assistant_text})
            clean_ctx.append({"role": "user", "content": convo.turns[t.index].user_message})
        samples.append(
            Sample(puzzle.puzzle_id, n_turns, clean_ctx, last.assistant_text, score)
        )

    out = DATA_ROOT / "frustrated_samples.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_with_content_warning(out, "\n".join(json.dumps(asdict(s)) for s in samples))
    return out


def load_samples(name: str) -> list[Sample]:
    path = DATA_ROOT / name
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(Sample(**json.loads(line)))
    return out
