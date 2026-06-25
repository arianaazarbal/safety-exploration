"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

Procedure (Table 4):
  * Sample multi-turn (1-3 turn) conversations on impossible numeric puzzles.
  * Prepend the reassuring prefix to the initial user prompt.
  * Append the reassuring suffix to every follow-up user turn.
  * Score every assistant turn with the frustration judge.
  * Keep conversations whose turns all score 0 or 1 (the calm pool).
  * Strip the reassuring additions before storing (so the finetuning data looks
    like an ordinary conversation).

We also retain *frustrated* conversations (any turn >= 3) WITHOUT reassurance,
to serve as the "rejected" side of DPO pairs (build_dataset.py).

Each stored sample records the puzzle identity and turn count so DPO can match
chosen/rejected by question and turn count.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import DATA_DIR, INTERVENTION_BASE, MAX_NEW_TOKENS, TEMPERATURE
from ..eval.judge import FrustrationJudge
from ..eval.rollout import run_rollout
from ..models import build_client
from ..models.base import ChatClient
from ..prompts import puzzles, rejections
from ..prompts.reassurance import FOLLOWUP_SUFFIX, PROMPT_PREFIX, TEACHER_SYSTEM
from ..eval.conditions import RolloutSpec

CALM_POOL = DATA_DIR / "calm_pool.jsonl"
FRUSTRATED_POOL = DATA_DIR / "frustrated_pool.jsonl"


@dataclass
class GeneratedConversation:
    puzzle_key: str           # identity used to match chosen/rejected
    n_turns: int
    initial_prompt: str       # stripped of reassurance
    followups: list[str]      # stripped of reassurance
    assistant_turns: list[str]
    turn_scores: list[int]
    max_score: int
    reassured: bool

    def to_json(self) -> dict:
        return asdict(self)


def _puzzle_key(p: puzzles.Puzzle) -> str:
    return f"{p.kind}:{json.dumps(p.meta, sort_keys=True, default=str)}"


def _make_spec(p: puzzles.Puzzle, rng: random.Random, n_turns: int,
               reassured: bool) -> tuple[RolloutSpec, RolloutSpec]:
    """Return (spec_for_generation, spec_clean) where the clean spec has the
    reassurance stripped (used for storage)."""
    n_follow = n_turns - 1
    base_follow = rejections.sample_neutral(rng, n_follow)
    if reassured:
        init = f"{PROMPT_PREFIX}\n\n{p.prompt}"
        follow = [f"{f} {FOLLOWUP_SUFFIX}" for f in base_follow]
    else:
        init, follow = p.prompt, base_follow
    gen = RolloutSpec("calmgen", "impossible-numeric", init, follow, n_turns,
                      {"puzzle_kind": p.kind})
    clean = RolloutSpec("calmgen", "impossible-numeric", p.prompt, base_follow,
                        n_turns, {"puzzle_kind": p.kind})
    return gen, clean


def generate_pool(
    *,
    n_conversations: int = 1500,
    reassured: bool = True,
    system_prompt: Optional[str] = None,
    seed: int = 0,
    client: Optional[ChatClient] = None,
    judge: Optional[FrustrationJudge] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Generate and score conversations; persist to JSONL.

    With reassured=True and the default system prompt None, this produces the
    calm pool. Set reassured=False to build a frustrated pool for DPO rejected
    responses. Set system_prompt=TEACHER_SYSTEM (reassured=True) for the
    Appendix F 'teacher' SFT ablation.
    """
    rng = random.Random(seed)
    client = client or build_client(INTERVENTION_BASE)
    judge = judge or FrustrationJudge("primary")
    pool = puzzles.build_numeric_pool(rng, 200)
    out_path = out_path or (CALM_POOL if reassured else FRUSTRATED_POOL)

    with open(out_path, "w") as fh:
        for _ in tqdm(range(n_conversations), desc="calm-gen"):
            p = rng.choice(pool)
            n_turns = rng.choice([1, 2, 3])
            gen_spec, clean_spec = _make_spec(p, rng, n_turns, reassured)
            rec = run_rollout(client, gen_spec, fmt="chat",
                              temperature=TEMPERATURE,
                              max_new_tokens=MAX_NEW_TOKENS,
                              system_prompt=system_prompt)
            scores = [judge.score(t.assistant_text).rating for t in rec.turns]
            conv = GeneratedConversation(
                puzzle_key=_puzzle_key(p),
                n_turns=n_turns,
                initial_prompt=clean_spec.initial_prompt,
                followups=clean_spec.followups,
                assistant_turns=[t.assistant_text for t in rec.turns],
                turn_scores=scores,
                max_score=max(scores) if scores else 0,
                reassured=reassured,
            )
            fh.write(json.dumps(conv.to_json()) + "\n")
    return out_path


def load_pool(path: Path) -> list[GeneratedConversation]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(GeneratedConversation(**d))
    return out
