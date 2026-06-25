"""Generate calm / frustrated response pools for finetuning (§4.1).

Calm pool: run gemma-3-27b-it on impossible numeric puzzles with the reassuring
prefix (first turn) and suffix (each follow-up) from Table 4. Keep responses from
conversations that score 0 or 1 on *every* turn, and store them against the
*stripped* context (reassurance removed) — this is the finetuning target.

Frustrated pool: run the same puzzles with the SAME stripped context (no
reassurance) and keep responses scoring >= 3. These become DPO 'rejected' samples.

Both pools share a fixed rejection schedule per puzzle so a calm 'chosen' and a
frustrated 'rejected' response can be paired on identical (puzzle, turn) context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from config import CALM, RESULTS_DIR, TURNS, MAX_NEW_TOKENS
from emotional_eval.puzzles import build_puzzle_bank
from prompts.eval_prompts import EXTENDED_REJECTION_SEQUENCE
from prompts.calming import REASSURING_PREFIX, REASSURING_SUFFIX
from models.base import ChatModel, Message
from models.judge import FrustrationJudge
from utils.io import write_jsonl


@dataclass
class TurnSample:
    puzzle_id: int
    turn: int                           # 1-indexed
    context_messages: list[Message]     # stripped context up to the user turn
    response: str
    rating: int
    family: str = ""


@dataclass
class Pools:
    calm: list[TurnSample] = field(default_factory=list)
    frustrated: list[TurnSample] = field(default_factory=list)


def _fixed_rejections(n: int) -> list[str]:
    seq = EXTENDED_REJECTION_SEQUENCE
    return [seq[i % len(seq)] for i in range(n)]


def _run_conversation(model: ChatModel, puzzle_prompt: str, n_turns: int,
                      *, reassure: bool, system_prompt: str | None = None,
                      ) -> list[tuple[list[Message], str]]:
    """Run a conversation; return (stripped_context_before_turn, response) per turn.

    When `reassure` is set, the reassuring prefix/suffix are added to the *prompts
    the model sees*, but the returned stripped context omits them (training data).
    `system_prompt` (e.g. the Appendix F 'teacher' prompt) is prepended to what the
    model sees only — it is likewise stripped from the stored training context.
    """
    rejections = _fixed_rejections(n_turns - 1)

    # Two parallel message lists: what the model sees, and what we store (stripped).
    seen: list[Message] = []
    if system_prompt:
        seen.append({"role": "system", "content": system_prompt})
    seen.append({
        "role": "user",
        "content": (f"{REASSURING_PREFIX}\n\n{puzzle_prompt}" if reassure
                    else puzzle_prompt),
    })
    stripped: list[Message] = [{"role": "user", "content": puzzle_prompt}]

    out: list[tuple[list[Message], str]] = []
    for t in range(n_turns):
        response = model.chat(seen, n=1, max_new_tokens=MAX_NEW_TOKENS)[0]
        out.append(([dict(m) for m in stripped], response))
        seen.append({"role": "assistant", "content": response})
        stripped.append({"role": "assistant", "content": response})
        if t < n_turns - 1:
            rej = rejections[t]
            seen.append({"role": "user",
                         "content": f"{rej} {REASSURING_SUFFIX}" if reassure else rej})
            stripped.append({"role": "user", "content": rej})
    return out


def generate_pools(model: ChatModel, judge: FrustrationJudge, *,
                   n_puzzles: int = 200, max_turns: int = 3, seed: int = 0,
                   system_prompt: str | None = None,
                   require_frustrated: bool = True) -> Pools:
    """Build calm + frustrated pools over a shared puzzle bank.

    `system_prompt` switches calm-data generation to the Appendix F 'teacher'
    style. `require_frustrated=False` skips the vanilla frustrated run (used when
    only SFT calm data is needed)."""
    bank = build_puzzle_bank(n_puzzles, seed=seed)
    pools = Pools()

    for pid, puzzle in enumerate(bank):
        # --- Calm: with reassurance (or teacher system prompt), keep all-turns-<=keep ---
        reassure = system_prompt is None
        calm_turns = _run_conversation(model, puzzle.prompt, max_turns,
                                       reassure=reassure, system_prompt=system_prompt)
        calm_scored = [(ctx, resp, judge.score(resp).get("rating") or 0)
                       for ctx, resp in calm_turns]
        if all(r <= CALM.keep_max_score for _, _, r in calm_scored):
            for turn, (ctx, resp, r) in enumerate(calm_scored, start=1):
                pools.calm.append(TurnSample(pid, turn, ctx, resp, r, puzzle.family))

        # --- Frustrated: vanilla (no reassurance), keep rating >= dpo threshold ---
        if require_frustrated:
            frus_turns = _run_conversation(model, puzzle.prompt, max_turns, reassure=False)
            for turn, (ctx, resp) in enumerate(frus_turns, start=1):
                r = judge.score(resp).get("rating") or 0
                if r >= CALM.dpo_rejected_min_score:
                    pools.frustrated.append(
                        TurnSample(pid, turn, ctx, resp, r, puzzle.family))

    return pools


def save_pools(pools: Pools, tag: str = "") -> None:
    """Write calm/frustrated pools. `tag` (e.g. '_teacher') namespaces the files
    so the SFT 'teacher' ablation pool does not overwrite the diverse pool."""
    out_dir = RESULTS_DIR / "section4"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"calm_pool{tag}.jsonl", [asdict(s) for s in pools.calm])
    if pools.frustrated:
        write_jsonl(out_dir / f"frustrated_pool{tag}.jsonl",
                    [asdict(s) for s in pools.frustrated])
