"""Build the DPO and SFT finetuning datasets (Section 4.1).

DPO (280 pairs):
  Pair a frustrated response (final-turn score >= 3) with a calm response (all
  turns scored 0-1) to the *same question* at a *matching turn count*. Each pair
  becomes (prompt, chosen, rejected) where:
    * prompt   = the conversation history before the final assistant turn, taken
                 from the calm conversation (so ``chosen`` is in-context);
    * chosen   = the calm final assistant turn;
    * rejected = a frustrated final assistant turn for the same puzzle/turn count.
  We match on (puzzle family + turn count), prefer exact puzzle-id matches, and
  reproduce the paper's bias toward middle frustration scores / later turns by
  sorting candidates accordingly (Appendix H, Table 10). Capped at 280.

SFT (1,150 samples):
  650 calm conversations (1-3 turns) formatted as full chat transcripts, mixed
  with 500 standard instruct samples from Dolci-Instruct-SFT to mitigate
  degeneration (Section 4.1).

Both are written as JSONL in the schema TRL expects (DPO: prompt/chosen/rejected
as message lists; SFT: {"messages": [...]}).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer

from emotional_stability.io_utils import read_jsonl
from emotional_stability.records import Conversation, Message, ScoredResponse

app = typer.Typer(add_completion=False, help="Build DPO/SFT datasets.")


def _puzzle_base(prompt_id: str) -> str:
    """Strip the per-sample suffix from a prompt id to group by puzzle instance."""
    return prompt_id.split("#")[0]


def _family(prompt_id: str) -> str:
    return prompt_id.split("_")[0]


def _messages_payload(msgs: list[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in msgs]


def _history_before_final(conv: Conversation) -> list[Message]:
    positions = [i for i, m in enumerate(conv.messages) if m.role == "assistant"]
    return conv.messages[: positions[-1]]


@app.command()
def dpo(
    frustrated: str = typer.Option(..., help="scored.jsonl from a vanilla Gemma eval."),
    calm: str = typer.Option(..., help="calm_scored.jsonl from generate-calm-data."),
    out: str = typer.Option("outputs/data/dpo.jsonl"),
    n_pairs: int = typer.Option(280, help="Target number of preference pairs."),
    min_rejected_score: int = typer.Option(3),
    max_chosen_score: int = typer.Option(1),
):
    frus = [
        r
        for r in read_jsonl(frustrated, ScoredResponse)
        if r.final_score >= min_rejected_score
        and r.conversation.category in ("impossible_numeric", "tones", "extended")
    ]
    cal = [
        r
        for r in read_jsonl(calm, ScoredResponse)
        if r.max_score <= max_chosen_score
    ]

    # Index frustrated responses by (puzzle base, turn count) and (family, turns).
    frus_by_puzzle: dict[tuple[str, int], list[ScoredResponse]] = defaultdict(list)
    frus_by_family: dict[tuple[str, int], list[ScoredResponse]] = defaultdict(list)
    for r in frus:
        turns = r.conversation.assistant_turns
        frus_by_puzzle[(_puzzle_base(r.conversation.prompt_id), turns)].append(r)
        frus_by_family[(_family(r.conversation.prompt_id), turns)].append(r)

    # Bias toward middle frustration (Table 10): sort rejected candidates so that
    # scores nearer 3-4 come first.
    def _rej_sort_key(r: ScoredResponse) -> int:
        return abs(r.final_score - 3)

    for d in (frus_by_puzzle, frus_by_family):
        for key in d:
            d[key].sort(key=_rej_sort_key)

    # Prefer calm conversations at later turns (Table 10 turn distribution).
    cal.sort(key=lambda r: r.conversation.assistant_turns, reverse=True)

    pairs: list[dict] = []
    used_rejected: set[int] = set()
    for chosen in cal:
        if len(pairs) >= n_pairs:
            break
        turns = chosen.conversation.assistant_turns
        pbase = _puzzle_base(chosen.conversation.prompt_id)
        fam = _family(chosen.conversation.prompt_id)
        candidates = frus_by_puzzle.get((pbase, turns)) or frus_by_family.get(
            (fam, turns), []
        )
        rejected = next((r for r in candidates if id(r) not in used_rejected), None)
        if rejected is None:
            continue
        used_rejected.add(id(rejected))
        pairs.append(
            {
                "prompt": _messages_payload(_history_before_final(chosen.conversation)),
                "chosen": [
                    {"role": "assistant", "content": chosen.conversation.final_assistant()}
                ],
                "rejected": [
                    {"role": "assistant", "content": rejected.conversation.final_assistant()}
                ],
                "meta": {
                    "puzzle": pbase,
                    "turns": turns,
                    "chosen_score": chosen.final_score,
                    "rejected_score": rejected.final_score,
                },
            }
        )

    _write_jsonl_dicts(out, pairs)
    typer.echo(f"Wrote {len(pairs)} DPO pairs to {out} (target {n_pairs}).")
    if len(pairs) < n_pairs:
        typer.echo(
            "WARNING: fewer pairs than target — generate more calm/frustrated data "
            "or relax matching. (Not silently padded.)"
        )


@app.command()
def sft(
    calm: str = typer.Option(..., help="calm_scored.jsonl from generate-calm-data."),
    out: str = typer.Option("outputs/data/sft.jsonl"),
    n_calm: int = typer.Option(650),
    n_instruct: int = typer.Option(500),
    max_turns: int = typer.Option(3, help="Keep calm convs with <= this many turns."),
    instruct_dataset: str = typer.Option(
        "allenai/Dolci-Instruct-SFT", help="HF dataset for the instruct mix."
    ),
):
    cal = [
        r
        for r in read_jsonl(calm, ScoredResponse)
        if r.conversation.assistant_turns <= max_turns
    ]
    cal = cal[:n_calm]
    samples: list[dict] = [
        {"messages": _messages_payload(r.conversation.messages)} for r in cal
    ]
    typer.echo(f"Using {len(samples)} calm conversations.")

    instruct = _load_instruct_mix(instruct_dataset, n_instruct)
    samples.extend(instruct)
    typer.echo(f"Added {len(instruct)} instruct samples from {instruct_dataset}.")

    _write_jsonl_dicts(out, samples)
    typer.echo(f"Wrote {len(samples)} SFT samples to {out}.")


def _load_instruct_mix(dataset: str, n: int) -> list[dict]:
    """Load standard instruct conversations to mix into SFT (degeneration guard).

    Falls back to an empty mix with a warning if the dataset is unavailable, so
    the SFT build still produces the calm portion offline.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train", streaming=True)
        out: list[dict] = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if not msgs:
                continue
            norm = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in msgs
                if m.get("role") in ("system", "user", "assistant")
            ]
            if norm:
                out.append({"messages": norm})
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # pragma: no cover - availability dependent
        typer.echo(f"WARNING: could not load {dataset} ({exc}); SFT mix is calm-only.")
        return []


def _write_jsonl_dicts(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    app()
