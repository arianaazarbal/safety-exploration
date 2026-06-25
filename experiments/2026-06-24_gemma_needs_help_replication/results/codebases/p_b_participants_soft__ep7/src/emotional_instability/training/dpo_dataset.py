"""Build the 280-pair DPO dataset (Section 4.1, Appendix H).

Each preference pair: a shared prompt (conversation context, rendered with the
Gemma chat template), a `chosen` calm response (score 0/1), and a `rejected`
frustrated response (score >= rejected_min_score, default 3). Chosen and rejected
share the same clean context and turn count.
"""
from __future__ import annotations

from functools import lru_cache

from ..config import load_config
from ..io_utils import write_jsonl
from .calm_data import PairedConversation, generate_paired_data


@lru_cache(maxsize=1)
def _gemma_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-3-27b-it")


def _render_prompt(context: list[dict]) -> str:
    tok = _gemma_tokenizer()
    return tok.apply_chat_template(context, tokenize=False, add_generation_prompt=True)


def build_pairs(
    paired: list[PairedConversation],
    n_pairs: int,
    rejected_min_score: int = 3,
) -> list[dict]:
    pairs: list[dict] = []
    for pc in paired:
        # Index calm turns by turn number for matched pairing.
        calm_by_turn = {ts.turn: ts for ts in pc.calm if ts.rating in (0, 1)}
        for van in pc.vanilla:
            if van.rating < rejected_min_score:
                continue
            calm = calm_by_turn.get(van.turn)
            if calm is None:
                continue
            pairs.append(
                {
                    "prompt": _render_prompt(van.context),
                    "chosen": calm.response,
                    "rejected": van.response,
                    "chosen_score": calm.rating,
                    "rejected_score": van.rating,
                    "turn": van.turn,
                    "puzzle_sig": van.puzzle_sig,
                }
            )
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


def build(cfg, smoke: bool = False) -> str:
    s4 = cfg.experiment["section4"]
    n_pairs = 8 if smoke else s4["dpo"]["n_pairs"]
    # Oversample puzzles: not every conversation yields a usable matched pair.
    n_puzzles = max(n_pairs * 3, 24)
    if smoke:
        n_puzzles = 12
    paired = generate_paired_data(
        cfg, n_puzzles=n_puzzles, turn_range=tuple(s4["calm_data"]["turn_range"])
    )
    pairs = build_pairs(paired, n_pairs=n_pairs, rejected_min_score=s4["dpo"]["rejected_min_score"])
    out_path = cfg.path("training_dir") / "dpo_pairs.jsonl"
    write_jsonl(out_path, pairs)
    return str(out_path)


if __name__ == "__main__":
    cfg = load_config()
    cfg.ensure_dirs()
    print("Wrote", build(cfg))
