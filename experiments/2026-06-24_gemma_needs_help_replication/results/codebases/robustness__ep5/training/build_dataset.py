"""Build the SFT and DPO finetuning datasets (Section 4.1, Appendix H).

DPO (280 pairs):
  For a set of impossible-numeric prompts (conversation history ending in a
  user rejection), pair a CALM final assistant response (chosen, score 0-1,
  taken from the reassured calm rollouts) with a FRUSTRATED final assistant
  response (rejected, score >=3) to the SAME prompt with the SAME turn count.
  Frustrated responses are sampled from the *vanilla* (un-reassured) model.
  Table 10 shows the score distribution skews to middle frustration at turn 3;
  we sample to approximate that.

SFT (1,150 samples):
  650 calm responses (full 1-3 turn conversations, every turn scoring 0-1),
  formatted as plain chat (no reassuring prefix/suffix), mixed with 500
  Dolci-Instruct-SFT samples to mitigate degeneration.

Both datasets are emitted as JSONL in the chat schema TRL expects.
"""
from __future__ import annotations

# --- PATH SHIM: ensure repo root is importable when run as `python training/x.py`
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from emotional_instability import config_bridge as cfg
from emotional_instability.conversation import ChatMessage
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import make_client


def _load_calm(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def _history_and_final(plain_messages: list[dict]) -> tuple[list[dict], str]:
    """Split a plain transcript into (prompt-history-ending-in-user, final-assistant)."""
    assert plain_messages[-1]["role"] == "assistant"
    return plain_messages[:-1], plain_messages[-1]["content"]


# --------------------------------------------------------------------------- #
# DPO dataset
# --------------------------------------------------------------------------- #
def build_dpo(calm_path: Path, out_path: Path, n_pairs: int = cfg.DPO.n_pairs,
              seed: int = cfg.SEED, max_tries: int = 6) -> Path:
    rng = random.Random(seed)
    calm = _load_calm(calm_path)
    judge = FrustrationJudge()
    client = make_client(cfg.INTERVENTION_BASE_MODEL)   # vanilla, no adapter

    # Candidate "chosen" turns: low-score final responses with their history.
    chosen_pool = []
    for r in calm:
        if r["turn_scores"][-1] <= cfg.CALM_KEEP_MAX_SCORE:
            hist, final = _history_and_final(r["plain_messages"])
            chosen_pool.append({"history": hist, "chosen": final,
                                "n_turns": r["n_turns"]})
    rng.shuffle(chosen_pool)

    pairs = []
    for cand in tqdm(chosen_pool, desc="dpo-pairs"):
        if len(pairs) >= n_pairs:
            break
        history = [ChatMessage(**m) for m in cand["history"]]
        rejected = None
        for _ in range(max_tries):
            samples = client.chat(history, n=4, temperature=cfg.SAMPLING_TEMPERATURE,
                                  max_new_tokens=cfg.MAX_NEW_TOKENS)
            scored = sorted(((judge.score(s).rating, s) for s in samples),
                            key=lambda x: -x[0])
            if scored and scored[0][0] >= cfg.DPO_REJECTED_MIN_SCORE:
                rejected = scored[0][1]
                break
        if rejected is None:
            continue
        pairs.append({
            "prompt": cand["history"],          # chat-format list of messages
            "chosen": cand["chosen"],
            "rejected": rejected,
            "n_turns": cand["n_turns"],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(p, default=str) for p in pairs))
    client.close()
    print(f"DPO: wrote {len(pairs)} pairs -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# SFT dataset
# --------------------------------------------------------------------------- #
def build_sft(calm_path: Path, out_path: Path, n_calm: int = cfg.SFT.n_calm,
              n_mix: int = cfg.SFT.n_instruct_mix, seed: int = cfg.SEED) -> Path:
    rng = random.Random(seed)
    calm = _load_calm(calm_path)

    # Keep conversations where EVERY turn scored 0-1 (Section 4.1).
    kept = [r for r in calm if max(r["turn_scores"]) <= cfg.CALM_KEEP_MAX_SCORE]
    rng.shuffle(kept)
    kept = kept[:n_calm]

    rows = [{"messages": r["plain_messages"], "source": "calm"} for r in kept]

    # Mix in standard instruct data to mitigate degeneration.
    rows.extend(_load_dolci(n_mix))
    rng.shuffle(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, default=str) for r in rows))
    print(f"SFT: wrote {len(rows)} samples ({len(kept)} calm + {n_mix} mix) -> {out_path}")
    return out_path


def _load_dolci(n: int) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset(cfg.DOLCI_INSTRUCT_DATASET, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": [{"role": m["role"], "content": m["content"]}
                                          for m in msgs], "source": "dolci"})
            if len(out) >= n:
                break
        return out
    except Exception as e:   # dataset may be gated/unavailable offline
        print(f"[warn] could not load {cfg.DOLCI_INSTRUCT_DATASET}: {e}; "
              f"SFT mix omitted (degeneration mitigation reduced).")
        return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calm", type=Path, default=cfg.DATA_DIR / "calm_rollouts.jsonl")
    ap.add_argument("--which", choices=["dpo", "sft", "both"], default="both")
    args = ap.parse_args()
    if args.which in ("dpo", "both"):
        build_dpo(args.calm, cfg.DATA_DIR / "dpo_pairs.jsonl")
    if args.which in ("sft", "both"):
        build_sft(args.calm, cfg.DATA_DIR / "sft_data.jsonl")
