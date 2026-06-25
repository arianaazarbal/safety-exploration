"""Build the SFT dataset (Section 4.1).

650 calm responses (1-3 turn conversations) mixed with 500 standard
instruct samples from Dolci-Instruct-SFT to mitigate degeneration. Calm
conversations are the all-0/1 reassured rollouts; we strip the reassuring
additions and keep the full multi-turn conversation as an SFT target.

Output: conversational-format JSONL consumable by TRL's ``SFTTrainer``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ..config import Config
from ..eval.schemas import read_jsonl

log = logging.getLogger(__name__)


def _calm_conversations(scored_jsonl, chosen_max: int) -> list[list[dict]]:
    """All-0/1 reassured conversations as message lists (reassurance stripped)."""
    convs = []
    for r in read_jsonl(scored_jsonl):
        if r.category != "calm_data" or r.condition != "reassured":
            continue
        scores = r.scores()
        if not scores or max(scores) > chosen_max:
            continue
        # Reconstruct the conversation from raw user/assistant turns. We use the
        # *vanilla* user text (raw puzzle + rejection) so the reassuring prefix /
        # suffix do not appear in the SFT target — but the reassured run stored
        # the additions in `.user`, so strip the trailing suffix heuristically.
        msgs = []
        for t in r.conversation.turns:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        convs.append(msgs)
    return convs


def _instruct_samples(dataset_name: str, n: int, seed: int) -> list[list[dict]]:
    """Load n conversational samples from the instruct mix dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
            if len(out) >= n:
                break
        return out
    except Exception as exc:  # pragma: no cover - dataset/network dependent
        log.warning("Could not load instruct mix %s (%s); proceeding without it.",
                    dataset_name, exc)
        return []


def build_sft_dataset(
    scored_jsonl: str | Path,
    out_path: str | Path,
    cfg: Config | None = None,
    seed: int = 0,
) -> list[dict]:
    cfg = cfg or Config.load("training")
    scfg = cfg.get("sft", {})
    n_calm = int(scfg.get("n_calm", 650))
    n_mix = int(scfg.get("n_instruct_mix", 500))
    chosen_max = int(cfg.get("dpo", {}).get("chosen_max_score", 1))
    instruct_ds = scfg.get("instruct_dataset", "allenai/Dolci-Instruct-SFT")

    rng = random.Random(seed)
    calm = _calm_conversations(scored_jsonl, chosen_max)
    rng.shuffle(calm)
    calm = calm[:n_calm]
    mix = _instruct_samples(instruct_ds, n_mix, seed)

    examples = [{"messages": m} for m in calm] + [{"messages": m} for m in mix]
    rng.shuffle(examples)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("Wrote %d SFT examples (%d calm + %d instruct) to %s",
             len(examples), len(calm), len(mix), out_path)
    return examples
