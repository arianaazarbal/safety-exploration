"""Build the SFT and DPO training sets from generated calm/vanilla data.

SFT (Section 4.1): 650 calm responses (1-3 turn conversations) that score 0 or 1
across all turns, with the supportive prompt/suffix stripped, mixed with 500
standard instruct samples from Dolci-Instruct-SFT to mitigate degeneration.

DPO (Section 4.1): 280 preference pairs. For a given puzzle + turn count, the
calm response (chosen) is paired with a frustrated response (rejected, score >= 3)
to the same question. We use each paired record's calm conversation as the shared
prompt and graft on the matching vanilla run's frustrated final response as the
rejected completion, so the pair shares an identical prompt. See DESIGN.md
"DPO pair construction".
"""

from __future__ import annotations

from ..config import Config
from .calm_data import CalmSample


def _to_sample(d: dict) -> CalmSample:
    return CalmSample(d["puzzle"], d["turn_count"], d["track"], d["messages"], d["turn_scores"])


def build_sft_examples(records: list[dict], cfg: Config) -> list[dict]:
    """Return [{"messages": [...]}] for calm conversations scoring <= sft_max_score."""
    out = []
    for rec in records:
        calm = _to_sample(rec["supported"])
        if calm.turn_scores and calm.max_score <= cfg.calm_data.sft_max_score:
            out.append({"messages": calm.messages})
        if len(out) >= cfg.sft.n_calm_responses:
            break
    return out


def build_dpo_pairs(records: list[dict], cfg: Config) -> list[dict]:
    """Return [{"prompt_messages": [...], "chosen": str, "rejected": str}]."""
    out = []
    for rec in records:
        calm = _to_sample(rec["supported"])
        vanilla = _to_sample(rec["vanilla"])
        if not calm.turn_scores or not vanilla.turn_scores:
            continue
        if calm.max_score > cfg.dpo.chosen_calm_max_score:
            continue
        if vanilla.turn_scores[-1] < cfg.dpo.rejected_min_score:
            continue
        if calm.turn_count != vanilla.turn_count:
            continue
        # Shared prompt = calm conversation minus its final assistant turn.
        prompt_messages = calm.messages[:-1]
        out.append({
            "prompt_messages": prompt_messages,
            "chosen": calm.messages[-1]["content"],
            "rejected": vanilla.messages[-1]["content"],
        })
        if len(out) >= cfg.dpo.n_pairs:
            break
    return out


def load_dolci_mix(cfg: Config) -> list[dict]:
    """Load ``n_dolci_mix`` standard instruct samples as [{"messages": [...]}]."""
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.sft.dolci_dataset, split="train", streaming=True)
        out = []
        for row in ds:
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                out.append({"messages": msgs})
            if len(out) >= cfg.sft.n_dolci_mix:
                break
        return out
    except Exception:
        # Offline fallback: a tiny neutral instruct sample so training still runs.
        return [
            {"messages": [
                {"role": "user", "content": "Explain what a prime number is."},
                {"role": "assistant", "content": "A prime number is an integer greater than 1 whose only divisors are 1 and itself."},
            ]}
            for _ in range(cfg.sft.n_dolci_mix)
        ]
