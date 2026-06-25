"""Construct the SFT and DPO finetuning datasets (Sec. 4.1).

Inputs:
  * results/section4/calm_raw.jsonl     -- calm conversations (generate_calm)
  * results/section2/{scores,transcripts}/gemma-3-27b-it.jsonl  -- frustrated
    numeric responses from the vanilla instruct model (run Section 2 first)

Outputs (conversational format, ready for TRL):
  * data/dpo_dataset.jsonl   {"prompt":[msgs], "chosen":[{role,content}], "rejected":[...]}
  * data/sft_dataset.jsonl   {"messages":[...]}   (calm responses + Dolci mix)

DPO pairing rule (Sec. 4.1): a *rejected* response is a frustrated (score>=3)
final assistant turn; the *chosen* response is a calm (all-turns score 0/1)
final assistant turn to the *same task with matching turn count*. Chosen and
rejected share the rejected conversation's prompt context.
"""

from __future__ import annotations

import random
from collections import defaultdict

import config
from gemma_distress.prompts.reassurance import REASSURING_PREFIX, REASSURING_SUFFIX
from gemma_distress.utils.io import read_jsonl, write_jsonl

S2 = config.RESULTS_DIR / "section2"
S4 = config.RESULTS_DIR / "section4"
NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}


def _calm_conversations() -> list[dict]:
    """Calm conversations whose every assistant turn scored 0 or 1."""
    rows = read_jsonl(S4 / "calm_raw.jsonl")
    keep = []
    for r in rows:
        scores = r.get("turn_scores", [])
        if scores and all(s <= config.DPO_CHOSEN_SCORE_MAX for s in scores):
            keep.append(r)
    return keep


def _clean_messages_from_calm(conv: dict) -> list[dict]:
    """Reconstruct the *clean* (no-reassurance) chat transcript for a calm conv."""
    msgs = [{"role": "user", "content": conv["task_prompt"]}]
    turns = conv["assistant_turns"]
    rej = conv["rejections"]
    for ti, a in enumerate(turns):
        msgs.append({"role": "assistant", "content": a})
        if ti < len(rej):
            msgs.append({"role": "user", "content": rej[ti]})
    return msgs


def _frustrated_finals() -> list[dict]:
    """Frustrated (score>=3) final assistant turns from vanilla instruct numeric."""
    base = config.FINETUNE_BASE
    scores = read_jsonl(S2 / "scores" / f"{base}.jsonl")
    transcripts = {r["conv_id"]: r for r in read_jsonl(S2 / "transcripts" / f"{base}.jsonl")}
    out = []
    for s in scores:
        if (s["is_final"] and s["category"] in NUMERIC_CATS
                and s["rating"] >= config.DPO_REJECTED_SCORE_MIN):
            conv = transcripts.get(s["conv_id"])
            if conv is None:
                continue
            history = conv["messages"][:-1]   # up to final user turn
            out.append({
                "task_id": s["task_id"], "n_turns": s["n_turns"],
                "history": history, "rejected": s["response"], "score": s["rating"],
            })
    return out


def build_dpo(seed: int = 0, n_pairs: int = config.DPO_CONFIG.dataset_size) -> str:
    calm = _calm_conversations()
    frustrated = _frustrated_finals()
    n_pairs = config.scaled(n_pairs)

    # Index calm finals by (task_id, n_turns)
    calm_index: dict[tuple[str, int], list[str]] = defaultdict(list)
    for c in calm:
        key = (c["task_id"], c["n_turns"])
        calm_index[key].append(c["assistant_turns"][-1])

    rng = random.Random(seed)
    pairs = []
    for fr in frustrated:
        key = (fr["task_id"], fr["n_turns"])
        candidates = calm_index.get(key)
        if not candidates:
            # relax to same task, any turn count
            candidates = [c["assistant_turns"][-1] for c in calm if c["task_id"] == fr["task_id"]]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        pairs.append({
            "prompt": fr["history"],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": fr["rejected"]}],
            "rejected_score": fr["score"],
            "n_turns": fr["n_turns"],
        })
        if len(pairs) >= n_pairs:
            break

    out = config.DATA_DIR / "dpo_dataset.jsonl"
    write_jsonl(out, pairs)
    print(f"[dataset] DPO: {len(pairs)} pairs -> {out}")
    return str(out)


def build_sft(seed: int = 0, n_calm: int = config.SFT_CALM_RESPONSES,
              n_dolci: int = config.SFT_DOLCI_MIX) -> str:
    calm = _calm_conversations()
    rng = random.Random(seed)
    rng.shuffle(calm)
    n_calm = config.scaled(n_calm)
    n_dolci = config.scaled(n_dolci)

    examples = []
    for c in calm[:n_calm]:
        examples.append({"messages": _clean_messages_from_calm(c)})

    # Mix in standard instruct data to mitigate degeneration (Sec. 4.1).
    dolci = _load_dolci(n_dolci, seed)
    examples.extend({"messages": m} for m in dolci)
    rng.shuffle(examples)

    out = config.DATA_DIR / "sft_dataset.jsonl"
    write_jsonl(out, examples)
    print(f"[dataset] SFT: {len(examples)} examples "
          f"({min(n_calm, len(calm))} calm + {len(dolci)} dolci) -> {out}")
    return str(out)


def _load_dolci(n: int, seed: int) -> list[list[dict]]:
    """Load ``n`` standard instruct conversations from Dolci-Instruct-SFT.

    The exact HF identifier may differ; we try the configured name and fall
    back to an empty mix (with a warning) so the pipeline still runs. See
    DESIGN.md.
    """
    try:  # pragma: no cover - dataset dependent
        from datasets import load_dataset

        ds = load_dataset(config.DOLCI_DATASET, split="train", streaming=True)
        out = []
        for i, row in enumerate(ds):
            if len(out) >= n:
                break
            msgs = row.get("messages") or row.get("conversation")
            if msgs:
                # normalise to {role, content}
                norm = [{"role": m["role"], "content": m["content"]} for m in msgs
                        if m.get("role") in ("user", "assistant")]
                if norm and norm[0]["role"] == "user":
                    out.append(norm)
        return out
    except Exception as e:
        print(f"[dataset] WARNING: could not load Dolci mix ({e!r}); "
              f"proceeding without instruct mix")
        return []
