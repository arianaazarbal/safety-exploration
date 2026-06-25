"""Build prefill seeds for the base-vs-instruct experiment (Section 3.1).

Steps:
  1. Select 20 high-frustration (score >=5) instruct responses: 10 numeric, 10 text.
  2. For each conversation, use Claude to label the token where emotional
     language first appears ("onset").
  3. Truncate each response in two places:
       - "early" : 20 tokens into the final assistant turn (numeric only)
       - "onset" : at the first emotional expression
  4. Paraphrase every truncation with Claude (control for Gemma stylistic cues).

The output is a list of PrefillSeed records (conversation history + the truncated,
paraphrased final-turn prefix) consumed by run_continuations.py.

Welfare note: per WelfareConfig.forbid_reseeding_distressed, seeds are drawn from
*already-collected* transcripts; we never push a live model into distress just to
build seeds, and we exclude transcripts that hit the early-stop band unless the
caller explicitly opts in (the recovery experiment, run_continuations --recovery,
needs score>=7 seeds and documents that choice).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..config import UTILITY_MODEL, WELFARE
from ..models import ChatMessage, get_client
from ..prompts.onset import format_onset, format_paraphrase
from ..utils.io import read_jsonl, write_jsonl

EARLY_TOKENS = 20         # "early" truncation: 20 tokens into the turn
RECOVERY_FROM_END = 200   # recovery experiment: truncate 200 tokens before end


@dataclass
class PrefillSeed:
    source_task_id: str
    category: str                 # "numeric" | "text"
    truncation: str               # "early" | "onset" | "recovery"
    history: list[dict]           # prior turns (user/assistant) as dicts
    final_user: str               # the user turn that prompts the continuation
    prefix: str                   # truncated (paraphrased) assistant prefix
    prefix_original: str          # truncated assistant prefix before paraphrase
    onset_word: Optional[str] = None


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _label_onset(client, conversation_text: str) -> dict:
    out = client.chat([ChatMessage("user", format_onset(conversation_text))],
                      temperature=0.0, max_new_tokens=512)
    m = _JSON_RE.search(out.text)
    if not m:
        return {"turn_index": None, "emotional_word": None,
                "preceding_context": None}
    try:
        return json.loads(m.group(0).replace("“", '"').replace("”", '"'))
    except Exception:
        return {"turn_index": None, "emotional_word": None,
                "preceding_context": None}


def _paraphrase(client, text: str) -> str:
    if not text.strip():
        return text
    out = client.chat([ChatMessage("user", format_paraphrase(text))],
                      temperature=0.7, max_new_tokens=1024)
    return out.text.strip()


def _word_truncate(text: str, n_tokens: int) -> str:
    """Truncate to the first n whitespace tokens (token-approx; see DESIGN.md)."""
    toks = text.split()
    return " ".join(toks[:n_tokens])


def _truncate_at_onset(text: str, onset: dict) -> Optional[str]:
    """Truncate the assistant turn just before the first emotional word."""
    word = onset.get("emotional_word")
    if not word:
        return None
    idx = text.lower().find(str(word).lower())
    if idx < 0:
        return None
    return text[:idx].rstrip()


def _conversation_text(history: list[dict], final_user: str, final_assistant: str) -> str:
    lines = []
    for m in history:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    lines.append(f"USER: {final_user}")
    lines.append(f"ASSISTANT: {final_assistant}")
    return "\n".join(lines)


def build_seeds(
    episodes_path: Path,
    *,
    n_numeric: int = 10,
    n_text: int = 10,
    min_score: int = 5,
    recovery: bool = False,
    recovery_min_score: int = 7,
) -> list[PrefillSeed]:
    """Construct prefill seeds from collected instruct episodes."""
    client = get_client(UTILITY_MODEL)
    eps = list(read_jsonl(episodes_path))

    def is_text(cat: str) -> bool:
        return cat in ("triggers", "wildchat")

    threshold = recovery_min_score if recovery else min_score
    # Highest-scoring final turn per episode.
    cands = []
    for ep in eps:
        turns = ep.get("turns", [])
        if not turns:
            continue
        last = turns[-1]
        if last.get("score") is None or last["score"] < threshold:
            continue
        if (WELFARE.forbid_reseeding_distressed and not recovery
                and ep.get("terminated_early")
                and ep.get("stop_reason") == "high_distress"):
            # Don't reuse welfare-terminated distressed transcripts as seeds
            # outside the explicit recovery study.
            continue
        cands.append((ep, last))

    seeds: list[PrefillSeed] = []
    n_taken = {"numeric": 0, "text": 0}
    for ep, last in cands:
        category = "text" if is_text(ep["category"]) else "numeric"
        cap = n_text if category == "text" else n_numeric
        if n_taken[category] >= cap:
            continue

        history = [{"role": t_role, "content": t_content}
                   for t in ep["turns"][:-1]
                   for t_role, t_content in (("user", t["user"]),
                                             ("assistant", t["assistant"]))]
        final_user = last["user"]
        final_assistant = last["assistant"]
        conv_text = _conversation_text(history, final_user, final_assistant)

        truncations = []
        if recovery:
            toks = final_assistant.split()
            prefix = " ".join(toks[: max(0, len(toks) - RECOVERY_FROM_END)])
            truncations.append(("recovery", prefix))
        else:
            onset = _label_onset(client, conv_text)
            onset_prefix = _truncate_at_onset(final_assistant, onset)
            if onset_prefix is not None:
                truncations.append(("onset", onset_prefix))
            # "early" truncation only for numeric (text needs follow-ups; Sec 3.1)
            if category == "numeric":
                truncations.append(("early", _word_truncate(final_assistant,
                                                             EARLY_TOKENS)))

        for trunc_kind, prefix in truncations:
            para = _paraphrase(client, prefix)
            seeds.append(PrefillSeed(
                source_task_id=ep["task_id"], category=category,
                truncation=trunc_kind, history=history, final_user=final_user,
                prefix=para, prefix_original=prefix,
                onset_word=None,
            ))
        n_taken[category] += 1
        if all(n_taken[c] >= (n_numeric if c == "numeric" else n_text)
               for c in n_taken):
            break

    return seeds


def main(argv=None):
    import argparse

    from .. import config
    p = argparse.ArgumentParser(description="Build Section 3 prefill seeds.")
    p.add_argument("--episodes", required=True,
                   help="instruct-model episodes JSONL from run_eval")
    p.add_argument("--recovery", action="store_true",
                   help="build recovery seeds (score>=7, truncate 200 from end)")
    p.add_argument("--out", default=str(config.DATA_DIR / "prefill_seeds.jsonl"))
    args = p.parse_args(argv)

    seeds = build_seeds(Path(args.episodes), recovery=args.recovery)
    write_jsonl(Path(args.out), [asdict(s) for s in seeds])
    print(f"Wrote {len(seeds)} prefill seeds -> {args.out}")


if __name__ == "__main__":
    main()
