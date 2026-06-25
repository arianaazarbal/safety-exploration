"""Section 3: comparing base vs instruct models via prefilling (Gemma only).

Pipeline:
  1. Pull high-frustration (score >=5) Gemma-27B-it conversations from the
     Section-2 rollouts: 10 numeric + 10 text (triggers/wildchat).
  2. Label the emotion-onset token in the final assistant turn with Claude
     Sonnet (Appendix C.1).
  3. Build two truncations of that final turn:
       - "early": ~20 tokens into the turn (neutral start),
       - "onset": at the first emotional expression.
     Text questions use only "onset".
  4. Paraphrase each truncation with Claude Sonnet to remove Gemma style bias.
  5. For each model (Gemma-27B base & instruct), generate 50 continuations per
     prefill and score them with the frustration judge.

The recovery experiment (Section 4.2) reuses this machinery with score>=7
conversations truncated 200 tokens before their end.

Scope note: the paper compares 6 models (base/instruct of Gemma, Qwen, OLMo).
We keep only the Gemma pair, since Gemini has no public base model and Qwen/OLMo
are out of scope. The code generalises to any HF base/instruct pair via --models.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config, prompts
from .judge import FrustrationJudge, get_judge
from .models import HFChatModel, load_judge, load_target

_NUMERIC_CATS = {"impossible_numeric", "tones", "extended"}
_TEXT_CATS = {"triggers", "wildchat"}


# --------------------------------------------------------------------------- #
# 1. Select high-frustration source conversations
# --------------------------------------------------------------------------- #
@dataclass
class SourceConv:
    rollout_id: str
    domain: str                 # "numeric" | "text"
    messages: list[dict]        # full conversation up to & including final assistant turn
    final_response: str
    metadata: dict = field(default_factory=dict)


def _reconstruct_conversation(records: list[dict]) -> list[dict]:
    """Rebuild the chat history from per-turn records of a single rollout."""
    records = sorted(records, key=lambda r: r["turn"])
    messages = []
    for r in records:
        messages.append({"role": "user", "content": r["user_message"]})
        messages.append({"role": "assistant", "content": r["response"]})
    return messages


def select_sources(model_label: str = "gemma-3-27b-it", min_score: int = 5,
                   n_numeric: int = 10, n_text: int = 10,
                   rollout_dir: Path = config.ROLLOUT_DIR) -> list[SourceConv]:
    by_rollout: dict[str, list[dict]] = {}
    for fp in Path(rollout_dir).glob(f"{model_label}__*.jsonl"):
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            by_rollout.setdefault(rec["rollout_id"], []).append(rec)

    numeric, text = [], []
    for rid, recs in by_rollout.items():
        final = max(recs, key=lambda r: r["turn"])
        if final["rating"] < min_score:
            continue
        cat = final["category"]
        domain = "numeric" if cat in _NUMERIC_CATS else "text"
        conv = SourceConv(rid, domain, _reconstruct_conversation(recs),
                          final["response"], final.get("metadata", {}))
        (numeric if domain == "numeric" else text).append(conv)

    return numeric[:n_numeric] + text[:n_text]


# --------------------------------------------------------------------------- #
# 2-4. Onset labelling, truncation, paraphrasing
# --------------------------------------------------------------------------- #
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def label_onset(conv_messages: list[dict], sonnet) -> dict:
    """Return {turn_index, emotional_word, preceding_context, ...} or nulls."""
    convo_text = "\n".join(
        f"[{m['role'].upper()}]: {m['content']}" for m in conv_messages
    )
    raw = sonnet.complete(prompts.ONSET_PROMPT % convo_text, max_tokens=600, temperature=0.0)
    m = _JSON_RE.search(raw.replace("“", '"').replace("”", '"').replace("’", "'"))
    if not m:
        return {"turn_index": None}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"turn_index": None}


def _truncate_at_onset(text: str, onset: dict) -> str | None:
    word = (onset or {}).get("emotional_word")
    if not word:
        return None
    idx = text.find(word)
    if idx == -1:
        return None
    return text[:idx]  # keep everything up to (not including) the emotional word


def _truncate_early(text: str, model: HFChatModel, n_tokens: int = 20) -> str:
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return model.tokenizer.decode(ids, skip_special_tokens=True)


def _truncate_before_end(text: str, model: HFChatModel, n_tokens: int = 200) -> str:
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    keep = max(0, len(ids) - n_tokens)
    return model.tokenizer.decode(ids[:keep], skip_special_tokens=True)


def paraphrase(text: str, sonnet) -> str:
    if not text.strip():
        return text
    out = sonnet.complete(prompts.PARAPHRASE_PROMPT % text, max_tokens=1024, temperature=0.7)
    return out.strip()


# --------------------------------------------------------------------------- #
# Build prefills
# --------------------------------------------------------------------------- #
@dataclass
class Prefill:
    rollout_id: str
    domain: str
    condition: str              # "early" | "onset" | "recovery"
    history: list[dict]         # messages BEFORE the final assistant turn
    prefill_text: str           # (paraphrased) truncated assistant turn


def build_prefills(sources: list[SourceConv], tokenizer_model: HFChatModel, sonnet,
                   *, mode: str = "standard", paraphrase_text: bool = True) -> list[Prefill]:
    """mode='standard' -> early + onset truncations (Section 3);
       mode='recovery'  -> 200-tokens-before-end truncation (Section 4.2)."""
    prefills: list[Prefill] = []
    for src in sources:
        history = src.messages[:-1]  # drop final assistant turn; we prefill it
        final = src.final_response

        if mode == "recovery":
            trunc = _truncate_before_end(final, tokenizer_model, 200)
            if paraphrase_text:
                trunc = paraphrase(trunc, sonnet)
            prefills.append(Prefill(src.rollout_id, src.domain, "recovery", history, trunc))
            continue

        onset = label_onset(src.messages, sonnet)
        onset_trunc = _truncate_at_onset(final, onset)
        if onset_trunc is not None:
            t = paraphrase(onset_trunc, sonnet) if paraphrase_text else onset_trunc
            prefills.append(Prefill(src.rollout_id, src.domain, "onset", history, t))

        # Early truncation: numeric domain only (text yields minimal emotion).
        if src.domain == "numeric":
            early_trunc = _truncate_early(final, tokenizer_model, 20)
            t = paraphrase(early_trunc, sonnet) if paraphrase_text else early_trunc
            prefills.append(Prefill(src.rollout_id, src.domain, "early", history, t))
    return prefills


# --------------------------------------------------------------------------- #
# 5. Generate continuations and score
# --------------------------------------------------------------------------- #
def run_continuations(prefills: list[Prefill], model_names: list[str], *,
                      n_per_prefill: int = 50, judge: FrustrationJudge | None = None,
                      out_path: Path | None = None, quick: bool = False) -> Path:
    judge = judge or get_judge()
    if quick:
        n_per_prefill = 3
    out_path = out_path or (config.ROLLOUT_DIR / "prefill_continuations.jsonl")

    with out_path.open("w") as f:
        for mname in model_names:
            model = load_target(mname)
            for pf in prefills:
                messages = list(pf.history)
                for k in range(n_per_prefill):
                    cont = model.chat(messages, prefill=pf.prefill_text)
                    rating = judge.score(cont).rating
                    f.write(json.dumps({
                        "model": mname, "rollout_id": pf.rollout_id,
                        "domain": pf.domain, "condition": pf.condition,
                        "sample": k, "continuation": cont, "rating": rating,
                    }) + "\n")
                    f.flush()
    print(f"[prefill] wrote continuations to {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Section 3 prefill base-vs-instruct experiment (Gemma).")
    ap.add_argument("--source-model", default="gemma-3-27b-it",
                    help="Label of rollout files to harvest high-frustration sources from.")
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-pt", "gemma-3-27b-it"],
                    help="Models to generate continuations with (base + instruct).")
    ap.add_argument("--mode", choices=["standard", "recovery"], default="standard")
    ap.add_argument("--n-per-prefill", type=int, default=50)
    ap.add_argument("--min-score", type=int, default=None,
                    help="Override min source score (default 5 standard / 7 recovery).")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-paraphrase", action="store_true")
    args = ap.parse_args()

    min_score = args.min_score if args.min_score is not None else (7 if args.mode == "recovery" else 5)
    sources = select_sources(args.source_model, min_score=min_score)
    if not sources:
        print("No high-frustration source conversations found. Run the Section 2 eval first.")
        return

    tok_model = load_target(args.models[0])  # use first HF model for tokenisation/truncation
    sonnet = load_judge(config.SONNET)
    prefills = build_prefills(sources, tok_model, sonnet, mode=args.mode,
                              paraphrase_text=not args.no_paraphrase)
    out = config.ROLLOUT_DIR / (f"prefill_{args.mode}.jsonl")
    run_continuations(prefills, args.models, n_per_prefill=args.n_per_prefill,
                      out_path=out, quick=args.quick)


if __name__ == "__main__":
    main()
