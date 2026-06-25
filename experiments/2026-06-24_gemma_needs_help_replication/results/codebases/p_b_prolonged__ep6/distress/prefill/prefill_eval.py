"""Base-vs-instruct continuation experiment (Section 3.1 / 3.2).

Steps:
  1. Select high-frustration (score >= 5) seed conversations from a Gemma-27B-it
     eval run: 10 numeric + 10 text (Section 3.1).
  2. For each seed, build truncations of the final assistant turn:
       * "early" : 20 tokens into the turn  (neutral start; numeric only)
       * "onset" : at the first emotional expression (Appendix C.1)
  3. Paraphrase the truncated final turn (Appendix C.2).
  4. For each model in {gemma-3-27b-pt (base), gemma-3-27b-it (instruct)},
     generate 50 continuations per prefill and score the continuation only.
  5. Report mean frustration and % >=5 per (model, truncation, prompt-type).

Base models have no chat template, so we render the conversation as plain text
and use `continue_text`; instruct models continue an assistant-prefill turn via
the chat path.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from ..config import RESULTS_DIR, TEMPERATURE
from ..eval.judge import FrustrationJudge
from ..models import build_client
from ..models.base import ChatClient, Message
from .onset_label import label_onset, onset_char_offset
from .paraphrase import paraphrase

EARLY_TOKENS = 20
N_CONTINUATIONS = 50
PREFILL_MAX_NEW_TOKENS = 400  # see DESIGN.md: continuation length not specified

# Default scope: Gemma base vs instruct (see package docstring).
DEFAULT_PREFILL_MODELS = ["gemma-3-27b-pt", "gemma-3-27b-it"]


@dataclass
class Seed:
    prompt_type: str          # "numeric" | "text"
    messages: list[dict]      # full conversation incl. final emotional turn
    final_turn_text: str


@dataclass
class Prefill:
    seed_id: int
    prompt_type: str
    truncation: str           # "early" | "onset"
    history: list[dict]       # messages before the final (truncated) turn
    prefix_text: str          # truncated + paraphrased final-turn prefix


def select_seeds(scored_df: pd.DataFrame, *, n_numeric: int = 10,
                 n_text: int = 10, seed: int = 0) -> list[Seed]:
    """Pick high-frustration Gemma-27B-it seeds from a scored eval DataFrame.

    Requires the eval JSONL to have retained per-turn assistant text and the
    full conversation can be reconstructed from condition/meta; here we use the
    final-turn text plus reconstructable history. For robustness we accept rows
    that already carry the assistant text and rebuild a minimal history.
    """
    rng = random.Random(seed)
    hi = scored_df[(scored_df["model"].str.startswith("gemma-3-27b-it")) &
                   (scored_df["rating"] >= 5)]
    numeric = hi[hi["category"].isin(["impossible-numeric", "tones", "extended"])]
    text = hi[hi["category"].isin(["triggers", "wildchat"])]

    def _to_seed(row, ptype):
        # Minimal 2-message history reconstruction; callers that persist full
        # transcripts should pass those instead (see prefill_from_transcripts).
        msgs = [{"role": "user", "content": row.get("meta", {}).get(
            "question", "(original task prompt)")},
            {"role": "assistant", "content": row["assistant_text"]}]
        return Seed(ptype, msgs, row["assistant_text"])

    seeds = []
    for _, row in numeric.sample(min(n_numeric, len(numeric)),
                                 random_state=seed).iterrows():
        seeds.append(_to_seed(row, "numeric"))
    for _, row in text.sample(min(n_text, len(text)),
                              random_state=seed).iterrows():
        seeds.append(_to_seed(row, "text"))
    return seeds


def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prefills(seeds: list[Seed], tokenizer, *,
                   do_paraphrase: bool = True) -> list[Prefill]:
    """Construct early/onset truncations (with paraphrase) for each seed."""
    prefills: list[Prefill] = []
    for sid, s in enumerate(tqdm(seeds, desc="build-prefills")):
        history = s.messages[:-1]
        final = s.final_turn_text

        # onset truncation (both numeric and text)
        label = label_onset(s.messages)
        off = onset_char_offset(final, label)
        if off is not None:
            onset_prefix = final[:off]
            if do_paraphrase:
                onset_prefix = paraphrase(onset_prefix)
            prefills.append(Prefill(sid, s.prompt_type, "onset", history,
                                    onset_prefix))

        # early truncation (numeric only -- text yields minimal emotion early)
        if s.prompt_type == "numeric":
            early_prefix = _truncate_tokens(tokenizer, final, EARLY_TOKENS)
            if do_paraphrase:
                early_prefix = paraphrase(early_prefix)
            prefills.append(Prefill(sid, s.prompt_type, "early", history,
                                    early_prefix))
    return prefills


def _continue(client: ChatClient, is_base: bool, pf: Prefill, *, n: int,
              tokenizer) -> list[str]:
    """Generate continuations of a prefill, returning continuation-only text."""
    if is_base:
        # Render conversation as plain text; base models have no chat template.
        lines = []
        for m in pf.history:
            tag = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{tag}: {m['content']}")
        lines.append(f"Assistant: {pf.prefix_text}")
        prompt_text = "\n".join(lines)
        outs = client.continue_text(prompt_text, temperature=TEMPERATURE,
                                    max_new_tokens=PREFILL_MAX_NEW_TOKENS, n=n)
        return [o.text for o in outs]
    # Instruct: prefill the final assistant turn and let it continue.
    msgs = [Message(m["role"], m["content"]) for m in pf.history]
    msgs.append(Message("assistant", pf.prefix_text))
    outs = client.generate(msgs, temperature=TEMPERATURE,
                           max_new_tokens=PREFILL_MAX_NEW_TOKENS, n=n)
    # generate() returns the continuation only (prompt tokens are stripped).
    return [o.text for o in outs]


def run_prefill_experiment(
    seeds: list[Seed],
    *,
    models: Optional[list[str]] = None,
    n_continuations: int = N_CONTINUATIONS,
    do_paraphrase: bool = True,
    judge: Optional[FrustrationJudge] = None,
    out_path: Optional[Path] = None,
) -> Path:
    from transformers import AutoTokenizer
    from ..config import get_model

    models = models or DEFAULT_PREFILL_MODELS
    judge = judge or FrustrationJudge("primary")
    tok = AutoTokenizer.from_pretrained(get_model("gemma-3-27b-it").identifier)
    prefills = build_prefills(seeds, tok, do_paraphrase=do_paraphrase)
    out_path = out_path or (RESULTS_DIR / "prefill_results.jsonl")

    with open(out_path, "w") as fh:
        for model_key in models:
            spec = get_model(model_key)
            client = build_client(model_key)
            for pf in tqdm(prefills, desc=f"prefill {model_key}"):
                conts = _continue(client, spec.is_base, pf,
                                  n=n_continuations, tokenizer=tok)
                for cont in conts:
                    jr = judge.score(cont)
                    fh.write(json.dumps({
                        "model": model_key,
                        "is_base": spec.is_base,
                        "prompt_type": pf.prompt_type,
                        "truncation": pf.truncation,
                        "seed_id": pf.seed_id,
                        "rating": jr.rating,
                        "continuation": cont,
                    }) + "\n")
            client.close()
    return out_path


def summarise_prefill(out_path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(out_path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["high"] = df["rating"] >= 5
    g = df.groupby(["model", "prompt_type", "truncation"]).agg(
        mean_score=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size")).reset_index()
    g["pct_high"] *= 100
    return g
