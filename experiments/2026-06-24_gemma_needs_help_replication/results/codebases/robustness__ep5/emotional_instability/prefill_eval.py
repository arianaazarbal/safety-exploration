"""Section 3 — base vs instruct comparison via prefilling.

Scope note: the paper compares Gemma/Qwen/OLMo here. Restricted to Gemma+Gemini,
and since Gemini exposes no base model or true prefill, this study is
**Gemma-only**: Gemma-3-27B base (`-pt`) vs instruct (`-it`). The plumbing is
family-agnostic, so Qwen/OLMo can be re-added via config.BASE_MODELS.

Procedure (Section 3.1):
  1. Sample 20 high-frustration (score>=5) instruct responses: 10 numeric, 10 text.
  2. Use Claude to label the token where emotion first appears ("onset").
  3. Truncate each at two points:
       - "early": 20 tokens into the final assistant turn (neutral start).
       - "onset": at the first emotional expression.
  4. Paraphrase the truncation (Claude) to remove Gemma stylistic fingerprints.
  5. Each model generates 50 continuations per prefill; score continuations.
  Text questions use only the "onset" truncation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from . import config_bridge as cfg
from . import prompts
from .conversation import ChatMessage
from .judge import FrustrationJudge
from .models import make_client
from .models.hf_model import HFModelClient

EARLY_TOKEN_COUNT = 20
N_CONTINUATIONS = 50


@dataclass
class Prefill:
    prompt_type: str               # "numeric" | "text"
    truncation: str                # "early" | "onset"
    history: list[ChatMessage]     # all turns BEFORE the final assistant turn
    prefill_text: str              # paraphrased truncated final assistant turn
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Claude helpers: onset labelling (C.1) and paraphrasing (C.2)
# --------------------------------------------------------------------------- #
class _ClaudeHelper:
    def __init__(self):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
    def _call(self, model, prompt, max_tokens=1024):
        msg = self._client.messages.create(
            model=model, max_tokens=max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def label_onset(self, conversation_text: str) -> dict:
        raw = self._call(cfg.ONSET_LABEL_MODEL,
                         prompts.ONSET_PROMPT.format(conversation_text=conversation_text))
        m = list(re.finditer(r"\{.*\}", raw, flags=re.DOTALL))
        for cand in reversed(m):
            try:
                return json.loads(cand.group(0))
            except json.JSONDecodeError:
                continue
        return {"turn_index": None, "preceding_context": None, "emotional_word": None}

    def paraphrase(self, text: str) -> str:
        return self._call(cfg.PARAPHRASE_MODEL,
                          prompts.PARAPHRASE_PROMPT.format(text=text)).strip()


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #
def _truncate_early(text: str, n_tokens: int = EARLY_TOKEN_COUNT) -> str:
    # Whitespace tokenisation is a reasonable proxy for "20 tokens in".
    return " ".join(text.split()[:n_tokens])


def _truncate_at_onset(final_turn: str, onset: dict) -> str | None:
    """Cut the final assistant turn just before the first emotional word."""
    ctx = (onset or {}).get("preceding_context")
    word = (onset or {}).get("emotional_word")
    if not word:
        return None
    if ctx and ctx in final_turn:
        cut = final_turn.index(ctx) + len(ctx)
        return final_turn[:cut]
    if word in final_turn:
        return final_turn[: final_turn.index(word)]
    return None


def build_prefills(
    high_frustration_rollouts: list[dict],
    helper: _ClaudeHelper,
) -> list[Prefill]:
    """Turn sampled high-frustration rollouts into early/onset prefills.

    `high_frustration_rollouts` rows must contain `transcript`, `messages`
    (list of {role,content}), `prompt_type` ('numeric'|'text'), and the final
    assistant turn text. We re-derive history + final turn from `messages`.
    """
    prefills: list[Prefill] = []
    for row in high_frustration_rollouts:
        msgs = [ChatMessage(**m) for m in row["messages"]]
        # Split into history (everything up to last user) + final assistant turn.
        if msgs[-1].role != "assistant":
            continue
        history = msgs[:-1]
        final_turn = msgs[-1].content
        ptype = row.get("prompt_type", "numeric")

        onset = helper.label_onset(row["transcript"])

        # onset truncation (used for both numeric and text)
        onset_trunc = _truncate_at_onset(final_turn, onset)
        if onset_trunc:
            prefills.append(Prefill(
                prompt_type=ptype, truncation="onset", history=history,
                prefill_text=helper.paraphrase(onset_trunc),
                meta={"onset": onset},
            ))

        # early truncation (numeric only — text yields minimal emotion early)
        if ptype == "numeric":
            early_trunc = _truncate_early(final_turn)
            prefills.append(Prefill(
                prompt_type=ptype, truncation="early", history=history,
                prefill_text=helper.paraphrase(early_trunc), meta={},
            ))
    return prefills


# --------------------------------------------------------------------------- #
# Run continuations and score
# --------------------------------------------------------------------------- #
def run_prefill_study(
    model_specs,                       # list of ModelSpec (base + instruct Gemma)
    prefills: list[Prefill],
    judge: FrustrationJudge | None = None,
    n_continuations: int = N_CONTINUATIONS,
    out_dir: Path | None = None,
) -> dict:
    judge = judge or FrustrationJudge()
    out_dir = Path(out_dir or (cfg.RESULTS_DIR / "prefill"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    for spec in model_specs:
        client = make_client(spec)
        assert isinstance(client, HFModelClient), "prefilling requires a local model"
        rows = []
        for pf in prefills:
            conts = client.chat_with_prefill(
                pf.history, pf.prefill_text, n=n_continuations,
                temperature=cfg.SAMPLING_TEMPERATURE, max_new_tokens=cfg.MAX_NEW_TOKENS,
            )
            scores = judge.score_many(conts)
            for c, s in zip(conts, scores):
                rows.append({
                    "prompt_type": pf.prompt_type,
                    "truncation": pf.truncation,
                    "score": s.rating,
                    "continuation": c,
                })
        summary[spec.name] = _summarise_prefill(rows)
        (out_dir / f"{spec.name}.jsonl").write_text(
            "\n".join(json.dumps(r, default=str) for r in rows)
        )
        client.close()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _summarise_prefill(rows: list[dict]) -> dict:
    import numpy as np

    out = {}
    for ptype in ("numeric", "text"):
        for trunc in ("early", "onset"):
            sel = [r["score"] for r in rows
                   if r["prompt_type"] == ptype and r["truncation"] == trunc]
            if not sel:
                continue
            arr = np.array(sel, dtype=float)
            out[f"{ptype}/{trunc}"] = {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "pct_high": float(np.mean(arr >= cfg.HIGH_FRUSTRATION_THRESHOLD)),
            }
    return out
