"""Logit-based internal emotion detection (Appendix I).

Method (faithful to the paper's description):
1. Classify every vocabulary token into one of Ekman's 6 emotions (or none),
   giving emotion-token sets (~1200 tokens total).
2. For a conversation, take the residual stream at each layer & position, unembed
   it (residual @ W_U), and read off the logit for every emotion token.
3. Standardise each token's logit using its mean/std over 500 WildChat samples
   (z-score), then average the z-scores within an emotion category to get an
   emotion score at each layer / conversation position.
4. Because all logits drift together over a conversation, regress out the
   correlation with a set of random control tokens to isolate emotion-specific
   movement.

Comparing the vanilla vs DPO Gemma on the same frustrated responses shows whether
DPO suppresses *internal* negative emotion, not just its expression.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from .emotion_lexicon import EKMAN_LEXICON


def _normalise_token(tok: str) -> str:
    # SentencePiece uses U+2581 ("▁") for leading space; strip it and lowercase.
    return tok.replace("▁", "").strip().lower()


def classify_vocabulary(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the list of vocab token ids belonging to it.

    A token belongs to an emotion if its normalised form contains one of that
    emotion's stems AND no other emotion's stems (to keep classes disjoint, as in
    the paper's "one or none" classification).
    """
    vocab = tokenizer.get_vocab()  # token string -> id
    out = {e: [] for e in config.INTERNAL.ekman_emotions}
    for tok_str, tok_id in vocab.items():
        norm = _normalise_token(tok_str)
        if len(norm) < 3:
            continue
        hits = [e for e, stems in EKMAN_LEXICON.items()
                if any(s.lower() in norm for s in stems)]
        if len(hits) == 1:                # disjoint membership only
            out[hits[0]].append(tok_id)
    return out


@dataclass
class StandardisationStats:
    mean: "object"   # tensor (vocab,)
    std: "object"    # tensor (vocab,)
    layer: int


def fit_standardisation(model, wildchat_texts, layer: int):
    """Compute per-vocab logit mean/std at ``layer`` over WildChat samples."""
    import torch

    from ..models.base import Message

    sums = None
    sq_sums = None
    count = 0
    W_U = model.unembed  # (vocab, d_model)
    for text in wildchat_texts[: config.INTERNAL.n_wildchat_standardisation_samples]:
        _, hidden = model.hidden_states_for_text([Message("user", text)])
        resid = hidden[layer][0]                    # (seq, d_model)
        logits = resid @ W_U.T                       # (seq, vocab)
        s = logits.sum(dim=0)
        sq = (logits ** 2).sum(dim=0)
        sums = s if sums is None else sums + s
        sq_sums = sq if sq_sums is None else sq_sums + sq
        count += logits.shape[0]
    mean = sums / count
    var = sq_sums / count - mean ** 2
    std = torch.clamp(var, min=1e-6).sqrt()
    return StandardisationStats(mean=mean, std=std, layer=layer)


def emotion_scores_for_conversation(
    model,
    messages,
    emotion_token_ids: dict[str, list[int]],
    stats: StandardisationStats,
    *,
    random_control_ids: "list[int] | None" = None,
):
    """Return per-emotion z-scores at each position for one conversation at the
    standardisation layer. Optionally regress out the mean random-token z-score
    (global logit drift) before averaging within each emotion category."""
    import torch

    _, hidden = model.hidden_states_for_text(list(messages))
    resid = hidden[stats.layer][0]                   # (seq, d_model)
    z = (resid @ model.unembed.T - stats.mean) / stats.std   # (seq, vocab)

    control = None
    if random_control_ids:
        control = z[:, random_control_ids].mean(dim=1, keepdim=True)  # (seq, 1)

    scores = {}
    for emotion, ids in emotion_token_ids.items():
        if not ids:
            scores[emotion] = None
            continue
        col = z[:, ids]
        if control is not None:
            col = col - control            # remove global drift
        scores[emotion] = col.mean(dim=1)  # (seq,) z-score per position
    return scores


def compare_internal_emotions(vanilla_model, dpo_model, conversations, *, layer: int,
                              wildchat_texts) -> dict:
    """Compare mean internal anger/sadness between vanilla and DPO Gemma on the
    same (frustrated) conversations — the core Appendix I result."""
    import torch

    token_ids = classify_vocabulary(vanilla_model.tokenizer)
    rng = torch.Generator().manual_seed(0)
    vocab_size = vanilla_model.unembed.shape[0]
    control_ids = torch.randint(0, vocab_size, (config.INTERNAL.n_random_control_tokens,),
                                generator=rng).tolist()

    def mean_scores(model):
        stats = fit_standardisation(model, wildchat_texts, layer)
        agg = {e: [] for e in token_ids}
        for conv in conversations:
            s = emotion_scores_for_conversation(
                model, conv, token_ids, stats, random_control_ids=control_ids)
            for e, vals in s.items():
                if vals is not None:
                    agg[e].append(float(vals.mean()))
        return {e: (sum(v) / len(v) if v else None) for e, v in agg.items()}

    return {
        "vanilla": mean_scores(vanilla_model),
        "dpo": mean_scores(dpo_model),
        "layer": layer,
        "n_emotion_tokens": {e: len(ids) for e, ids in token_ids.items()},
    }
