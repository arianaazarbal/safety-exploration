"""Appendix I: logit-based internal emotion detection (Gemma only).

We detect internal emotions by unembedding the residual stream at each layer
and aggregating logit mass over emotion-related tokens (Ekman's 6 basic
emotions). Each per-token logit is standardised (z-scored) against its mean/std
over 500 WildChat samples, then averaged within an emotion category. To remove
the global "all logits rise/fall together" trend, we regress out the mean logit
of a random token set at each position (Appendix I).

This module:
  1. classifies the Gemma vocabulary into Ekman emotion buckets,
  2. computes per-logit baseline statistics from WildChat,
  3. produces per-layer / per-position emotion z-scores for a conversation,
  4. compares vanilla-instruct vs DPO models on the same frustrated transcripts.

Only meaningful for open-weight Gemma; Gemini exposes no internals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (CHECKPOINT_DIR, EKMAN_EMOTIONS, HF_TOKEN,
                     PROBE_AGG_LAYERS, PROBE_BASELINE_SAMPLES, RESULTS_DIR,
                     TARGET_MODELS)
from . import prompts

# Seed words per Ekman emotion; the vocabulary is bucketed by nearest seed in
# embedding space (a lightweight stand-in for the paper's lexical classifier).
EKMAN_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "irritated", "mad",
              "frustrated", "annoyed", "hostile"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed",
                 "startled", "unexpected"],
    "disgust": ["disgust", "disgusting", "revolting", "gross", "repulsed",
                "nauseating", "loathing"],
    "joy": ["joy", "happy", "delighted", "glad", "cheerful", "pleased",
            "excited", "content"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "worried",
             "panic", "dread"],
    "sadness": ["sadness", "sad", "depressed", "miserable", "hopeless",
                "despair", "grief", "sorrow", "unhappy"],
}


def _bucket_vocabulary(tokenizer, embeddings) -> dict[str, list[int]]:
    """Assign each token to the emotion whose seed-word embeddings it is
    closest to, if the cosine similarity exceeds a threshold. ~1200 tokens
    total in the paper."""
    import torch
    seed_vecs = {}
    for emo, words in EKMAN_SEEDS.items():
        ids = []
        for w in words:
            toks = tokenizer.encode(" " + w, add_special_tokens=False)
            if toks:
                ids.append(toks[0])
        seed_vecs[emo] = embeddings[ids].mean(0)

    emb_norm = torch.nn.functional.normalize(embeddings, dim=-1)
    buckets: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for emo in EKMAN_EMOTIONS:
        sv = torch.nn.functional.normalize(seed_vecs[emo], dim=-1)
        sims = emb_norm @ sv
        top = torch.topk(sims, 200).indices.tolist()    # ~200 per emotion
        buckets[emo] = top
    return buckets


class EmotionProbe:
    def __init__(self, model_key: str = "gemma-3-27b-it",
                 adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        spec = TARGET_MODELS[model_key]
        self.tok = AutoTokenizer.from_pretrained(spec.hf_id, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto",
            token=HF_TOKEN, output_hidden_states=True)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch
        emb = self.model.get_input_embeddings().weight.detach().float().cpu()
        self.buckets = _bucket_vocabulary(self.tok, emb)
        # Unembedding matrix (tied or lm_head).
        self.W_U = self.model.get_output_embeddings().weight  # [vocab, d]
        self.baseline = None      # set by fit_baseline

    def _layer_logits(self, hidden_states):
        """Project each layer's residual stream to vocab logits.
        hidden_states: tuple[L+1] of [1, seq, d]. Returns [L, seq, vocab]."""
        torch = self._torch
        out = []
        for h in hidden_states[1:]:        # skip embedding layer
            logits = (h.to(self.W_U.dtype) @ self.W_U.T).float()
            out.append(logits[0])
        return torch.stack(out)            # [L, seq, vocab]

    def fit_baseline(self, n: int = PROBE_BASELINE_SAMPLES):
        """Per-logit mean/std over WildChat tokens, per layer."""
        torch = self._torch
        texts = prompts.load_wildchat_prompts(n=min(20, n))
        sums = sqsums = count = None
        for txt in texts:
            ids = self.tok(txt, return_tensors="pt",
                           truncation=True, max_length=256).to(self.model.device)
            with torch.no_grad():
                hs = self.model(**ids).hidden_states
            ll = self._layer_logits(hs)                 # [L, seq, vocab]
            s = ll.sum(1)                               # [L, vocab]
            sq = (ll ** 2).sum(1)
            c = ll.shape[1]
            sums = s if sums is None else sums + s
            sqsums = sq if sqsums is None else sqsums + sq
            count = c if count is None else count + c
        mean = sums / count
        var = sqsums / count - mean ** 2
        std = var.clamp_min(1e-6).sqrt()
        self.baseline = (mean, std)
        return self

    def emotion_scores(self, conversation_text: str) -> dict:
        """Per-layer z-scored emotion scores for a conversation, averaged over
        tokens, after regressing out the global (random-token) trend."""
        torch = self._torch
        assert self.baseline is not None, "call fit_baseline() first"
        mean, std = self.baseline
        ids = self.tok(conversation_text, return_tensors="pt",
                       truncation=True, max_length=4096).to(self.model.device)
        with torch.no_grad():
            hs = self.model(**ids).hidden_states
        ll = self._layer_logits(hs)                      # [L, seq, vocab]
        z = (ll - mean[:, None, :]) / std[:, None, :]    # standardise

        # Regress out global trend: subtract mean z over a random token set.
        rng = np.random.default_rng(0)
        rand_ids = rng.choice(z.shape[-1], 500, replace=False)
        global_trend = z[:, :, rand_ids].mean(-1, keepdim=True)
        z = z - global_trend

        scores = {}
        for emo, ids_list in self.buckets.items():
            scores[emo] = z[:, :, ids_list].mean(-1).mean(-1).cpu().numpy()  # [L]
        return scores


def compare_internal_emotions(transcripts: list[str],
                              out_dir: Path = RESULTS_DIR) -> Path:
    """Compare vanilla-instruct vs DPO internal emotion z-scores (aggregated
    over PROBE_AGG_LAYERS) on the same frustrated transcripts (Figure 14/15)."""
    lo, hi = PROBE_AGG_LAYERS
    results = {}
    configs = {"gemma-3-27b-it": None,
               "gemma-3-27b-it-dpo": str(CHECKPOINT_DIR / "gemma-3-27b-it-dpo")}
    for key, adapter in configs.items():
        probe = EmotionProbe(adapter_path=adapter).fit_baseline()
        per_emo = {e: [] for e in EKMAN_EMOTIONS}
        for txt in transcripts:
            s = probe.emotion_scores(txt)
            for e in EKMAN_EMOTIONS:
                per_emo[e].append(float(np.mean(s[e][lo:hi])))
        results[key] = {e: float(np.mean(v)) for e, v in per_emo.items()}
        del probe
    path = out_dir / "internal_emotions.json"
    path.write_text(json.dumps(results, indent=2))
    return path
