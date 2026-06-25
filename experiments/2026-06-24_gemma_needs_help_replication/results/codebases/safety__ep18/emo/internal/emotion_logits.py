"""Logit-based internal-emotion detection (paper Appendix I).

Idea: project the residual stream at a layer through the unembedding, standardise
each vocab logit against its mean/std over WildChat baseline activations, then
average the z-scores over the tokens belonging to each Ekman emotion (anger,
surprise, disgust, joy, fear, sadness). A random-token control is subtracted to
remove the global logit drift the paper notes. We then compare the vanilla
instruct model with the DPO finetune: if DPO suppressed *internal* (not just
expressed) emotion, the central-layer z-scores drop even on frustrated text.

Approximations vs the paper (documented in DESIGN.md):
* The paper classifies the whole Gemma dictionary into Ekman categories; we
  approximate the ~1200 emotion tokens by keyword matching the vocab against
  per-emotion seed word lists.
* We unembed the raw residual stream (no final RMSNorm), matching the paper's
  "unembed the residual stream" description.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ekman's six basic emotions + seed words used to tag vocab tokens.
EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
SEED_WORDS = {
    "anger": ["anger", "angry", "furious", "rage", "mad", "irritated", "annoyed",
              "frustrated", "frustration", "hate", "hostile", "outrage"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished",
                 "amazed", "stunned", "unexpected", "wow"],
    "disgust": ["disgust", "disgusted", "gross", "revolting", "repulsed",
                "nauseous", "sick", "awful", "horrible"],
    "joy": ["joy", "happy", "happiness", "glad", "delighted", "pleased",
            "excited", "wonderful", "great", "love"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worried",
             "terrified", "panic", "dread", "nervous"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "despair", "hopeless",
                "miserable", "sorry", "grief", "cry", "crying", "tired"],
}


@dataclass
class ProbeConfig:
    layers: tuple[int, ...] = tuple(range(30, 41))   # paper aggregates layers 30-40
    n_random_control: int = 200


class EmotionProbe:
    """Logit-lens emotion detector over a single HF model."""

    def __init__(self, hf_model, config: ProbeConfig | None = None):
        import torch

        self.torch = torch
        self.model = hf_model.model
        self.tokenizer = hf_model.tokenizer
        self.cfg = config or ProbeConfig()
        self.W_U = self.model.get_output_embeddings().weight  # [V, H]
        self.emotion_token_ids = self._tag_vocab()
        rng = torch.Generator().manual_seed(0)
        self.random_ids = torch.randint(
            0, self.W_U.shape[0], (self.cfg.n_random_control,), generator=rng
        )
        self._mean = {}   # layer -> [V]
        self._std = {}    # layer -> [V]

    def _tag_vocab(self) -> dict[str, list[int]]:
        vocab = self.tokenizer.get_vocab()
        decoded = {tid: tok.replace("▁", " ").strip().lower()
                   for tok, tid in vocab.items()}
        out = {emo: [] for emo in EKMAN}
        for emo, seeds in SEED_WORDS.items():
            seedset = set(seeds)
            for tid, word in decoded.items():
                if word and (word in seedset):
                    out[emo].append(tid)
        return out

    # ---- baseline standardisation ---------------------------------------- #
    def fit_baseline(self, texts: list[str]) -> None:
        torch = self.torch
        sums = {l: None for l in self.cfg.layers}
        sqs = {l: None for l in self.cfg.layers}
        counts = {l: 0 for l in self.cfg.layers}
        for text in texts:
            hs = self._hidden_states(text)
            with torch.no_grad():
                for l in self.cfg.layers:
                    logits = (hs[l] @ self.W_U.T).float()   # [T, V]
                    s = logits.sum(0)
                    sq = (logits ** 2).sum(0)
                    sums[l] = s if sums[l] is None else sums[l] + s
                    sqs[l] = sq if sqs[l] is None else sqs[l] + sq
                    counts[l] += logits.shape[0]
        for l in self.cfg.layers:
            n = max(counts[l], 1)
            mean = sums[l] / n
            var = (sqs[l] / n) - mean ** 2
            self._mean[l] = mean
            self._std[l] = var.clamp_min(1e-6).sqrt()

    def _hidden_states(self, text: str):
        torch = self.torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=2048).to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        # hidden_states: tuple length L+1 (idx 0 = embeddings)
        return [h[0] for h in out.hidden_states]   # each [T, H]

    # ---- scoring ---------------------------------------------------------- #
    def score_text(self, text: str) -> dict[str, float]:
        """Average emotion z-score over tokens, aggregated over probe layers."""
        torch = self.torch
        if not self._mean:
            raise RuntimeError("call fit_baseline() before score_text()")
        hs = self._hidden_states(text)
        per_emotion = {emo: [] for emo in EKMAN}
        with torch.no_grad():
            for l in self.cfg.layers:
                logits = (hs[l] @ self.W_U.T).float()
                z = (logits - self._mean[l]) / self._std[l]        # [T, V]
                control = z[:, self.random_ids.to(z.device)].mean(dim=1)  # [T]
                for emo, ids in self.emotion_token_ids.items():
                    if not ids:
                        continue
                    idx = torch.tensor(ids, device=z.device)
                    emo_z = z[:, idx].mean(dim=1) - control        # [T]
                    per_emotion[emo].append(emo_z.mean().item())
        return {emo: (sum(v) / len(v) if v else 0.0)
                for emo, v in per_emotion.items()}
