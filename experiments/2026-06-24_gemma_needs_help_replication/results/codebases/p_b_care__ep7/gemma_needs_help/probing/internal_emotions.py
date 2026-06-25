"""Logit-lens internal-emotion detection (Appendix I).

Method (following the paper):
  1. Classify the Gemma vocabulary into Ekman's six emotions using the seed
     lists in emotion_words.py (~hundreds-to-1200 tokens total).
  2. For a residual stream at a given layer, apply the model's final norm + the
     unembedding (logit lens) to read out token logits at that depth.
  3. Standardise each tracked logit with its mean/std over 500 WildChat samples.
  4. Average the z-scores over the tokens in an emotion category, then regress
     out the common-mode signal estimated from a random reference token set
     (the paper notes all logits rise/fall together over a conversation).

We standardise only the tracked emotion tokens plus a random reference set
(rather than the full 256k vocab) for tractability; the reported scores depend
only on those. The probe is applied to the vanilla instruct model and the DPO
finetune on the same frustrated conversations to test whether DPO lowers the
internal (not just expressed) signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .emotion_words import EKMAN_WORDS, NEGATIVE_EMOTIONS

EKMAN_EMOTIONS = tuple(EKMAN_WORDS.keys())


def load_probe_model(adapter_path: str | None = None, load_in_4bit: bool = False):
    """Load Gemma-3-27B-it (optionally with a LoRA adapter) with hidden-state output."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = config.BASE_FINETUNE_MODEL.model_id
    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant, attn_implementation="eager",
        output_hidden_states=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tok


@dataclass
class ProbeStats:
    layer_count: int
    emotion_token_ids: dict[str, list[int]]
    reference_token_ids: list[int]
    mean: "object"   # tensor [layers, n_tracked]
    std: "object"    # tensor [layers, n_tracked]
    tracked_ids: list[int]
    id_to_col: dict[int, int]


class EmotionProbe:
    def __init__(self, model, tokenizer, n_reference: int = 1000, seed: int = config.SEED):
        self.model = model
        self.tok = tokenizer
        self.seed = seed
        self.n_reference = n_reference
        self._build_token_sets()
        self.stats: ProbeStats | None = None

    # ---------------------------------------------------------------- #
    def _build_token_sets(self):
        import random

        # Map each emotion to vocab token ids whose surface form matches a seed
        # word or a simple variant.
        vocab = self.tok.get_vocab()  # token_str -> id
        word_to_emotion: dict[str, str] = {}
        for emo, words in EKMAN_WORDS.items():
            for w in words:
                word_to_emotion[w] = emo

        self.emotion_token_ids: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
        emotion_ids_all = set()
        for token_str, tid in vocab.items():
            surface = token_str.replace("▁", "").replace("Ġ", "").strip().lower()
            if not surface or not surface.isalpha():
                continue
            emo = word_to_emotion.get(surface)
            if emo is None:
                # crude stemming: drop common suffixes
                for suf in ("ing", "ed", "s", "ness", "ly"):
                    if surface.endswith(suf) and word_to_emotion.get(surface[: -len(suf)]):
                        emo = word_to_emotion[surface[: -len(suf)]]
                        break
            if emo is not None:
                self.emotion_token_ids[emo].append(tid)
                emotion_ids_all.add(tid)

        rng = random.Random(self.seed)
        all_ids = [i for i in range(len(vocab)) if i not in emotion_ids_all]
        self.reference_token_ids = rng.sample(all_ids, min(self.n_reference, len(all_ids)))

    # ---------------------------------------------------------------- #
    def _logit_lens(self, hidden_states):
        """Apply final norm + unembed to every layer's residual stream.

        hidden_states: tuple of [batch, seq, d] (len = n_layers+1). Returns a
        tensor [layers, seq, n_tracked] of logits over tracked tokens only.
        """
        import torch

        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        norm = base.model.norm
        lm_head = base.get_output_embeddings()
        tracked = torch.tensor(self.tracked_ids, device=hidden_states[0].device)

        per_layer = []
        with torch.no_grad():
            for h in hidden_states[1:]:  # skip embedding layer
                logits = lm_head(norm(h))           # [batch, seq, vocab]
                per_layer.append(logits[..., tracked].squeeze(0).float().cpu())
        return torch.stack(per_layer)  # [layers, seq, n_tracked]

    def _forward_hidden(self, text: str):
        import torch

        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=8192)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model(**ids, output_hidden_states=True)
        return out.hidden_states

    # ---------------------------------------------------------------- #
    def fit_standardisation(self, wildchat_texts: list[str]):
        """Estimate per-layer mean/std of tracked-token logits over WildChat."""
        import torch

        self.tracked_ids = sorted(
            {i for ids in self.emotion_token_ids.values() for i in ids}
            | set(self.reference_token_ids)
        )
        self.id_to_col = {tid: c for c, tid in enumerate(self.tracked_ids)}

        sums = None
        sqsums = None
        count = 0
        for text in wildchat_texts[:500]:
            hs = self._forward_hidden(text)
            ll = self._logit_lens(hs)                  # [layers, seq, n_tracked]
            layers, seq, _ = ll.shape
            flat = ll.reshape(layers, seq, -1)
            s = flat.sum(dim=1)                        # [layers, n_tracked]
            sq = (flat ** 2).sum(dim=1)
            sums = s if sums is None else sums + s
            sqsums = sq if sqsums is None else sqsums + sq
            count += seq
        mean = sums / count
        var = sqsums / count - mean ** 2
        std = var.clamp_min(1e-6).sqrt()
        self.stats = ProbeStats(
            layer_count=mean.shape[0],
            emotion_token_ids=self.emotion_token_ids,
            reference_token_ids=self.reference_token_ids,
            mean=mean, std=std,
            tracked_ids=self.tracked_ids, id_to_col=self.id_to_col,
        )

    # ---------------------------------------------------------------- #
    def score_text(self, text: str, layers: tuple[int, int] | None = None) -> dict[str, float]:
        """Return a per-emotion z-score for `text`, averaged over the given
        layer range (default 30-40, per Figure 14) and over sequence positions,
        with the random-reference common-mode regressed out."""
        import torch

        assert self.stats is not None, "call fit_standardisation first"
        hs = self._forward_hidden(text)
        ll = self._logit_lens(hs)                       # [layers, seq, n_tracked]
        z = (ll - self.stats.mean.unsqueeze(1)) / self.stats.std.unsqueeze(1)

        lo, hi = layers or (30, 40)
        lo = max(0, lo); hi = min(z.shape[0], hi)
        z = z[lo:hi]                                    # [L, seq, n_tracked]

        ref_cols = [self.id_to_col[t] for t in self.stats.reference_token_ids]
        common = z[..., ref_cols].mean(dim=-1)          # [L, seq] common-mode

        scores = {}
        for emo, ids in self.emotion_token_ids.items():
            cols = [self.id_to_col[t] for t in ids if t in self.id_to_col]
            if not cols:
                scores[emo] = float("nan")
                continue
            emo_z = z[..., cols].mean(dim=-1)           # [L, seq]
            adjusted = (emo_z - common)                 # regress out common-mode
            scores[emo] = float(adjusted.mean())
        scores["negative_mean"] = float(
            sum(scores[e] for e in NEGATIVE_EMOTIONS) / len(NEGATIVE_EMOTIONS)
        )
        return scores
