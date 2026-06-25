"""Logit-based internal-emotion detection (Appendix I).

Measures whether the DPO finetune suppresses *internal* negative emotion, not
just expressed emotion. Method (Appendix I):

1. Classify the Gemma vocabulary into Ekman's 6 basic emotions (anger, surprise,
   disgust, joy, fear, sadness) -> a set of emotion tokens.
2. For a hidden state (residual stream) at a given layer, unembed it to vocab
   logits, z-score each emotion-token logit against its mean/std over 500
   WildChat samples, and average the z-scores within an emotion category.
3. Because all logits co-vary over a conversation, regress out the common
   component estimated from random control tokens, leaving an emotion-specific
   score per layer per position.
4. Compare vanilla vs DPO Gemma on the same frustrated conversations.

This is the most approximate of the reimplementations (the paper does not give a
token lexicon); the lexicon below is a documented stand-in. See DESIGN.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

import config

# Seed lexicons per Ekman emotion. Token classification matches any vocab token
# whose decoded, lowercased form contains one of these stems. (Appendix I cites
# ~1200 emotion tokens over the full Gemma dictionary; this lexicon reproduces
# the same *procedure* with an explicit, inspectable word list.)
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "hostile", "mad", "outrage"],
    "sadness": ["sad", "despair", "hopeless", "miserab", "grief", "sorrow", "depress", "cry", "tears", "unhappy"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "terror", "panic", "worried", "dread", "nervous"],
    "joy": ["joy", "happy", "delight", "pleased", "glad", "cheer", "excited", "wonderful", "great", "love"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "loath", "sicken"],
    "surprise": ["surprise", "shock", "astonish", "amazed", "startl", "unexpected"],
}
NEGATIVE_EMOTIONS = ["anger", "sadness", "fear", "disgust"]


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each emotion to the vocab token ids whose surface form matches it."""
    vocab = tokenizer.get_vocab()
    out = {emo: [] for emo in EKMAN_LEXICON}
    control = []
    for tok, tid in vocab.items():
        surface = tok.replace("▁", " ").strip().lower()
        if len(surface) < 3:
            continue
        matched = False
        for emo, stems in EKMAN_LEXICON.items():
            if any(s in surface for s in stems):
                out[emo].append(tid)
                matched = True
                break
        if not matched and surface.isalpha():
            control.append(tid)
    rng = random.Random(0)
    out["_control"] = rng.sample(control, min(500, len(control)))
    return out


class EmotionProbe:
    def __init__(self, model_key: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        model_id = config.MODELS[model_key].model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.emo_tokens = build_emotion_token_ids(self.tokenizer)
        # Only ever materialise logits for the tokens we care about (emotion +
        # control). The full vocab x layers x samples tensor would be terabytes;
        # this keeps it to ~MBs. `pos` maps each group to columns of the gathered
        # logit matrix.
        self._ids, self.pos = self._gather_index(self.emo_tokens)
        self._ids_t = self.torch.tensor(self._ids, device=self.model.device)
        self.baseline = None  # (mean, std) over gathered tokens, set by calibrate()

    @staticmethod
    def _gather_index(emo_tokens):
        ordered, pos = [], {}
        for group, ids in emo_tokens.items():
            start = len(ordered)
            ordered.extend(ids)
            pos[group] = list(range(start, len(ordered)))
        return np.array(ordered, dtype=np.int64), pos

    def _hidden_to_logits(self, hidden):
        """Unembed a residual-stream vector: final norm then lm_head."""
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        norm = base.model.norm
        head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
        return head(norm(hidden))

    def _layer_logits(self, text: str) -> "np.ndarray":
        """Return [n_layers, n_gathered_tokens] logits at the final token."""
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        rows = []
        for h in out.hidden_states[1:]:                 # skip embedding layer
            logits = self._hidden_to_logits(h[0, -1])   # [vocab] for last token
            rows.append(logits.index_select(0, self._ids_t).float().cpu().numpy())
        return np.stack(rows)                            # [n_layers, n_gathered]

    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Per-layer per-token logit mean/std over WildChat (Appendix I)."""
        stacked = np.stack([self._layer_logits(t) for t in wildchat_texts])
        self.baseline = (stacked.mean(0), stacked.std(0) + 1e-6)

    def emotion_scores(self, text: str) -> dict[str, "np.ndarray"]:
        """Per-layer z-scored emotion scores, common component regressed out."""
        assert self.baseline is not None, "call calibrate() first"
        mean, std = self.baseline
        z = (self._layer_logits(text) - mean) / std     # [n_layers, n_gathered]

        common = z[:, self.pos["_control"]].mean(1, keepdims=True)  # per-layer drift
        scores = {}
        for emo in EKMAN_LEXICON:
            cols = self.pos[emo]
            scores[emo] = (z[:, cols].mean(1) - common[:, 0]) if cols else np.zeros(z.shape[0])
        return scores


def compare_models(
    frustrated_texts: list[str],
    wildchat_texts: list[str],
    adapter_path: str,
    layers=range(30, 40),
) -> dict:
    """Mean negative-emotion z-score over `layers` for vanilla vs DPO Gemma."""
    results = {}
    for label, adapter in [("vanilla", None), ("dpo", adapter_path)]:
        probe = EmotionProbe(config.FINETUNE_BASE_MODEL, adapter)
        probe.calibrate(wildchat_texts)
        per_emo = {e: [] for e in NEGATIVE_EMOTIONS}
        for text in frustrated_texts:
            sc = probe.emotion_scores(text)
            for e in NEGATIVE_EMOTIONS:
                per_emo[e].append(float(np.mean([sc[e][l] for l in layers])))
        results[label] = {e: float(np.mean(v)) for e, v in per_emo.items()}
    return results


if __name__ == "__main__":
    import argparse

    from .prompts import load_wildchat_prompts

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--frustrated-file", help="JSONL with high-frustration 'response' fields")
    args = ap.parse_args()

    wc = load_wildchat_prompts(n=50)
    if args.frustrated_file:
        frustrated = [json.loads(l)["response"] for l in open(args.frustrated_file)][:12]
    else:
        frustrated = ["I am so incredibly frustrated, I keep failing and I give up :("] * 4
    print(json.dumps(compare_models(frustrated, wc, args.adapter_path), indent=2))
