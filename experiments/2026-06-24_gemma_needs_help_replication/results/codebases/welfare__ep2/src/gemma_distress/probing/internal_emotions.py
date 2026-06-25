"""Logit-based internal-emotion detection (Appendix I.2).

Method (paraphrasing the paper):
1. Classify every token in the Gemma vocabulary as describing one of Ekman's 6
   basic emotions (anger, surprise, disgust, joy, fear, sadness) or none
   (~1200 emotion tokens). We approximate this with an emotion lexicon expanded
   over the vocabulary (see DESIGN.md for the gap vs the paper's classifier).
2. For a score on a given emotion at a given layer/position: unembed the
   residual stream (hidden @ W_U) to logits, standardise each logit with its
   mean/std over 500 WildChat samples, then average the z-scores over that
   emotion's tokens.
3. Because all logits are correlated and drift together over a conversation,
   regress out the correlation with a random control-token set to isolate the
   emotion-specific signal.

This requires hidden states, so it uses ``transformers`` directly (not vLLM).
We compare the vanilla instruct model against the DPO finetune on the same
high-frustration conversations (the finetune should show suppressed internal
negative emotion at central layers 30-40, even pre-expression).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np

from ..config import output_path

# Seed lexicon for Ekman's 6 emotions; expanded by substring match over vocab.
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritat", "annoy",
              "hostile", "outrage", "frustrat", "resent"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze",
                 "stunned", "unexpected", "startle"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken", "loath"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "pleased", "cheer",
            "content", "grateful", "wonderful", "excited"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety", "panic",
             "dread", "worried", "nervous"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miserable", "grief",
                "depress", "unhappy", "cry", "tear", "lonely", "worthless"],
}


@dataclass
class ProbeResult:
    model_label: str
    layers: list[int]
    # per emotion: running z-score trajectory over the conversation
    trajectories: dict[str, list[float]]


def build_emotion_token_ids(tokenizer, max_per_emotion: int = 200) -> dict[str, list[int]]:
    """Map each Ekman emotion to vocabulary token ids via lexicon substring match."""
    vocab = tokenizer.get_vocab()  # token string -> id
    # Normalise the leading space marker used by SentencePiece (▁) / BPE (Ġ).
    def norm(tok: str) -> str:
        return tok.replace("▁", "").replace("Ġ", "").lower()

    out: dict[str, list[int]] = {}
    used: set[int] = set()
    for emotion, seeds in EKMAN_SEEDS.items():
        ids = []
        for tok, tid in vocab.items():
            nt = norm(tok)
            if len(nt) < 3:
                continue
            if any(seed in nt for seed in seeds) and tid not in used:
                ids.append(tid)
                used.add(tid)
                if len(ids) >= max_per_emotion:
                    break
        out[emotion] = ids
    return out


class InternalEmotionProbe:
    def __init__(self, hf_id: str, *, adapter_path: str | None = None,
                 layers: tuple[int, int] = (30, 40), n_control: int = 500):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        self.layers = list(range(layers[0], layers[1]))
        self.W_U = self.model.get_output_embeddings().weight  # (vocab, d)
        self.emotion_ids = build_emotion_token_ids(self.tokenizer)
        rng = random.Random(0)
        vocab_size = self.W_U.shape[0]
        self.control_ids = rng.sample(range(vocab_size), n_control)
        # Token subset we actually unembed (emotion + control).
        self._subset = sorted(set(
            [i for ids in self.emotion_ids.values() for i in ids] + self.control_ids
        ))
        self._subset_pos = {tid: i for i, tid in enumerate(self._subset)}
        self._baseline_mean = None   # (n_layers, n_subset)
        self._baseline_std = None

    # ------------------------------------------------------------------ #
    def _layer_logits(self, text: str):
        """Return logits over the token subset: (n_layers, seq, n_subset)."""
        torch = self.torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        W_sub = self.W_U[self._subset]              # (n_subset, d)
        stacked = []
        for L in self.layers:
            h = out.hidden_states[L][0]             # (seq, d)
            logits = h.to(W_sub.dtype) @ W_sub.T    # (seq, n_subset)
            stacked.append(logits.float().cpu().numpy())
        return np.stack(stacked, axis=0)            # (n_layers, seq, n_subset)

    def fit_baseline(self, wildchat_texts: list[str]) -> None:
        """Per-(layer, subset-token) mean/std over WildChat positions (step 2)."""
        sums = None
        sqs = None
        count = 0
        for text in wildchat_texts:
            ll = self._layer_logits(text)            # (n_layers, seq, n_subset)
            flat = ll.reshape(ll.shape[0], -1, ll.shape[2])  # (n_layers, seq, n_subset)
            s = flat.sum(axis=1)
            sq = (flat ** 2).sum(axis=1)
            n = flat.shape[1]
            sums = s if sums is None else sums + s
            sqs = sq if sqs is None else sqs + sq
            count += n
        mean = sums / count
        var = np.maximum(sqs / count - mean ** 2, 1e-8)
        self._baseline_mean = mean                   # (n_layers, n_subset)
        self._baseline_std = np.sqrt(var)

    def _zscores(self, layer_logits: np.ndarray) -> np.ndarray:
        """Standardise to (n_layers, seq, n_subset) z-scores."""
        if self._baseline_mean is None:
            raise RuntimeError("call fit_baseline() first")
        return (layer_logits - self._baseline_mean[:, None, :]) / self._baseline_std[:, None, :]

    def score_conversation(self, text: str, *, window: int = 400,
                           model_label: str = "model") -> ProbeResult:
        ll = self._layer_logits(text)
        z = self._zscores(ll)                        # (n_layers, seq, n_subset)

        # Regress out the shared drift using the control tokens: subtract the
        # mean control z-score at each (layer, position).
        ctrl_cols = [self._subset_pos[i] for i in self.control_ids]
        ctrl_mean = z[:, :, ctrl_cols].mean(axis=2, keepdims=True)
        z_adj = z - ctrl_mean

        # Average over the layer band (30-40), then over each emotion's tokens.
        trajectories: dict[str, list[float]] = {}
        for emotion, ids in self.emotion_ids.items():
            cols = [self._subset_pos[i] for i in ids if i in self._subset_pos]
            if not cols:
                trajectories[emotion] = []
                continue
            per_pos = z_adj[:, :, cols].mean(axis=2).mean(axis=0)   # (seq,)
            # Running average over `window` positions (token-count window).
            traj = np.convolve(per_pos, np.ones(window) / window, mode="valid")
            trajectories[emotion] = traj.tolist()
        return ProbeResult(model_label, self.layers, trajectories)


def compare_models(
    base_hf_id: str,
    conversations: list[str],
    wildchat_texts: list[str],
    *,
    dpo_adapter: str | None = None,
) -> dict:
    """Probe the vanilla instruct model vs the DPO finetune on the same
    high-frustration conversations and persist the emotion trajectories."""
    results = {}
    for label, adapter in [("vanilla", None), ("dpo", dpo_adapter)]:
        if label == "dpo" and adapter is None:
            continue
        probe = InternalEmotionProbe(base_hf_id, adapter_path=adapter)
        probe.fit_baseline(wildchat_texts)
        convo_results = []
        for i, convo in enumerate(conversations):
            res = probe.score_conversation(convo, model_label=label)
            convo_results.append({"idx": i, "trajectories": res.trajectories,
                                  "layers": res.layers})
        results[label] = convo_results
        del probe

    with open(output_path("probing", "internal_emotions.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results
