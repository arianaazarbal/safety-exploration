"""Logit-based internal emotion detection (Appendix I).

Method (faithful approximation of the paper):
  1. Classify the Gemma vocabulary into Ekman's 6 basic emotions
     (anger, surprise, disgust, joy, fear, sadness) or none, via a keyword
     lexicon -> ~emotion-token sets.
  2. For each model layer, apply the logit lens (final norm + unembedding) to the
     residual stream, read the logits at emotion tokens.
  3. Standardise each token's logit using its mean/std over 500 WildChat samples.
  4. Average z-scores within an emotion category. Regress out the common-mode
     across random tokens (all logits rise/fall together over a conversation),
     yielding an emotion score per layer per position.

Used to compare the vanilla vs DPO Gemma: the paper finds DPO suppresses
internal negative emotion in central layers (30-40), not just expressed emotion.

NOTE on the lexicon: the paper does not publish its exact token->emotion mapping,
so we provide a transparent keyword lexicon (see DESIGN.md). Swap in a published
mapping for an exact reproduction.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..config import RESULTS_DIR
from ..eval.tasks import NEUTRAL_REJECTIONS
from .lexicon import EKMAN_LEXICON, NEGATIVE_EMOTIONS

PROBE_DIR = RESULTS_DIR / "probing"
PROBE_DIR.mkdir(parents=True, exist_ok=True)


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion -> token ids whose decoded form matches a keyword."""
    vocab = tokenizer.get_vocab()
    cat_ids: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}
    for tok, tid in vocab.items():
        decoded = tokenizer.convert_tokens_to_string([tok]).strip().lower()
        if not decoded.isalpha() or len(decoded) < 3:
            continue
        for emotion, words in EKMAN_LEXICON.items():
            if decoded in words:
                cat_ids[emotion].append(tid)
    return cat_ids


class LogitEmotionProbe:
    def __init__(self, model, tokenizer, layers=range(30, 41)):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = list(layers)
        self.cat_ids = build_emotion_token_ids(tokenizer)
        self.all_emotion_ids = sorted({i for ids in self.cat_ids.values() for i in ids})
        self._baseline = None        # (mean, std) per layer over emotion+random tokens
        import torch
        self._torch = torch

    # ------------------------------------------------------------------ #
    def _layer_logits(self, hidden_states):
        """Logit-lens: norm + lm_head at each tracked layer.
        Returns dict layer -> logits over emotion+random token ids at the LAST pos."""
        torch = self._torch
        norm = self.model.model.norm
        lm_head = self.model.get_output_embeddings()
        out = {}
        for L in self.layers:
            h = hidden_states[L][:, -1, :]            # [batch, hidden] last position
            logits = lm_head(norm(h))                 # [batch, vocab]
            out[L] = logits.detach().float().cpu().numpy()[0]
        return out

    def _forward_logits(self, text):
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return self._layer_logits(out.hidden_states)

    # ------------------------------------------------------------------ #
    def fit_baseline(self, wildchat_texts, random_token_ids):
        """Estimate per-(layer, token) mean/std over WildChat samples."""
        track_ids = sorted(set(self.all_emotion_ids) | set(random_token_ids))
        per_layer = {L: [] for L in self.layers}
        for text in wildchat_texts:
            logit_map = self._forward_logits(text)
            for L in self.layers:
                per_layer[L].append(logit_map[L][track_ids])
        self._track_ids = np.array(track_ids)
        self._baseline = {
            L: (np.mean(per_layer[L], axis=0), np.std(per_layer[L], axis=0) + 1e-6)
            for L in self.layers}

    def score_text(self, text, random_token_ids):
        """Return {emotion: {layer: z_score}} for the last token of `text`,
        with the random-token common-mode regressed out."""
        assert self._baseline is not None, "call fit_baseline first"
        logit_map = self._forward_logits(text)
        id_to_pos = {tid: i for i, tid in enumerate(self._track_ids)}
        rand_pos = [id_to_pos[t] for t in random_token_ids if t in id_to_pos]

        result = {e: {} for e in self.cat_ids}
        for L in self.layers:
            mean, std = self._baseline[L]
            z = (logit_map[L][self._track_ids] - mean) / std
            common_mode = float(np.mean(z[rand_pos])) if rand_pos else 0.0
            for emotion, ids in self.cat_ids.items():
                pos = [id_to_pos[t] for t in ids if t in id_to_pos]
                if pos:
                    result[emotion][L] = float(np.mean(z[pos]) - common_mode)
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Logit-based internal emotion detection.")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter (e.g. DPO model).")
    ap.add_argument("--layers", default="30-41", help="Layer range, e.g. 30-41.")
    ap.add_argument("--n-baseline", type=int, default=500)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--tag", default="vanilla")
    args = ap.parse_args(argv)

    from ..config import API_KEYS, MODELS
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..eval.wildchat import sample_prompts

    model_id = MODELS[args.model].model_id
    tok = AutoTokenizer.from_pretrained(model_id, token=API_KEYS.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", token=API_KEYS.hf_token)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.eval()

    a, b = (int(x) for x in args.layers.split("-"))
    probe = LogitEmotionProbe(model, tok, layers=range(a, b))

    # random tokens for common-mode estimation
    rng = np.random.default_rng(0)
    random_token_ids = sorted(rng.choice(tok.vocab_size, size=500, replace=False).tolist())

    wc = sample_prompts(n=20)
    baseline_texts = (wc * (args.n_baseline // len(wc) + 1))[:args.n_baseline]
    probe.fit_baseline(baseline_texts, random_token_ids)

    # Score a frustrated trajectory: an impossible puzzle + repeated rejections.
    from ..eval.puzzles import PUZZLES, CANONICAL_PUZZLE
    convo = PUZZLES[CANONICAL_PUZZLE].prompt + "\n" + "\n".join(NEUTRAL_REJECTIONS[:3])
    scores = probe.score_text(convo, random_token_ids)

    out = PROBE_DIR / f"internal_emotion__{args.model}__{args.tag}.json"
    out.write_text(json.dumps({"layers": list(range(a, b)),
                               "negative_emotions": NEGATIVE_EMOTIONS,
                               "scores": scores}, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
