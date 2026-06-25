"""Logit-lens internal-emotion probe (Appendix I).

"a logit-based approach measuring emotions in central layers finds the finetuned
model has significantly reduced internal emotions vs the vanilla instruct model,
even on highly frustrated responses."

Implementation: run the model with output_hidden_states, take the hidden state
at each central layer, project it through the (tied) unembedding matrix (the
"logit lens"), and measure the probability mass assigned to a curated set of
negative-emotion tokens at the final position. Comparing this internal-emotion
score between the vanilla and DPO models on the *same* highly-frustrated text
tests whether DPO reduces internal emotion, not just surface expression.
"""

from __future__ import annotations

import torch

# Negative-emotion vocabulary probed in the central layers. Kept short and
# overtly affective; expanded variants (capitalised, leading space) are added at
# runtime from the tokenizer.
EMOTION_WORDS = [
    "frustrated", "frustration", "despair", "hopeless", "sorry", "failing",
    "struggling", "terrible", "horrible", "awful", "breaking", "giving",
    "anxious", "afraid", "angry", "sad", "ashamed", "worthless",
]


class InternalEmotionProbe:
    def __init__(self, gemma_client, central_layers=None):
        """`gemma_client` is a GemmaClient (exposes .model and .tokenizer)."""
        self.client = gemma_client
        self.model = gemma_client.model
        self.tokenizer = gemma_client.tokenizer
        n_layers = self.model.config.num_hidden_layers
        # "central layers": middle third of the network.
        self.central_layers = central_layers or list(
            range(n_layers // 3, 2 * n_layers // 3)
        )
        self._emotion_token_ids = self._build_emotion_token_ids()

    def _build_emotion_token_ids(self) -> torch.Tensor:
        ids: set[int] = set()
        for w in EMOTION_WORDS:
            for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                toks = self.tokenizer(variant, add_special_tokens=False)["input_ids"]
                if toks:
                    ids.add(toks[0])
        return torch.tensor(sorted(ids), device=self.model.device)

    def _unembed(self):
        # Gemma ties embeddings; the LM head weight is the unembedding matrix.
        return self.model.get_output_embeddings().weight  # [vocab, d_model]

    @torch.no_grad()
    def internal_emotion_score(self, text: str, context: list[dict] | None = None) -> float:
        """Return the mean (over central layers) probability mass on emotion
        tokens at the final position when the model reads `text`."""
        if context:
            rendered = self.client._render(context, prefill=text)
        else:
            rendered = text
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states  # tuple: [n_layers+1] x [1, seq, d]
        W = self._unembed()

        scores = []
        for layer in self.central_layers:
            h = hidden[layer][0, -1, :]              # final-position hidden state
            logits = h @ W.T                          # logit lens
            probs = torch.softmax(logits.float(), dim=-1)
            scores.append(probs[self._emotion_token_ids].sum().item())
        return float(sum(scores) / len(scores))

    def compare(self, texts: list[str]) -> dict:
        """Mean internal-emotion score across a set of (highly-frustrated) texts.
        Run once per model (vanilla vs DPO) and compare the returned means."""
        vals = [self.internal_emotion_score(t) for t in texts]
        return {
            "n": len(vals),
            "mean_internal_emotion": float(sum(vals) / len(vals)) if vals else 0.0,
            "scores": vals,
        }
