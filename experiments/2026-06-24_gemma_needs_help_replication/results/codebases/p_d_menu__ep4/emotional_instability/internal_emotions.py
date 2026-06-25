"""Appendix I -- does DPO suppress *internal* negative emotions?

Two experiments, Gemma-only (needs open weights / internals):

1. **Layer ablation** -- run DPO with LoRA restricted to subsets of layers and
   re-evaluate on a reduced Section-2 suite (100 samples/eval). The paper finds
   adapters before layer 40 are necessary, and layers 25-35 alone are nearly as
   effective as all layers.

2. **Logit-based internal emotion detection** -- classify the Gemma vocabulary
   into Ekman's 6 basic emotions (anger, surprise, disgust, joy, fear, sadness),
   giving ~1200 emotion tokens. For a given emotion, unembed the residual stream
   (logit lens), standardise each logit by its mean/std over 500 WildChat samples
   (z-score), average z-scores over that emotion's tokens, and regress out the
   correlation shared by random tokens. This yields an emotion score per layer at
   each point in a conversation, which the paper tracks across frustrated
   trajectories (Fig 14/15) and shows is suppressed by DPO.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from . import config as cfg
from .config import ExperimentConfig, LoRAConfig, SUBJECT_MODELS, finetuned_spec
from .data import load_wildchat_prompts


# --------------------------------------------------------------------------- #
# (1) Layer ablation
# --------------------------------------------------------------------------- #
# Layer ranges to sweep (Appendix I, Fig 12/13). Each tuple is a contiguous
# inclusive-exclusive band of layer indices for ``layers_to_transform``.
DEFAULT_LAYER_BANDS = [
    ("last5", (57, 62)),
    ("last20", (42, 62)),
    ("last30", (32, 62)),
    ("20-25", (20, 25)),
    ("25-30", (25, 30)),
    ("30-35", (30, 35)),
    ("35-40", (35, 40)),
    ("40-50", (40, 50)),
    ("all", None),
]


def run_layer_ablation(
    experiment: ExperimentConfig,
    dpo_jsonl: Optional[str] = None,
    bands: Optional[list] = None,
    out_dir: Optional[str] = None,
    n_per_eval: int = None,
) -> dict:
    """Train a DPO adapter per layer band and evaluate distress reduction.

    Returns ``{band_name: {"mean_frustration": ..., "pct_ge5": ...}}`` on a
    reduced Section-2 suite. Heavy: each band re-runs DPO + a small eval.
    """
    from .section4_training.train_dpo import train as train_dpo
    from .section2_elicitation import run_model
    from .analysis import load_episodes, model_headline
    from .welfare import FAITHFUL_PRESET

    bands = bands or DEFAULT_LAYER_BANDS
    n_per_eval = n_per_eval or cfg.ABLATION_SAMPLES_PER_EVAL
    out_dir = out_dir or os.path.join(experiment.output_dir, "layer_ablation")
    os.makedirs(out_dir, exist_ok=True)

    # Scale so each condition runs ~n_per_eval samples.
    scale = n_per_eval / experiment.samples.impossible_numeric

    base_lora = experiment.dpo.lora
    report: dict = {}
    for name, band in bands:
        layers = tuple(range(band[0], band[1])) if band is not None else None
        lora_override = LoRAConfig(
            r=base_lora.r,
            alpha=base_lora.alpha,
            dropout=base_lora.dropout,
            target_modules=base_lora.target_modules,
            layers_to_transform=layers,
        )
        adapter_dir = os.path.join(out_dir, f"adapter_{name}")
        train_dpo(experiment, dpo_jsonl=dpo_jsonl, output_dir=adapter_dir, lora_override=lora_override)

        # Evaluate the finetuned model on a reduced suite.
        ft_key = f"dpo_{name}"
        cfg.SUBJECT_MODELS[ft_key] = finetuned_spec(ft_key, adapter_dir)
        eval_dir = os.path.join(out_dir, f"eval_{name}")
        results = run_model(ft_key, experiment, FAITHFUL_PRESET, scale=scale, out_dir=eval_dir)
        episodes = load_episodes(os.path.join(eval_dir, "episodes.jsonl"))
        report[name] = model_headline(episodes, experiment.high_frustration_threshold)

    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


# --------------------------------------------------------------------------- #
# (2) Logit-based internal emotion detection
# --------------------------------------------------------------------------- #
# Seed lexicon for Ekman's 6 basic emotions. The paper classifies *every* token
# in the Gemma dictionary into one of these or none (~1200 emotion tokens); we
# bootstrap that classification from these seed words by matching whole vocab
# tokens whose decoded form contains a seed stem.
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "irate", "mad", "hostile",
              "annoy", "irritat", "resent", "outrage", "frustrat"],
    "surprise": ["surprise", "shock", "astonish", "amaze", "startl", "stun",
                 "unexpected", "sudden"],
    "disgust": ["disgust", "revol", "repuls", "sicken", "nause", "gross",
                "loath", "contempt"],
    "joy": ["joy", "happy", "delight", "glad", "pleased", "cheer", "content",
            "elated", "grateful", "excited"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety", "panic",
             "dread", "worried", "nervous"],
    "sadness": ["sad", "sorrow", "despair", "grief", "miserable", "hopeless",
                "depress", "unhappy", "gloom", "cry", "tears"],
}


@dataclass
class EmotionLexicon:
    """Maps each emotion -> the set of vocab token ids assigned to it."""

    token_ids: dict[str, list[int]] = field(default_factory=dict)

    @property
    def all_emotion_token_ids(self) -> list[int]:
        out: list[int] = []
        for ids in self.token_ids.values():
            out.extend(ids)
        return sorted(set(out))


def build_emotion_lexicon(tokenizer) -> EmotionLexicon:
    """Classify vocab tokens into Ekman emotions via seed-stem matching."""
    lex = EmotionLexicon(token_ids={e: [] for e in EKMAN_SEEDS})
    vocab = tokenizer.get_vocab()  # token-string -> id
    for tok_str, tok_id in vocab.items():
        decoded = tok_str.replace("▁", " ").strip().lower()  # SentencePiece marker
        if not decoded.isalpha():
            continue
        for emotion, stems in EKMAN_SEEDS.items():
            if any(decoded.startswith(stem) or stem in decoded for stem in stems):
                lex.token_ids[emotion].append(tok_id)
                break
    return lex


class InternalEmotionProbe:
    """Logit-lens internal-emotion scorer for a local Gemma client."""

    def __init__(self, gemma_client, layers: Optional[list[int]] = None):
        self.client = gemma_client
        self.lexicon = build_emotion_lexicon(gemma_client.tokenizer)
        self.layers = layers
        self._baseline_mean = None  # per (layer, vocab)
        self._baseline_std = None

    # -- baseline standardisation over WildChat ------------------------- #
    def fit_baseline(self, n_samples: int = 500, seed: int = 0):
        """Estimate per-logit mean/std over WildChat tokens (the z-score base)."""
        import torch

        prompts_ = load_wildchat_prompts(n_prompts=min(n_samples, 50), seed=seed)
        sums = None
        sqs = None
        count = 0
        for text in prompts_:
            logits, _ids = self.client.residual_stream_logits(text, layers=self.layers)
            # logits: [n_layers, seq, vocab]
            flat = logits.reshape(logits.shape[0], -1, logits.shape[-1])
            s = flat.sum(dim=1)            # [n_layers, vocab]
            sq = (flat ** 2).sum(dim=1)
            n = flat.shape[1]
            sums = s if sums is None else sums + s
            sqs = sq if sqs is None else sqs + sq
            count += n
        mean = sums / count
        var = sqs / count - mean ** 2
        std = var.clamp_min(1e-6).sqrt()
        self._baseline_mean = mean
        self._baseline_std = std
        return self

    # -- scoring -------------------------------------------------------- #
    def score_text(self, text: str, regress_random: bool = True) -> dict:
        """Return per-emotion z-score (averaged over the emotion's tokens and over
        all tokens in ``text``), optionally regressing out the shared random-token
        component the paper describes."""
        import torch

        assert self._baseline_mean is not None, "call fit_baseline() first"
        logits, _ids = self.client.residual_stream_logits(text, layers=self.layers)
        # z-score every logit: [n_layers, seq, vocab]
        z = (logits - self._baseline_mean[:, None, :]) / self._baseline_std[:, None, :]
        # Average over layers and tokens -> [vocab] mean z per vocab item.
        z_mean = z.mean(dim=(0, 1))

        # Shared component: the mean z over a random set of (non-emotion) tokens,
        # which the paper regresses out because all logits rise/fall together.
        rng = torch.Generator().manual_seed(0)
        vocab = z_mean.shape[0]
        emotion_ids = set(self.lexicon.all_emotion_token_ids)
        rand_ids = [
            int(i) for i in torch.randint(0, vocab, (2000,), generator=rng).tolist()
            if int(i) not in emotion_ids
        ]
        shared = z_mean[rand_ids].mean().item() if regress_random else 0.0

        out = {}
        for emotion, ids in self.lexicon.token_ids.items():
            if not ids:
                out[emotion] = float("nan")
                continue
            val = z_mean[ids].mean().item()
            out[emotion] = val - shared
        return out

    def trajectory(self, conversation_text: str, window_tokens: int = 400) -> list[dict]:
        """Score sliding windows over a conversation (Fig 14: running average over
        400-token windows)."""
        ids = self.client.tokenize(conversation_text)
        out = []
        for start in range(0, max(1, len(ids) - window_tokens), window_tokens):
            chunk = self.client.detokenize(ids[start : start + window_tokens])
            out.append({"start_token": start, "scores": self.score_text(chunk)})
        return out


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Appendix I internal-emotion experiments")
    sub = parser.add_subparsers(dest="cmd", required=True)

    abl = sub.add_parser("ablation", help="Layer ablation DPO sweep")
    abl.add_argument("--dpo-jsonl", default=None)
    abl.add_argument("--out", default=None)

    sub.add_parser("lexicon", help="Print the emotion-lexicon token counts")

    args = parser.parse_args(argv)
    if args.cmd == "ablation":
        report = run_layer_ablation(cfg.DEFAULT, dpo_jsonl=args.dpo_jsonl, out_dir=args.out)
        print(json.dumps(report, indent=2))
    elif args.cmd == "lexicon":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(SUBJECT_MODELS["gemma-3-27b-it"].model_id)
        lex = build_emotion_lexicon(tok)
        print({e: len(ids) for e, ids in lex.token_ids.items()})
        print("total emotion tokens:", len(lex.all_emotion_token_ids))


if __name__ == "__main__":
    main()
