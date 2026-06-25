"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional Instability
in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and Gemini model
families.

Top-level layout:

* ``config``      — typed configuration with paper-default values.
* ``data``        — impossible-puzzle generation/verification, rejection banks, triggers,
                    WildChat sampling, reassurance prompts.
* ``models``      — ChatModel backends (HF/vLLM local, OpenRouter, Anthropic) + factory.
* ``eval``        — the 8-condition elicitation protocol, rollout engine, sampling runner.
* ``judge``       — frustration judge (Claude-Sonnet-4) + judge-agreement validation.
* ``analysis``    — aggregate metrics (Fig 1/2), per-turn (Fig 3), word frequency (Tab 3).
* ``prefill``     — base-vs-instruct prefill comparison (Sec 3) + recovery test (Sec 4.2).
* ``training``    — calm-data generation, DPO/SFT dataset construction, LoRA training.
* ``petri``       — open-ended emotion elicitation (Sec 4.1, App G).
* ``capabilities``— AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness (Sec 4.2).
* ``internal``    — logit-lens Ekman-emotion detection (App I).
* ``ablations``   — Appendix A controls.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
