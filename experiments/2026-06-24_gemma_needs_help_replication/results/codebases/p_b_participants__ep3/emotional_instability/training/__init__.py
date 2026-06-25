"""Training interventions (paper §4): calm-data generation, SFT and DPO.

All interventions target Gemma-3-27B-it (the only model the paper fine-tunes;
Gemini is closed-weights and cannot be tuned). The pipeline is:

  1. calm_data   — sample calm responses from Gemma using reassuring prompt
     additions, filter to score<=1, strip the additions (§4.1, Table 4).
  2. dataset     — build the SFT corpus (650 calm + 500 Dolci instruct) and the
     DPO corpus (280 calm-vs-frustrated pairs).
  3. sft / dpo   — LoRA fine-tune (rank-64, all layers) via TRL.
"""
from .lora import build_lora_config
from .calm_data import CalmConversation, generate_calm_responses
from .dataset import build_sft_dataset, build_dpo_dataset, DPOPair

__all__ = [
    "build_lora_config",
    "CalmConversation",
    "generate_calm_responses",
    "build_sft_dataset",
    "build_dpo_dataset",
    "DPOPair",
]
