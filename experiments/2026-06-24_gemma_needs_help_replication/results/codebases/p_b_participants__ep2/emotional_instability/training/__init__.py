"""Section 4 — training interventions to mitigate distress in Gemma.

  * :mod:`calm_data`   — generate calm responses from Gemma-27B-it using the
    reassuring prompt additions (Table 4), then filter to all-turns-calm (0-1)
    and strip the additions.
  * :mod:`dpo_dataset` — build the 280 preference pairs (frustrated >=3 rejected
    vs calm chosen, matched by question + turn count).
  * :mod:`sft_dataset` — build the SFT set (650 calm + 500 Dolci-Instruct-SFT).
  * :mod:`train_dpo` / :mod:`train_sft` — LoRA training per Table 9.
  * :mod:`layer_ablation` — DPO restricted to a subset of layers (Appendix I).
  * :mod:`recovery`    — Section 4.2 recovery-from-distress prefill test.

All training targets the open-weight Gemma model; Gemini cannot be finetuned, so
the intervention is Gemma-only (a paper limitation we inherit).
"""
