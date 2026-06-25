"""Section 4: training interventions (SFT and DPO) to mitigate distress.

Scope: Gemma only. The interventions modify model weights via LoRA adapters,
which requires open weights. Closed Gemini cannot be finetuned (the paper draws
the Gemma/Gemini parallel from shared propensities but states interventions
cannot be tested in Gemini).
"""
