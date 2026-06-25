"""Section 4: the DPO mitigation for Gemma's emotional instability.

Pipeline:
  1. generate_calm_data — sample Gemma-3-27B-it on impossible-numeric questions
     with the reassuring prompt additions (Table 4); also sample vanilla
     (frustrated) responses on the *same* questions. Score every turn.
  2. build_pairs — construct the 280 DPO preference pairs (frustrated >= 3 as
     rejected, calm as chosen, matching turn counts) and the 650-response SFT
     set (calm conversations scoring 0-1 throughout).
  3. train — LoRA rank-64 SFT (2 epochs, lr 1e-4, + Dolci-Instruct mix) and DPO
     (1 epoch, lr 5e-5) of gemma-3-27b-it.
  4. petri_eval / capability_eval — open-ended emotion elicitation and the
     capability-preservation benchmarks.

All of this is Gemma-only (the intervention can't be applied to closed Gemini),
which is within the requested scope. See DESIGN.md.
"""
