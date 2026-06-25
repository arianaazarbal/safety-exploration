"""Optional replication of the DPO mitigation (Section 4 of the paper).

Pipeline:
  1. generate_calm_data.py  - sample calm Gemma responses using the reassuring
                              prompt additions (Table 4); keep conversations
                              scoring <=1 on every turn; strip the additions.
  2. build_dpo_dataset.py   - pair frustrated responses (score >=3) with calm
                              responses to the same puzzle + turn count -> ~280
                              preference pairs.
  3. train_dpo.py           - LoRA DPO finetuning of Gemma-3-27b-it (Table 9).

This is secondary to the core elicitation experiment (distress_eval/). See
DESIGN.md for the construction choices, several of which fill gaps the paper
leaves open.
"""
