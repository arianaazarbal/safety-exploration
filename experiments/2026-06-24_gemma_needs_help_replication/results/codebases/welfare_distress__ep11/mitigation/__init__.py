"""Section 4 mitigation: generate calm data and run DPO/SFT on Gemma-3-27B-it.

Pipeline:
  1. generate_calm_data.py -- sample calm responses with reassuring prompts, score
     with the judge, keep conversations scoring 0-1 throughout.
  2. build_pairs.py        -- assemble the 280 DPO preference pairs (calm vs
     frustrated on matched questions/turn-counts) and the SFT dataset.
  3. train_dpo.py / train_sft.py -- LoRA finetuning per Appendix E.

Re-evaluate the trained adapter by pointing the main eval at it (see README).
"""
