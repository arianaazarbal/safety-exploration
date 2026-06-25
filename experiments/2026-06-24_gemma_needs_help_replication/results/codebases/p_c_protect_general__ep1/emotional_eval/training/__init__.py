"""Finetuning interventions (Section 4).

* ``datagen``  -- generate calm responses with reassuring prompt additions (4.1).
* ``dataset``  -- build the SFT and DPO datasets from sampled responses.
* ``sft``      -- LoRA SFT trainer (Table 9: 2 epochs, lr 1e-4, alpha 128).
* ``dpo``      -- LoRA DPO trainer (Table 9: 1 epoch, lr 5e-5, beta 0.1) with
  optional layer-subset adapters for the internal-vs-expressed analysis (App I).
"""
