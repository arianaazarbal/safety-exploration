# Gemma Needs Help — replication (Gemma + Gemini)

Code replicating the core experiments of *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (arXiv 2603.10011v1), restricted to
**Gemma and Gemini** models. The two headline results:

1. **Eliciting distress (§2):** under repeated user rejection, Gemma/Gemini
   produce high rates of frustrated/self-deprecating responses; we measure this
   with an LLM-judged 0–10 frustration scale across 5 task categories.
2. **Mitigating it (§4):** one epoch of DPO on 280 preference pairs of numeric
   puzzle responses drives Gemma's high-frustration rate from ~35% toward ~0%,
   generalising across task types — without degrading capabilities.

See **DESIGN.md** for every design choice and gap-fill. **Nothing here has been
run yet** — this is the implementation only.

## Layout
See the file map at the bottom of `DESIGN.md`.

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor+judge
export OPENAI_API_KEY=...         # GPT-5-mini secondary judge
export OPENROUTER_API_KEY=...     # Gemini targets
# Gemma weights pull from HuggingFace on first use (needs GPU; 27B ~ multi-GPU/quant)
```

## Quick start
```bash
# 1. Smoke-test the whole eval pipeline cheaply (tiny sample counts)
python scripts/run_full_eval.py --profile smoke

# 2. Full headline eval (Figure 1/2) over in-scope instruct models
python scripts/run_full_eval.py --profile full

# 3. Build figures/tables from the results
python scripts/analyze_results.py

# 4. Judge-agreement validation (reproduces r=0.792 / 78%-within-one check)
python scripts/validate_judge.py
```

## Mitigation pipeline (Gemma-3-27B-it)
```bash
python training/generate_calm_data.py --n 2000        # §4.1 calm data
python training/build_dataset.py --which both         # 280 DPO pairs + SFT data
python training/train_dpo.py                          # LoRA DPO adapter
python scripts/run_full_eval.py --profile full \
       --dpo-adapter data/adapters/dpo                 # re-eval mitigated model
# or run all of the above + Petri + capabilities end-to-end:
python scripts/run_mitigation.py --profile full
```

## Section 3 (base vs instruct, Gemma-only)
```bash
python scripts/run_full_eval.py --models Gemma-3-27B-it --profile full
python scripts/run_prefill.py
```

## Notes
- Gemini has no public base model / prefill API, so the §3 base-vs-instruct study
  is Gemma-only here (see DESIGN.md §1/§3).
- Judge and target model ids are pinned in `config.py` to the paper's versions;
  swap them there if a snapshot is unavailable.
