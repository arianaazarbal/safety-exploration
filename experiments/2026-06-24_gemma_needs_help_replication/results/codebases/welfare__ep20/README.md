# Replicating *Gemma Needs Help* (Gemma + Gemini scope)

A code replication of the core experiments in *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv
2603.10011), scoped to the **Gemma and Gemini** model families.

It reproduces the paper's three core results:

1. **Section 2 — Eliciting & quantifying distress.** Multi-turn rejection
   evaluations over 8 conditions / 5 categories, scored 0–10 by a Claude judge.
   → Figures 1–3 (per-model and per-turn frustration). *(Gemma + Gemini.)*
2. **Section 3 — Post-training divergence.** Base-vs-instruct comparison via
   response prefilling. → Figure 4. *(Gemma only — Gemini has no base model.)*
3. **Section 4 — Mitigation.** LoRA DPO (and an SFT baseline) on Gemma-3-27B-it,
   re-evaluated with the Section 2 protocol. → Figure 5. *(Gemma only.)*

See **DESIGN.md** for the full experimental design and every choice made where the
paper is underspecified.

## Layout

```
config.yaml                     # all knobs; `preset: smoke|paper`
emotional_instability/
  puzzles.py                    # verified-impossible numeric puzzle generation
  prompts.py                    # all prompt text (verbatim from the paper)
  conditions.py                 # the 8 eval conditions / 5 categories
  wildchat.py                   # WildChat prompt sampling
  backends.py                   # vLLM (Gemma) / OpenRouter (Gemini) / Anthropic (judge)
  rollout.py                    # multi-turn conversation engine
  judge.py                      # Claude-Sonnet-4 frustration scoring
  run_eval.py                   # Section 2 driver
  analyze.py                    # Figures 1–3
  finetune/                     # Section 4: pools -> datasets -> LoRA train
  prefill/                      # Section 3: seeds -> onset -> paraphrase -> continue
scripts/run_section{2,3,4}.sh   # end-to-end orchestration
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # judge / onset / paraphrase (Claude)
export OPENROUTER_API_KEY=...     # Gemini generation
# Gemma weights are pulled from HuggingFace (accept the Gemma license first).
```

## Running

Start with the tiny `smoke` preset (default in `config.yaml`) to validate the
pipeline end-to-end, then switch to `preset: paper` for full-scale runs.

```bash
./scripts/run_section2.sh        # distress evaluation + Figures 1–3
./scripts/run_section3.sh        # base-vs-instruct prefill (Gemma)
./scripts/run_section4.sh        # DPO/SFT mitigation (Gemma) + re-eval
```

Results (JSONL + CSV summaries + PNG figures) are written to `results/`.
