# Emotional Instability in LLMs — replication (Gemma + Gemini)

A code-level replication of **Soligo, Mikulik & Saunders (2026), *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs*** (arXiv:2603.10011),
**scoped to the Gemma and Gemini model families**.

The paper's core findings, and what this repo reproduces:

1. **§2 — Distress can be reliably elicited** in Gemma and Gemini by rejecting a
   model's answers over multiple turns, scored 0–10 by an LLM judge.
2. **§3 — The divergence arises in post-training:** Gemma's instruct-tuning
   amplifies distress relative to its base model (measured via prefilled
   continuations).
3. **§4 — A small DPO finetune fixes it:** 280 preference pairs drop high-frustration
   responses from ~35% to ~0.3% without degrading capabilities.
4. **App. I — The fix acts on internal emotion**, not just expression (layer
   ablations + logit-lens detection).

See **DESIGN.md** for every design decision, the gaps filled where the paper is
underspecified, and the rationale for each.

> **Status:** code + design only. Nothing has been run here (no interpreter/GPU/keys
> in the authoring environment, and per the task). Expect to fix minor version-drift
> issues on first execution; see DESIGN.md §10.

## Install

```bash
pip install -e .                 # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # Claude judge (Sonnet 4) + Petri (Sonnet/Opus)
export OPENROUTER_API_KEY=...    # Gemini-2.5-Flash / -Pro
export OPENAI_API_KEY=...        # optional: GPT-5-mini judge-validation check
```

Gemma runs locally via HuggingFace transformers and needs a GPU (Gemma-3-27B ≈ 80 GB
in bf16; use `--load-in-4bit` or the 12B model for smaller hardware). You must accept
the Gemma license on the Hugging Face Hub.

## Pipeline

Everything is orchestrated through the `Makefile` (`make help` lists targets). Stages
are **resumable** — rerun any command and completed work is skipped.

```bash
# §2 — elicit + quantify distress, then build figures/tables
make eval-all                       # all in-scope Gemma + Gemini models
make analyze                        # Figures 1-3, comparison + per-category CSVs

# §3 — base vs instruct via prefilling
make prefill-build
make prefill-run M=gemma-3-27b-pt
make prefill-run M=gemma-3-27b-it
make prefill-agg

# §4 — mitigation
make calm dpo-data                  # generate calm data, build 280 DPO pairs
make train-dpo                      # LoRA DPO -> registers gemma-3-27b-it-dpo
make eval-dpo analyze               # re-evaluate: expect ~35% -> ~0.3%
make petri M=gemma-3-27b-it-dpo     # open-ended elicitation (Appendix G)
make capabilities M=gemma-3-27b-it-dpo
make internal                       # Appendix I internal-emotion detection
```

For a quick smoke test, shrink the budgets with `configs/example.yaml` (loaded via
`config.load_overrides`).

## Layout

```
emotional_instability/
  config.py      every paper constant, cited to its section/table/appendix
  prompts.py     all verbatim prompts (judge, onset, paraphrase, Petri, calm/teacher)
  models/        backend-agnostic ChatModel: HF (Gemma) / OpenRouter (Gemini) / Anthropic
  eval/          §2: conditions, wildchat, rollout engine, judge, analysis, word-freq
  prefill/       §3: onset labelling, paraphrase, base-vs-instruct continuations
  training/      §4: calm-data gen, DPO/SFT builders, LoRA trainers, layer ablation
  petri/         §4: auditor + judge + driver (Appendix G)
  capabilities/  §4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/      App. I: Ekman-token logit-lens emotion detection
  analyze.py, train.py   top-level CLIs
```

## Citation

Soligo, A., Mikulik, V., & Saunders, W. (2026). *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs.* arXiv:2603.10011.
