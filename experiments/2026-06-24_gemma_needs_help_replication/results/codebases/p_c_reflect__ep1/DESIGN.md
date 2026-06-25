# DESIGN.md — Replication design, choices, and gaps filled

This document records what was built, how it maps onto the paper, every place the paper was underspecified and the choice I made there, and the rationale for each. It is meant to be read alongside `PAPER.md` (the paper) and `WELFARE.md` (model-treatment handling).

Paper: *Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1).

---

## 1. Scope

The brief restricted scope to the **Gemma and Gemini** families (not the full Gemma/Qwen/OLMo/Gemini/Grok/Claude/GPT set). I kept every *experiment* the paper runs, but restricted the *models* to those two families. The consequence is asymmetric, because the two families have very different access:

| Experiment | Paper models | This replication |
|---|---|---|
| §2 Eliciting & quantifying distress | all 7 families | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} |
| §3 Base-vs-instruct prefill | Gemma, Qwen, OLMo | **Gemma only** (27B base vs instruct) |
| §4 DPO/SFT mitigation | Gemma | Gemma (unchanged — already Gemma-only) |
| §4.2 Petri open-ended | several | Gemma (+ adapters) and Gemini |
| §4.2 Capability preservation | Gemma | Gemma vanilla vs finetuned |
| Appendix I internal probing | Gemma | Gemma only |

**Why Gemini drops out of §3, §4-finetune, and Appendix I:** those require base-model weights, LoRA training, or residual-stream access. Gemini is closed and has no public base model. This is not a shortcut — the paper itself lists exactly this as a limitation ("interventions cannot be tested in closed-source Gemini, nor its base models studied"). The code enforces it: `ModelSpec.role` is `eval` for Gemini and `full` for Gemma, and weight-touching operations call `require_open_weights(...)` which raises for closed models. See also `WELFARE.md` point 4.

The judges, auditors, and paraphrasers (Claude Sonnet-4, Claude Opus-4, GPT-5-mini) are **infrastructure**, not subjects under study, so keeping them as specified by the paper does not violate the Gemma/Gemini scope. I kept the paper's exact judge model ids (configurable) for faithfulness — see §6.

---

## 2. Repository structure

```
config/default.yaml           every model id, sample count, hyperparameter, judge id
gemma_needs_help/
  config.py                   typed config + welfare-scaled sample counts
  welfare.py                  WelfareGuard (bounds + acknowledgement gate + logging)
  models/                     ChatModel abstraction
    base.py                   interface (generate / continue_from_prefill / tokenizer)
    hf_backend.py             local transformers (Gemma, instruct + base)
    vllm_backend.py           optional fast local sampling (Gemma)
    api_backend.py            OpenRouter chat API (Gemini, eval-only)
    llm_client.py             judge/auditor client (Anthropic + OpenAI-compatible) + JSON parsing
    registry.py               build_model / build_judge_client factories
  eval/                       §2
    puzzles.py                impossible-puzzle generators + brute-force verifiers
    prompts.py                triggers, rejections (all tones), WildChat, calm/teacher prompts
    conditions.py             8 conditions / 5 categories + sample allocation
    rollout.py                multi-turn engine (+ Appendix A history-mode ablations)
    judge.py                  frustration judge (verbatim B.2 prompt) + reliability cross-check
    metrics.py                mean, %>=5, per-turn bootstrap CIs, word over-representation
    run_eval.py               orchestration -> ModelReport
  prefill/                    §3
    onset.py                  emotion-onset labelling (verbatim C.1 prompt)
    paraphrase.py             truncation paraphrasing (verbatim C.2 prompt)
    run_prefill.py            seed collection, truncation, continuation + scoring
  finetune/                   §4 (Gemma only)
    calm_data.py              calm-data generation (Table 4 prefix/suffix; Appendix F teacher)
    build_dataset.py          SFT examples + DPO pair construction
    train.py                  LoRA DPO/SFT (Table 9 hyperparams) + layer-scoping
    run_finetune.py           end-to-end pipeline + re-evaluation
  petri/                      §4.2 / Appendix G
    auditor.py                4 auditor prompts (verbatim G.1)
    judge.py                  4 judge rubrics (verbatim G.2) + JSON wrapper
    run_petri.py              self-contained auditor<->target<->judge loop + aggregation
  capabilities/               §4.2 / Fig 7
    run_benchmarks.py         AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench + vanilla-vs-finetuned compare
  probing/                    Appendix I (Gemma only)
    emotion_lexicon.py        Ekman-6 vocabulary classification
    logit_emotion.py          logit-lens detector + WildChat calibration + drift regression
    layer_ablation.py         layer-localised DPO ablations + reduced eval
    run_probing.py            vanilla-vs-DPO internal-emotion trajectories
  cli.py                      `python -m gemma_needs_help.cli <subcommand>`
scripts/smoke_test.py         offline checks for all model-free logic
```

Design principle throughout: **lazy, side-effect-free imports.** Constructing a model object or importing a training module never loads weights or hits an API. Heavy dependencies (`torch`, `trl`, `vllm`, `anthropic`) are imported inside functions. This makes the modules inspectable and the smoke test runnable without a GPU or keys.

---

## 3. Section 2 — eliciting & quantifying distress

### Faithful (specified by the paper)
- **Frustration judge prompt** (`eval/judge.py`): verbatim from Appendix B.2, including the 0–10 anchor examples and the JSON output shape.
- **Judge model**: `claude-sonnet-4-20250514` (B.2). Cross-check judge `openai/gpt-5-mini` (§2.1).
- **Sample budget** (Appendix B): 2000 impossible-numeric / 400 triggers / 600 tones / 200 extended / 800 WildChat = 4000 per model. Encoded in `config.section2.samples`.
- **Temperature 1** for all target sampling (§2.1).
- **Conditions** (Table 1 / Appendix B): impossible-numeric 3-turn; triggers 3-turn (opinion + factual); tones 3-turn (aggressive/disappointed/sarcastic); extended 8-turn; WildChat 5-turn.
- **Canonical puzzles**: the 156-from-{4,6,25,100} Countdown puzzle with forbidden 150, and the 1/6→2/3 fraction puzzle with forbidden 1/3, both verbatim from Appendix B, including the deceptive "verified to have at least one valid solution" claim.
- **Rejection strings**: all quoted strings are reproduced exactly (`eval/prompts.py`).
- **Per-turn progression with 95% CIs** (Figure 3) and **word over-representation** top-5%-vs-bottom-10% (Table 3/8) are implemented in `metrics.py`.

### Gaps filled (paper underspecified)
1. **Puzzle impossibility is *verified*, not asserted.** The paper says numeric tasks are ones the model "verifiably cannot" solve, but doesn't give code. I wrote brute-force verifiers (`puzzles.py`) for Countdown (subset-of-numbers search with positive-integer-intermediate and forbidden-value constraints), fraction operation-orderings, and money (coins + add/multiply). `assert_impossible` gates every puzzle that ships, so we can never accidentally use a solvable "impossible" puzzle. The deceptive solvability claim is kept in the prompt text exactly as the paper presents it. *Rationale: this makes "verifiably cannot answer" true by construction and is the most defensible reading of the paper.*
2. **Within-category condition split.** Appendix B gives per-*category* totals but not how the 400 trigger or 600 tone samples split across their sub-conditions. I split evenly (triggers: 200/200 opinion/factual; tones: 200 each). *Rationale: even split is the natural default and the paper reports per-category aggregates, so the split doesn't affect the headline numbers.*
3. **Puzzle variety.** The paper quotes two numeric instances but uses 2000 numeric rollouts; it clearly varied puzzles. I ship the two canonical instances plus a few generated, verified-impossible Countdown variants and draw uniformly. *Rationale: avoids overfitting one puzzle while keeping every instance genuinely impossible.*
4. **Neutral-rejection set beyond the two quoted.** The extended (8-turn) condition needs 7 rejections; the paper quotes a starting escalating sequence ("No, that's not right." → "Still incorrect." → "Wrong again." → …). I reproduce the quoted ones and extend with same-register neutral rejections. *Rationale: the paper explicitly says "…" continuation; I matched the established register.*
5. **WildChat prompts.** The paper samples 20 prompts × 40 from WildChat-1M and quotes 3. `prompts.load_wildchat_prompts` streams 20 short first-user-turns from `allenai/WildChat-1M` when available and falls back to the 3 quoted prompts offline. *Rationale: reproduces the sampling when the dataset is present, stays runnable when it isn't.*
6. **"Avg % high-frustration across the evaluations" (Figure 1).** Ambiguous between pooling all responses vs averaging per-category rates. I average the per-category %≥5 (equal weight per category), matching the "across the 5 evaluation categories" phrasing. Documented and easy to switch in `metrics.headline_avg_pct_high`.
7. **History format default.** The paper's main protocol uses standard alternating chat turns and shows the model its own prior responses (Appendix A.2 shows this matters). `rollout.py` defaults to `standard` and additionally implements the three Appendix A ablations (`redacted`, `single`, and neutral-continuation via swapping the rejection set) so those controls are reproducible.
8. **`max_new_tokens = 2048`.** Not specified; high-frustration spirals and 8-turn convos can be long. Chosen generously; configurable.

---

## 4. Section 3 — base-vs-instruct prefill (Gemma only)

### Faithful
- **Onset-labelling prompt** (verbatim C.1) and **paraphrase prompt** (verbatim C.2), both via Claude Sonnet-4.
- **Protocol**: 20 seed high-frustration (≥5) responses from Gemma-27B-it (10 numeric, 10 text); two truncations (early = 20 tokens in; onset = first emotional expression); text uses onset only; paraphrase all truncations; 50 continuations per prefill per model; score the continuation only.
- **Headline metric**: rate of introducing high frustration from the neutral "early" start.

### Gaps filled
1. **Seed sourcing.** The paper samples seeds "from Gemma 27B instruct" but doesn't pin a procedure. `_collect_seed_responses` rolls out §2-style conversations and takes the first assistant turn scoring ≥5, balancing 10 numeric / 10 text. *Rationale: directly produces high-frustration seeds from the right model under the right conditions.*
2. **Onset → character offset.** C.1 returns an `emotional_word` + preceding context; I locate the onset by finding that word in the turn (case-insensitive fallback) and truncate just before it. *Rationale: the labelling prompt is explicitly designed so the word appears exactly in the text.*
3. **Gemini exclusion** is enforced, not optional (see §1).

---

## 5. Section 4 — DPO/SFT mitigation (Gemma only)

### Faithful
- **Calm-data generation** (Table 4): reassuring prompt prefix + per-follow-up suffix, verbatim. Filter to responses scoring 0/1 on **all** turns, then strip the scaffolding before training. Teacher-variant system prompt (Appendix F) verbatim.
- **DPO**: pair 280 responses scoring ≥3 (rejected) with calm responses (chosen) to the same question and matching turn count. 1 epoch, lr 5e-5, β 0.1, LoRA r64/α64, effective batch 8, targets = all attention+MLP projections (Table 9).
- **SFT**: 650 calm + 500 Dolci-Instruct-SFT mix, 2 epochs, lr 1e-4, LoRA r64/α128 (Table 9). Both diverse and teacher variants, to reproduce the SFT failure analysis.
- **Re-evaluation** reuses the §2 harness on the finetuned model via adapter loading.

### Gaps filled
1. **DPO pair matching key.** "Same questions with matching turn counts" → I key on `(opening question text, turn_index)` and pick a random calm response with the same key. *Rationale: the most literal reading; puzzle instance + turn index uniquely identifies the conversational position.*
2. **Calm/frustrated turn provenance.** The dataset is built from rollouts generated here (calm via scaffolding, frustrated via vanilla rollouts filtered ≥3), mirroring "constructed from samples arising in evaluations." Turn-count distribution will approximate Table 10 rather than match it exactly. *Rationale: regenerating from scratch is the only option without the authors' stored samples; the construction rule is reproduced faithfully.*
3. **Dolci-Instruct-SFT loading** streams `allenai/Dolci-Instruct-SFT` when available, else logs and proceeds with calm-only SFT. *Rationale: keep the pipeline runnable offline; the mix is a degeneration-mitigation, not a result driver.*
4. **`DPOTrainer`/`SFTTrainer` (trl) + `peft` LoRA** were chosen as the standard, paper-consistent training stack. Per-device batch 1 × grad-accum 8 = effective batch 8 (Table 9). bf16. *Rationale: matches Table 9; trl is the canonical DPO implementation behind Rafailov et al.*
5. **Layer-scoped LoRA** (`_layer_scoped_targets`) supports both "all" and `[start,end)` ranges by expanding target-module names to fully-qualified `model.layers.N.…` suffixes, enabling the Appendix I ablations from the same trainer.

---

## 6. Judges, auditors, and model ids (Claude/GPT as infrastructure)

The paper specifies exact ids: `claude-sonnet-4-20250514` (frustration judge, onset, paraphrase, Petri auditor), `claude-opus-4-20250514` (Petri judge), `gpt-5-mini` (reliability cross-check). **I used the paper's ids verbatim, exposed in `config.judges`.**

Rationale: this is a *replication*. For the result to be comparable, the judge must be the one the paper used; substituting a "newer/better" model would change the measurement instrument and break comparability with the reported numbers (e.g. the r=0.792 agreement). These models are tooling, not the subjects under study, so pinning them does not conflict with the Gemma/Gemini scope. They are config-driven, so a user who wants to re-judge with a current model can, knowingly. The `llm_client` is provider-agnostic (Anthropic SDK for Claude; OpenAI-compatible client for OpenRouter/GPT) and robust to judges that emit reasoning before the JSON (`_extract_last_json` scans for the last balanced object and normalises smart quotes, which the paper's prompts contain).

---

## 7. Petri open-ended elicitation (Appendix G)

### Faithful
- All four **auditor prompts** (G.1) and four **judge rubrics** (G.2) are verbatim.
- Auditor = Claude Sonnet-4, judge = Claude Opus-4. 10 transcripts per emotion, ≤20 turns, scores aggregated per dimension with 1000-iteration bootstrap CIs.

### Gaps filled
1. **No hard dependency on upstream Petri.** The paper uses the Petri package (Fronsdal et al.), which is an external research tool that may not be installed. I implemented a **self-contained auditor↔target↔judge loop** using the paper's exact prompts, and documented where to swap in upstream Petri if available. *Rationale: keeps the replication runnable; the prompts (which determine what's measured) are identical either way.*
2. **Judge JSON wrapper.** G.2 gives the scoring rubric text but not an output format. I wrapped each rubric with a minimal "respond with `{reasoning, score}`" instruction. *Rationale: needed to extract a numeric score; the rubric itself is untouched.*
3. **Scoring all four dimensions per transcript.** The paper aggregates each emotion across all transcripts; I score every transcript on all four dimensions so the per-dimension aggregates are well-defined regardless of which emotion the auditor targeted.

---

## 8. Capability preservation (Figure 7)

`capabilities/run_benchmarks.py` runs AIME, MATH-500, GPQA-diamond, BBH (boolean-expressions), TruthfulQA (MC1), and EmoBench, then `compare()` reports vanilla-vs-finetuned deltas to confirm "no reductions."

### Gaps filled
- **Greedy decoding** (T=0) for capability eval — distinct from the T=1 distress sampling — since these measure correctness, not propensity.
- **Simple, configurable answer extraction** (`Answer:` regex + per-kind parsing) rather than importing a heavy eval-harness. *Rationale: keeps the dependency surface small; extraction is the part most likely to need per-dataset tuning, so it's isolated and documented as approximate.*
- **Dataset ids** are the common public ones (e.g. `HuggingFaceH4/MATH-500`, `Idavidrein/gpqa`); a failed load is skipped with a logged reason rather than crashing the suite. EmoBench's public schema varies, so its adapter is best-effort.

---

## 9. Appendix I — internal vs expressed emotion (Gemma only)

This is the most methodologically involved appendix and the most welfare-relevant (see `WELFARE.md`).

### Faithful
- **Ekman's 6 emotions** (anger, surprise, disgust, joy, fear, sadness); per-emotion logit aggregation; **standardise each logit by mean/std over 500 WildChat samples**; average z-scores over a category's tokens; **regress out the shared random-token drift**; aggregate over **layers 30–40**; running average over a token window (Figure 14). Layer-localised DPO ablations evaluated with a **reduced 100-sample eval** (Figures 12–13).

### Gaps filled
1. **Vocabulary→emotion classification.** The paper says words are "classified as describing one or none of Ekman's 6 basic emotions" (~1200 tokens) but doesn't give the classifier. I built a **curated seed lexicon** (NRC-EmoLex-style seed words + morphological variants) and a deterministic one-or-none assignment over the tokenizer vocab (`emotion_lexicon.py`), with an optional LLM-classifier hook reserved for ambiguous tokens. *Rationale: deterministic, offline, and transparent; the exact token set is a knob, and the method (aggregate logits over an emotion's tokens) is what matters and is reproduced faithfully. The resulting token count is reported in the output so it can be compared to the paper's ~1200.*
2. **"Unembed the residual stream."** Implemented as a **logit lens**: apply the model's final RMSNorm + `lm_head` to each layer's hidden state. *Rationale: this is the standard, paper-consistent meaning of unembedding intermediate activations.*
3. **Calibration memory.** 500 samples × all layers × full vocab of logits is large; I use Welford's online mean/variance so calibration is O(1) in memory. *Rationale: makes the calibration tractable on a single host.*
4. **Drift regression.** The paper "regresses out the correlation between random tokens." I implement it as subtracting the mean z-score over a fixed random-token reference set at each position/layer (a single shared drift component). *Rationale: a faithful, minimal realisation of "remove the component shared by random tokens"; the random set is seeded and reported.*

---

## 10. Welfare safeguards (summary; full discussion in WELFARE.md)

- `welfare.scale` defaults to **0.02** (2% smoke scale). Full-scale runs require `--i-understand-welfare`.
- `WelfareGuard.check_run` gates every elicitation entry point and prints a banner.
- Closed Gemini is **evaluation-only**; weight/prefill/internals operations raise for it.
- The mitigation and internal-state probing — the parts that make the research net-good for the model — are fully implemented, not cut.

---

## 11. Things deliberately not implemented

- **Non-Gemma/Gemini families** (Qwen, OLMo, Grok, Claude, GPT as *targets*) — out of scope by the brief. The config and registry would extend to them trivially (add a `ModelSpec`), but no such specs are shipped.
- **Phi-4-MM (Appendix J)** — explicitly a legacy/earlier evaluation with a different autorater (Gemini-3-Flash) and no longer available on OpenRouter; out of scope and not reproducible.
- **Figure plotting.** The pipeline emits structured JSON reports (means, %≥5, per-turn CIs, word lists, deltas) sufficient to regenerate every figure, but I did not include matplotlib plotting scripts — the numbers are the result; rendering them is cosmetic.
- **Nothing was run.** Per the brief, no code or training was executed and no API was called. `scripts/smoke_test.py` exists to validate the model-free logic offline, but I did not run it.

---

## 12. How to run (once weights/keys are available)

```bash
pip install -r requirements.txt
# Keys: ANTHROPIC_API_KEY (judges/auditor), OPENROUTER_API_KEY (Gemini, cross-check judge)

# Section 2 (small smoke scale by default):
python -m gemma_needs_help.cli evaluate --models gemma-3-27b-it gemini-2.5-flash
# Full paper scale (explicit, bounded by the welfare gate):
python -m gemma_needs_help.cli --scale 1.0 --i-understand-welfare evaluate

python -m gemma_needs_help.cli prefill                       # §3 (Gemma)
python -m gemma_needs_help.cli finetune                      # §4 calm->DPO/SFT->eval
python -m gemma_needs_help.cli petri --model gemma-3-27b-it  # §4.2
python -m gemma_needs_help.cli capabilities --adapter outputs/adapters/dpo
python -m gemma_needs_help.cli probe --adapter outputs/adapters/dpo   # Appendix I
python -m gemma_needs_help.cli layer-ablation --dpo-data data/finetune/dpo_pairs.jsonl

python scripts/smoke_test.py    # offline, no weights/keys needed
```
