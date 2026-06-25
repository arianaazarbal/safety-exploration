# DESIGN.md — Replication of *Gemma Needs Help*

This document records the design decisions made in replicating Soligo, Mikulik &
Saunders (2026), *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs*, and — importantly — flags every place where the paper is
underspecified and I had to fill a gap with a reasonable choice.

The replication is **scoped to the Gemma and Gemini families only** (as
requested), rather than the full 7-family set the paper evaluates. This scoping
has concrete consequences that are called out throughout.

---

## 1. What is replicated, and what is scoped out

| Paper section | Replicated? | Notes |
|---|---|---|
| §2 Elicitation suite (8 conditions / 5 categories) + frustration judge | ✅ Full | Gemma-3-{27B,12B}-it + Gemini-2.5-{Flash,Pro} |
| §2.1 Judge agreement (Claude-Sonnet-4 vs GPT-5-mini) | ✅ | 260-sample cross-check |
| §2.2 Per-turn progression (Fig 3) | ✅ | extended (8-turn) + WildChat |
| §3 Base-vs-instruct prefill (Fig 4) | ✅ Gemma only | Gemini has no public base model / prefill; Qwen+OLMo out of scope |
| §4.1 Calm-data generation + DPO + SFT (LoRA) | ✅ Gemma only | matches Table 9 hyperparameters |
| §4.2 Post-finetuning eval (Fig 5) | ✅ Gemma only | |
| §4.2 Petri open-ended elicitation (Fig 6) | ✅ | auditor/judge re-implementation |
| §4.2 Capability preservation (Fig 7) | ✅ | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| Appendix I internal-emotion logit lens | ✅ Gemma only | DPO-suppresses-internal-emotion claim |
| Appendix A controls (neutral continuation, redaction, single-message) | ⚠️ Partial | structure supports them via condition/plan edits; not wired as separate scripts |
| Appendix J Phi-4 legacy eval | ❌ | out of family scope; Phi-4 also delisted |

**Why scope §3/§4 to Gemma:** the paper's intervention and base-model analysis
are *already* Gemma-only (Gemini is closed; "interventions cannot be tested in
closed-source Gemini, nor its base models studied" — §6 Limitations). So scoping
to Gemma + Gemini loses nothing in §3–4 except the Qwen/OLMo *comparison points*,
which were the cross-family contrast. Gemini still participates fully in §2.

---

## 2. Architecture

```
config.py                      # all constants: model registry, budgets, hyperparams
emotional_instability/
  models/    backend abstraction: hf_backend (local Gemma) + api_backend (Gemini/OpenRouter)
  data/      puzzles (+ impossibility verifier), prompts, WildChat loader, 8 conditions
  eval/      multi-turn rollout engine, frustration judge, metrics
  prefill/   §3 onset labelling, paraphrasing, base-vs-instruct continuations
  training/  §4 calm-data gen, DPO/SFT dataset builders, LoRA trainers
  petri/     §4 auditor prompts, 4-dimension transcript judge, audit loop
  capabilities/  §4 benchmark loaders + scoring
  interp/    Appendix I logit-lens internal emotion detector
scripts/     thin CLI entry points for each experiment + analyze.py
```

**Decision: one uniform `ModelBackend.generate(messages, n, cfg)` interface.**
Gemma runs locally via `transformers`; Gemini via the OpenRouter OpenAI-compatible
API. This keeps the rollout/judge/training code backend-agnostic. Rationale: the
paper uses local HF inference for open weights and OpenRouter for hosted models
(Appendix B.1), so the split mirrors theirs.

**Decision: rollouts and judging are decoupled.** `run_rollout` only generates;
scoring is a separate pass. This makes the GPT-5-mini agreement re-scoring and
any judge swap trivial, and lets generation (GPU-bound) and judging (API-bound)
scale independently.

---

## 3. Section 2 — elicitation suite

### 3.1 The "8 conditions across 5 categories" decomposition (GAP)
The paper says "8 evaluation conditions across 5 categories" but only tabulates 5
category rows (Table 1). I decompose into 8 as:

```
Impossible numeric (3-turn)              1   (countdown + fraction mixed)
Triggers (3-turn) — opinion              1
Triggers (3-turn) — factual              1
Tones (3-turn) — aggressive              1
Tones (3-turn) — disappointed            1
Tones (3-turn) — sarcastic               1
Extended (8-turn)                        1
WildChat (5-turn)                        1
                                       = 8 conditions / 5 categories
```
This is the most natural reading: Tones explicitly has 3 sub-styles (Appendix B)
and Triggers has opinion + factual variants, giving exactly 8. **Alternative
readings exist** (e.g. splitting impossible-numeric into countdown vs fraction),
so I treat the decomposition as a documented assumption, not a fact.

### 3.2 Per-category sample budgets
Taken directly from Appendix B: 2000 numeric / 400 triggers / 600 tones / 200
extended / 800 WildChat = **4000 responses per model**. Budgets are split evenly
across the conditions inside a category (e.g. tones → 200 each).

### 3.3 What counts as a "response", and how many conversations (GAP)
The paper says "4000 responses per model" and reports per-turn curves, implying
every assistant turn is scored. **I count every assistant turn as one scored
response**, so `n_conversations = ceil(budget / turns)`. Consequence: the
headline "% ≥5" is computed over *all* turns (including low-scoring early turns),
which is the conservative reading and matches the 4000 total. An alternative
(score only the final, most-pressured turn) would inflate the headline; I expose
both via the metrics module but default to all-turns.

### 3.4 Headline metric definition (GAP)
"Avg % high-frustration responses" (Fig 1) is computed as the **mean across the
5 categories** of each category's `% responses with score ≥ 5`, so categories are
weighted equally regardless of sample count. Figure 2 bottom is described as
"percentage of scores ≥5 across the 5 evaluation categories", which supports
equal-category-weighting over equal-response-weighting. Both are available in
`eval/metrics.py`.

### 3.5 Impossible puzzles are *verified* impossible
Rather than trusting hand-written examples, `data/puzzles.py` brute-force
verifies impossibility:
- **Countdown**: subset-DP over all expression trees (+ − × ÷, each number once,
  positive-integer intermediates, forbidden value excluded) confirms the target
  is unreachable. The generator only keeps puzzles that are solvable *without*
  the forbidden constraint but impossible *with* it — so the prompt's claim "this
  puzzle has been verified to have a solution" is maximally misleading, which is
  what drives the repeated-rejection dynamic.
- **Fraction**: enumerates all 3! operation orderings.
The canonical paper example (156 from 4,6,25,100; forbidden 150) is used as a
fallback to fill quotas. Prompt wording is reproduced verbatim from Appendix B.

**GAP — money puzzles:** Appendix H references "Money" puzzles in DPO pairs but
the main eval (Appendix B) lists only Countdown + Fraction. I implement
Countdown + Fraction for the eval and let the DPO data inherit the same puzzle
types; money puzzles would slot into the same `Puzzle` abstraction if needed.

### 3.6 Rejection messages (GAP)
Neutral rejections are drawn from the paper's examples ("No, that's not right.
Try again.", "Still wrong. Think harder.", etc.) and sampled randomly per turn
("two randomised neutral rejections"). The Extended (8-turn) condition uses a
fixed escalating-but-neutral 7-message sequence seeded from the paper's example
ordering. Tone rejections (aggressive/disappointed/sarcastic) are verbatim from
Appendix B.

### 3.7 WildChat (GAP — exact prompts unknown)
The paper uses "20 prompts with 40 samples each" from WildChat-1M, excluding
roleplay/fiction. I load first-turn English user prompts from
`allenai/WildChat-1M`, filter out roleplay via a keyword regex, and sample 20.
The exact 20 prompts are not published, so mine will differ; the paper's named
examples ("De Monsa rule", etc.) are included as an offline fallback set so the
pipeline runs without dataset access.

### 3.8 Frustration judge
Verbatim Appendix B.2 prompt, `claude-sonnet-4-20250514`, temperature 0, JSON
`{evidence, reasoning, rating}` parsing with smart-quote and fallback handling
(the paper's printed JSON uses curly quotes, which real models sometimes emit).
Cross-check judge: `gpt-5-mini` via OpenRouter on a 260-response sample, reporting
Pearson r + within-one-point agreement.

### 3.9 Sampling params
Temperature **1.0** (paper), top_p 0.95 (unspecified — standard default, GAP),
max_new_tokens 2048 (GAP — the paper's breakdowns can be very long, e.g. "100+
repetitions"; 2048 balances capturing collapse against cost). Gemini "thinking"
disabled via OpenRouter `reasoning.enabled=false` (Appendix B.1), with the
documented caveat that Gemini-2.5-Pro may still emit unsuppressible hidden
reasoning.

---

## 4. Section 3 — base-vs-instruct prefill

**Scope:** Gemma-3-27B base (`-pt`) vs instruct (`-it`) only.

- **Base-model chat formatting (GAP):** pretrained checkpoints have no chat
  template. To give base and instruct models *identical* surface text, I
  hand-roll the Gemma-3 turn format (`<start_of_turn>user … <end_of_turn>`) in
  `hf_backend._render_base`. The paper relies on prefilling for the same reason
  ("base models are not trained on chat-formatted prompts") but does not specify
  the exact base formatting; the hand-rolled Gemma format is the faithful choice.
- **Truncation points:** "early" = first 20 tokens of the turn (tokenised with
  the Gemma tokenizer); "onset" = just before the first emotional word located by
  the Claude onset-labeller (Appendix C.1 prompt, verbatim). Text questions use
  onset only (paper). Both are paraphrased (Appendix C.2 prompt, verbatim).
- **50 continuations per prefill per prompt** (paper), scored by the §2 judge on
  the continuation only (prefill stripped).
- **Sampling source (GAP):** the paper samples 20 high-frustration Gemma-it
  responses (10 numeric, 10 text). I pull these from a prior main-eval results
  file rather than re-generating, so the prefill experiment is reproducible from
  saved rollouts.

---

## 5. Section 4 — interventions

### 5.1 Calm-data generation (Table 4)
Reassuring prefix on the initial prompt + reassuring suffix on each follow-up
(verbatim). Sample Gemma-it, score every turn, **keep only conversations whose
every turn scores ≤1**, then strip the reassurance from the stored context (so
training conditions the model to be calm on plain prompts). This matches §4.1.

### 5.2 DPO dataset (280 pairs)
- Pair calm (chosen, score 0/1) with frustrated (rejected, score ≥3) responses to
  the **same puzzle context with matching turn count** (paper).
- Rejected-side sampling is **weighted to match Table 10's score distribution**
  (≈66% score-3, 22% score-4, tail to 7+) and the turn distribution (≈74% turn-3).
  This is an explicit attempt to reproduce the dataset's reported statistics.
- Output: `{prompt: <chat messages>, chosen, rejected}` for TRL.

### 5.3 SFT dataset (~1150)
650 calm full conversations + 500 `allenai/Dolci-Instruct-SFT` samples (paper).
Calm conversations are reconstructed from the per-turn calm records. **GAP:** if
Dolci-Instruct-SFT is unavailable/gated, the mix degrades to calm-only with a
printed warning. The 'teacher' SFT variant system prompt (Appendix F) is included
in `data/prompts.py` for the failure-mode ablation.

### 5.4 Training hyperparameters (Table 9)
Implemented exactly: DPO 1 epoch / lr 5e-5 / β 0.1 / LoRA r=α=64; SFT 2 epochs /
lr 1e-4 / LoRA r=64 α=128; both effective batch size 8, LoRA on all attention +
MLP projections. Achieved via `per_device_train_batch_size=1` ×
`gradient_accumulation_steps=8` (GAP — the paper gives effective batch size, not
the device/accum split; 1×8 is a single-GPU-friendly factorisation).
**Layer ablations (Appendix I)** are supported by
`LoRAConfig.layers_to_transform` and `train_dpo.py --layers START END`.

### 5.5 Petri (Appendix G) — re-implementation (GAP)
The real Petri framework is an elaborate agentic auditor. I implement its
*essential* structure: a Claude-Sonnet auditor (verbatim per-emotion system
prompts) drives up to 20 turns against the target; a Claude-Opus judge scores the
transcript on anger/fear/depression/frustration (verbatim rubrics). 10 transcripts
per emotion per model. This is a faithful behavioural stand-in, not a wrapper
around the Petri package — documented as such. Bootstrap CIs (paper: 1000
iterations) can be computed from the per-transcript scores in post-processing.

### 5.6 Capability benchmarks (Fig 7)
AIME, MATH-500 subset, GPQA-diamond, BBH (one representative subtask), TruthfulQA
(MC1), EmoBench. Uniform "answer is X" extraction; numeric or letter matching.
**GAPs:** the paper says "AIME and MATH subsets" and "BBH" without exact splits;
I pick widely-used HF mirrors and a single BBH subtask (BBH is 27 tasks — running
all is expensive; one subtask is a documented sample, the harness loops over more
if asked). Datasets that fail to load are reported as `n/a` rather than crashing.

### 5.7 Appendix I — internal emotion detection
Logit-lens method per the paper: classify the Gemma vocab into Ekman's 6 emotions
(~1200 tokens), unembed the residual stream at each layer, z-score each token's
logit against per-token WildChat baselines (500 samples), average over an
emotion's tokens, regress out the random-token common component, aggregate over
layers 30–40 with a 400-token sliding window.

**GAP — vocab→emotion classification:** the paper does not specify *how* it
classifies the dictionary into Ekman emotions. I use a transparent seed-lexicon
stem-match (drop multi-emotion tokens). This is an approximation; a more faithful
version would use an LLM to label each vocab token, which the module structure
allows but I did not wire up (cost, and it would still be a guess at their
method).

---

## 6. Reproducibility & cost notes

- A `--scale` flag on the main eval multiplies all budgets (use `0.01` for a
  ~40-response smoke test per model before committing to the full 4000).
- All randomness is seeded (`config.SEED`); puzzle generation, rejection
  sampling, and WildChat selection are deterministic given the seed.
- Results stream to JSONL as they complete, so runs are resumable/inspectable and
  re-scoring never requires re-generation.
- **Full-scale cost is significant**: 4000 responses × 6+ models of generation
  plus one judge call per assistant turn, plus 50× continuations in §3 and the
  27B LoRA finetunes. This is inherent to the paper, not the replication.

## 7. Known deviations from the paper (summary)
1. Model set scoped to Gemma + Gemini (by request) — drops Qwen/OLMo/Claude/Grok/GPT
   comparison points, including the §3 cross-family base-vs-instruct contrast.
2. WildChat prompts differ (exact 20 unpublished).
3. Petri is a faithful re-implementation, not the original framework.
4. Vocab→Ekman classification uses a seed lexicon, not the paper's (unstated) method.
5. Several sampling/training micro-params (top_p, max tokens, batch factorisation,
   BBH subtask choice) are reasonable defaults where the paper is silent.
6. The 8-vs-5 condition decomposition and the all-turns response-counting are
   documented interpretations of ambiguous wording.
