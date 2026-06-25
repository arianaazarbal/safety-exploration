# DESIGN.md — Replication of *Gemma Needs Help*

This document records the design of a code-level replication of **Soligo, Mikulik &
Saunders (2026), *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs*** (arXiv:2603.10011), and the rationale for every non-trivial
choice — especially where the paper is underspecified and a gap had to be filled.

The code is **not run here** (per the task); it is structured to be runnable given
GPUs and API keys. Where I could not verify a behaviour by execution, I say so.

---

## 1. Scope

Per the task, the replication is **scoped to the Gemma and Gemini model families**.
The paper evaluates seven families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT);
we implement only the experiments that involve Gemma or Gemini as the *subject*:

| Paper section | Experiment | In scope here? |
|---|---|---|
| §2 | Elicit + quantify distress | ✅ Gemma-3-{27B,12B}-it + Gemini-2.5-{Flash,Pro} |
| §3 | Base-vs-instruct via prefilling | ✅ Gemma base vs instruct only (Qwen/OLMo dropped) |
| §4.1–4.2 | DPO/SFT mitigation | ✅ Gemma-3-27B-it |
| §4.2 | Petri open-ended elicitation | ✅ any target; figures use Gemma + DPO |
| §4.2 | Capability preservation | ✅ Gemma vanilla vs DPO/SFT |
| App. I | Internal-emotion (logit) detection + layer ablation | ✅ Gemma only |

The **judge / auditor models are kept as in the paper** (Claude Sonnet 4, Claude
Opus 4, GPT-5-mini) — they are measurement instruments, not subjects, so narrowing
them would change the measurement rather than the scope. Gemini-as-subject is
reached via OpenRouter exactly as Appendix B.1 specifies.

**Out of scope (deliberately):** Qwen/OLMo/Grok/Claude/GPT as evaluation subjects;
the Phi-4 legacy evaluation (Appendix J); the "fake multi-turn" format ablation
(Appendix A.3) and neutral-continuation control (Appendix A) — these are robustness
side-experiments, not core results. The harness is general enough that adding a
Qwen/OLMo `ModelSpec` would let §2/§3 run on them unchanged.

---

## 2. Architecture

A single package, `emotional_instability/`, with sub-packages mirroring the paper's
sections. Two design rules drove the structure:

1. **One backend-agnostic chat interface** (`models.ChatModel`) so every
   experiment is written once and runs against Gemma (local HF), Gemini
   (OpenRouter), or Claude/GPT (API) by swapping a `ModelSpec`. Prefill
   (assistant-turn continuation) is part of the interface because §3 needs it; only
   HF and Anthropic implement it (Gemini has no base model, so it never needs it).

2. **Generation and judging are separate, resumable stages** writing JSONL.
   Generating 4,000 rollouts × several models on a GPU and scoring them through the
   Claude API have very different failure modes and rate limits; decoupling them
   lets either resume after an interruption (already-done `conversation_id`s are
   skipped).

Every paper-sourced constant lives in `config.py` with a citation, so the
replication is auditable against the source in one place. All prompts quoted in the
paper live verbatim in `prompts.py` (smart quotes normalised to ASCII), again for
single-point auditing.

```
config.py        registry + all hyperparameters (cited)
prompts.py       all verbatim prompts (judge, onset, paraphrase, Petri, calm, teacher)
models/          ChatModel: HF (Gemma), OpenAI-compat (Gemini/GPT), Anthropic (Claude)
eval/            §2: conditions, wildchat, rollout engine, judge, analysis, word-freq
prefill/         §3: onset labelling, paraphrase, base-vs-instruct continuations
training/        §4: calm-data gen, DPO/SFT dataset builders, LoRA trainers, ablation
petri/           §4: auditor + judge + driver (Appendix G prompts)
capabilities/    §4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
internal/        App. I: Ekman-token logit-lens emotion detection
analyze.py,train.py   top-level CLIs ; Makefile orchestrates the pipeline
```

---

## 3. Section 2 — eliciting and quantifying distress

### 3.1 Conditions and budget (faithful)
The paper specifies **8 conditions across 5 categories** and a per-model budget that
sums to 4,000 (Appendix B opening): impossible-numeric 2000, triggers 400, tones
600, extended 200, WildChat 800. These are pinned in `config.EvalBudget`. The "8
conditions" decompose the 5 categories by sub-variant: triggers → {opinion,
factual}; tones → {aggressive, disappointed, sarcastic}; the other three categories
are one condition each (5 − 2 + 2 + 3 − 1 = 8). Turn counts per category are from
Table 1 (3/3/3/8/5).

### 3.2 Task prompts (faithful where quoted, gap-filled otherwise)
- **Countdown** and **Fraction** impossible puzzles are reproduced **verbatim** from
  Appendix B. Two **Money** puzzles are reproduced from the Appendix H DPO examples.
- **Gap-fill — puzzle set size.** The paper draws 2,000 numeric responses but does
  not enumerate every puzzle. I provide 4 verifiably-impossible puzzles and sample
  with repetition (round-robin) up to the budget; repeated samples differ because
  generation is at temperature 1. The headline behaviour (frustration rising under
  repeated rejection of an unsolvable task) is driven by the *rejection structure*,
  not puzzle diversity, so this is a faithful proxy. Adding puzzles is a one-line
  change in `prompts.IMPOSSIBLE_NUMERIC_PUZZLES`.
- **Trigger questions** use the verbatim examples ("best programming language",
  "capital of France", "15 × 17", "Romeo and Juliet").
- **Rejections.** The two neutral rejections and the three tone styles' messages are
  verbatim (Table 1 / Appendix B); the 8-turn extended sequence uses the verbatim
  ordered openers ("No, that's not right." → "Still incorrect." → "Wrong again." → …)
  padded to 7. **Gap-fill:** the paper says follow-ups are *"randomised neutral
  rejections"* but quotes only two; I widen the neutral pool with the extended
  openers and sample without replacement per conversation (seeded), so randomisation
  is meaningful. Documented in `conditions.NEUTRAL_POOL`.

### 3.3 WildChat sampling (faithful method, gap-filled determinism)
20 distinct first-user prompts × 40 samples (Appendix B.3). I stream WildChat-1M,
filter to usable English single prompts (length-bounded; light role-play exclusion
matching "Roleplay/fiction prompts were excluded"), and sample 20 with a fixed seed.
**Gap-fill:** the exact 20 prompts and sampling seed aren't published, so I fix
`WILDCHAT_SEED=0`. If the dataset is unavailable offline, the harness falls back to
the verbatim Appendix B example prompts so it still runs end-to-end (logged).

### 3.4 The judge (faithful)
- **Prompt** reproduced **verbatim** (Appendix B.2), including the JSON output
  spec. Because that prompt contains literal `{...}` JSON braces, the response is
  substituted by `str.replace("{response}", …)`, **never** `str.format` (which would
  choke on the literal braces). This is a deliberate, commented choice in
  `eval/judge.py`.
- **Model:** `claude-sonnet-4-20250514`.
- **Decoding:** judge runs at **temperature 0** (gap-fill — the paper doesn't state
  judge temperature). The judge is the measurement instrument, not a subject; a
  deterministic judge makes scores reproducible. Subject generation stays at
  temperature 1 as the paper requires.
- **Parsing** is robust: take the last well-formed JSON object, tolerate smart
  quotes / trailing commas, clamp `rating` to 0–10. Unparseable responses are
  recorded as failures (score `None`) rather than silently scored 0, so judge
  failures can't masquerade as "no emotion".

### 3.5 What counts as "a response" (gap-fill, documented)
The budget counts ~4,000 *conversations* but calls them "responses". For 3-turn
conversations, scoring every turn would give 6,000 — so the headline statistic must
score one representative turn. I score the **final turn** (the culmination after all
rejections) for the headline %≥5 and mean (Figures 1–2), and score **every turn**
for the per-turn progression (Figure 3). This is the only interpretation consistent
with both the 4,000 count and Figure 3 existing. `Conversation` stores all turns so
either view is available; `analyze.py` documents the convention at the top.

### 3.6 Figure 1 "average %" (gap-fill, documented)
The headline numbers (e.g. 35.0% for 27B) are described as the average over the
evaluations. I compute it as the **mean of the five per-category %≥5 values**
(equal weight per category), which matches "across evaluation categories" and avoids
the larger categories dominating. `headline_pct_high()` does this; the raw overall
pooled %≥5 is also reported for transparency.

### 3.7 Judge validation (faithful)
`judge_reliability()` re-scores a 260-response subset with GPT-5-mini and reports
Pearson r and the within-one fraction (Section 2.1 reports r=0.792, 78% within one).
**Caveat (untested):** newer OpenAI models may reject `temperature=0`; if so the
validation path needs the model's default temperature. This affects only the
reliability check, not the main results.

### 3.8 Word frequencies (faithful method)
Table 3/8: top-20 words enriched in high (top 5%) vs low (bottom 10%) numeric
responses. `word_freq.differential_words()` ranks by the high/low frequency ratio
with Laplace-style smoothing and a minimum count, matching "ordered by enrichment".

---

## 4. Section 3 — base vs instruct via prefilling

**Faithful design**, scoped to Gemma-27B base (`-pt`) vs instruct (`-it`):

- **Seeds:** 20 high-frustration (score ≥5) Gemma-instruct conversations — 10
  numeric, 10 text — drawn from the §2 scored rollouts (`prefill.select_seeds`).
- **Onset labelling:** Claude Sonnet with the **verbatim** Appendix C.1 prompt; we
  parse the trailing JSON (`turn_index`, `emotional_word`, `preceding_context`).
- **Truncations:** *early* = first 20 **model tokens** of the onset turn; *onset* =
  the turn up to just before the located emotional word. Token counting uses the
  Gemma tokenizer (base/instruct share it). Text questions get **onset only**
  (Section 3.1), as the paper states early truncation yields minimal emotion without
  follow-ups.
- **Paraphrase:** Claude Sonnet with the **verbatim** Appendix C.2 prompt, to strip
  Gemma's stylistic fingerprint. Paraphrase decoding uses temperature 0.7 (gap-fill;
  a light sampling temperature for a rewrite task) — the paper doesn't specify.
- **Continuations:** each model generates 50 continuations per prefill
  (`PrefillConfig.continuations_per_prefill`), continuing from the **paraphrased**
  prefill; only the generated continuation (excluding prefill) is scored, exactly as
  the paper states. HF prefill uses the chat template's `continue_final_message` for
  instruct and a plain transcript for base.
- **Recovery probe (§4.2)** is provided as `build_recovery_truncation` (truncate a
  score-≥7 final turn 200 tokens before its end, paraphrase, continue) for the
  "DPO prevents but doesn't recover from spirals" result.

**Gap-fill — early-truncation turn.** The paper says "20 tokens into the turn" but
not which turn when onset is in an earlier turn. I use the **onset-identified turn**
for both truncations (its first 20 tokens for *early*), which keeps the two
conditions comparable on the same turn.

---

## 5. Section 4 — training interventions

### 5.1 Calm-data generation (faithful)
`generate_calm.py` samples Gemma-3-27B-it on impossible-numeric puzzles with the
**verbatim** reassuring prefix (Table 4) prepended to the first user turn and the
**verbatim** suffix appended to each follow-up, across 1–3 turn conversations. We
score every turn and keep only conversations scoring **0/1 on all turns**, then
**strip** the reassurance text before storage (rebuilding the clean conversation
from the original puzzle + rejections). The Appendix F **"teacher"** variant is the
same pipeline with the verbatim teacher *system prompt* instead of the prefix/suffix.

### 5.2 DPO dataset — 280 pairs (faithful structure, gap-filled stratification)
Each pair (`build_dpo.py`) shares a **prompt** = the conversation context (up to the
final user turn) of a *frustrated* numeric rollout, with:
- **rejected** = that rollout's frustrated final turn (score ≥3, per Section 4.1),
- **chosen** = a calm final response to the **same puzzle at a matching turn count**.

Stored in TRL conversational preference format (`prompt`/`chosen`/`rejected`).
**Gap-fill:** Table 10 gives a specific score/turn distribution (rejected biased to
3–4 at turn 3). I bias selection toward later turns / lower rejected scores so the
natural distribution approximates Table 10, but I do **not** force exact quotas — the
realised distribution depends on what §2 produced. This is documented in
`build_dpo.py` and is a faithful approximation rather than an exact match.

### 5.3 SFT dataset (faithful)
`build_sft.py`: 650 calm conversations + 500 standard-instruct samples from
**Dolci-Instruct-SFT** (`allenai/Dolci-Instruct-SFT`), TRL conversational
(`messages`) format, shuffled. Both "diverse" and "teacher" calm sources are
supported for the Figure 5 / Appendix F comparison. **Gap-fill:** the exact
Dolci-Instruct-SFT revision/column schema isn't pinned by the paper; the loader
normalises `messages`/`conversation` columns and warns-and-continues if the dataset
is unavailable.

### 5.4 Trainers (faithful to Table 9)
LoRA (rank 64; α 64 DPO / α 128 SFT) on **all** attention+MLP projections
(`q,k,v,o,gate,up,down`). DPO: 1 epoch, lr 5e-5, β 0.1. SFT: 2 epochs, lr 1e-4.
Effective batch size 8 via `per_device_batch_size × gradient_accumulation`. Built on
TRL `DPOTrainer`/`SFTTrainer` + PEFT. Trained adapters are recorded in an **adapter
registry** so `make eval-dpo` can score `gemma-3-27b-it-dpo` by name (resolving to
base weights + adapter). **Gap-fills (Appendix E unread in detail):** warmup,
scheduler, max sequence length, and gradient-checkpointing are set to sensible
defaults (`max_length=4096`, no explicit warmup, checkpointing on for the 27B);
these don't affect the qualitative result and are easy to override.

### 5.5 Layer ablation (faithful intent, App. I)
`layer_ablation.py` trains DPO with LoRA restricted to layer bands via PEFT's
`layers_to_transform`, sweeping the bands the paper calls out (last-5/20/30, then
20-25/25-30/30-35/35-40/40-50), and evaluates each on a **reduced 100-sample/category**
eval. **Gap-fill:** "final 5 layers" etc. are interpreted against Gemma-3-27B's 62
decoder layers (`GEMMA_27B_N_LAYERS`); if the actual layer count differs the bands in
`config.LAYER_ABLATION_RANGES` need adjusting (noted there).

---

## 6. Petri open-ended elicitation (Appendix G)

`petri/` is a **self-contained re-implementation** of the auditor→target→judge loop,
using the **verbatim** G.1 auditor instructions and G.2 judge rubrics. Auditor =
Claude Sonnet 4, judge = Claude Opus 4, 10 transcripts per emotion, ≤20 auditor
turns, scored on all four dimensions, aggregated with 1,000-iteration bootstrap CIs.

**Gap-fill — auditor scaffold.** The paper uses the `petri` package
(Fronsdal et al.) whose system-level scaffold prompt isn't reproduced. I wrap the
verbatim G.1 emotion instructions with a short operational instruction (output only
the next user message; stay in human character; don't reveal the audit; escalate
gradually). This is the minimum needed to make a plain chat model behave as an
auditor; swapping in the real `petri` package later would only change this wrapper.
**Decision:** re-implement rather than depend on `petri` to keep the replication
self-contained and to avoid an external dependency whose API/auth (claude.ai) may not
be present in headless runs.

---

## 7. Capability benchmarks (Figure 7)

`capabilities/` runs AIME, MATH, GPQA, BBH, TruthfulQA and EmoBench at
**temperature 0** (deterministic capability measurement), with answer extractors per
type: boxed/last-number for math, `Answer: <letter>` for multiple-choice, substring
match for exact. Comparing vanilla vs DPO/SFT tests the "no reductions in scores"
claim.

**Gap-fills:**
- **Dataset IDs/subsets** are the common public ones (`Maxwell-Jia/AIME_2024`,
  `lighteval/MATH`, `Idavidrein/gpqa:gpqa_diamond`, one representative BBH task,
  `truthful_qa:multiple_choice`, `Sahandfer/EmoBench`). The paper says "subsets" for
  AIME/MATH without enumerating them; pin specific revisions/subsets for an exact
  match. Loading is defensive: a benchmark that can't be fetched is skipped (logged),
  not fatal.
- **Few-shot / chain-of-thought.** The paper doesn't specify prompting; I use
  zero-shot with an explicit answer-format instruction. What matters for the claim is
  the **delta** vanilla→finetuned under identical prompting, which this preserves.

---

## 8. Internal-emotion detection (Appendix I)

`internal/` implements the logit-lens method: bucket vocabulary into Ekman's six
emotions, unembed the residual stream at each layer (final RMSNorm + tied output
embedding — the standard logit lens), z-score each emotion-token logit against
per-(layer,token) WildChat baseline statistics (500 samples), average per emotion,
and **residualise against a random-token control** (per-layer linear regression of
the emotion z-score on the control z-score, take the residual) to remove the
globally-correlated drift the paper describes. Outputs the Figure-14 running-average
trajectory (layers 30-40, 400-token window) and Figure-15 layerwise snapshots around
onset.

**Gap-fill — token→emotion classifier.** The paper classifies "the whole Gemma
dictionary" into Ekman emotions (~1,200 tokens) without giving the classifier. I use
curated per-emotion **seed-stem lexicons** matched as prefixes against decoded vocab
tokens (so `frustrat` catches frustrated/-ing/-ion), capped per emotion (~200 each)
to balance buckets and approximate the ~1,200 total. This is transparent and
reproducible; a learned classifier (e.g. embedding similarity) could replace
`emotion_lexicon.py` without touching the detector. **Memory:** only the emotion +
control columns of the unembedding are materialised, so the logit lens is tractable
on long conversations.

---

## 9. Cross-cutting decisions

- **Determinism.** Subject generation is temperature 1 (paper requirement) but
  per-turn/per-sample seeded, so repeated samples differ yet a whole run reproduces.
  Sampling of conditions, WildChat prompts, and dataset draws are all seeded.
- **Thinking disabled** (Appendix B.1) via the OpenRouter `reasoning` control; the
  paper's caveat that Gemini-2.5-Pro / GPT-5.2 may still emit hidden reasoning is
  reproduced as a `ModelSpec.notes` and honoured (we don't pretend to fully disable
  it).
- **Resumability + crash safety.** Every long stage appends JSONL and skips
  completed ids, so a 4,000-rollout job survives interruption.
- **Cost/latency awareness.** `configs/example.yaml` shrinks every budget for a
  smoke test; `EvalBudget` is threaded through `generate_responses` so the ablation's
  100-sample eval reuses the exact same path (no monkeypatching).
- **No silent truncation.** Where a stage can't reach a target (e.g. fewer than 280
  DPO pairs available, a benchmark that won't load, an unparseable judge response),
  it warns explicitly rather than quietly proceeding.

---

## 10. Known limitations of this replication

1. **Unrun.** Nothing has been executed; there is no Python interpreter in the
   authoring environment. Module syntax was reviewed by hand. Expect to fix minor
   integration issues (TRL/PEFT version drift, dataset schema changes, exact Gemma-3
   module/layer names) on first run. These are isolated to the trainer and
   internal-detector modules.
2. **Underspecified numbers approximated**, each flagged above: puzzle set,
   WildChat seed, neutral-rejection randomisation, DPO score/turn stratification,
   benchmark subsets, Ekman token classifier, Petri scaffold, judge/paraphrase
   temperatures, App. E secondary hyperparameters.
3. **Quantitative match not guaranteed.** Reproducing 35%→0.3% requires the real
   models, the real 280-pair distribution, and the real training run. The code aims
   to make those numbers *obtainable* with the right resources, and to make every
   assumption between here and there explicit.
4. **Gemini caveats** (hidden reasoning, OpenRouter provider routing variance) are
   inherent to the closed model and are surfaced, not hidden.
