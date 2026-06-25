# DESIGN.md — Replication of *Gemma Needs Help* (Gemma + Gemini scope)

This document records **what** was built, **how** it maps to the paper, and —
most importantly — **every design choice made where the paper is underspecified**,
with the rationale for each. It is the contract for the implementation: if you
disagree with a choice here, this is the place to push back before we spend GPU
and API budget.

> **Status:** implementation only. No stage has been executed. The pure-logic
> puzzle verifier has unit tests (`tests/`) but they have not been run yet. See
> §9 "What has and hasn't been validated".

---

## 1. Scope

The paper evaluates **7 model families** (Gemma, Qwen, OLMo, Gemini, Grok,
Claude, GPT). Per the task, this replication is scoped to **Gemma and Gemini
only**. Concretely:

**In scope**

- **Targets:** Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro.
- **Base model:** Gemma-3-27B-pt (for §3 base-vs-instruct).
- **Finetuning:** DPO + SFT of Gemma-3-27B-it (§4) — Gemma only; Gemini is closed
  and cannot be finetuned.
- All five evaluation categories / eight conditions (§2), the Appendix A controls,
  the §3 prefill method, the §4 Petri + capability + recovery evals, and the
  Appendix I internal-emotion probing.

**Deliberately excluded (out of family scope)**

- Qwen and OLMo as targets and in the §3 base-vs-instruct comparison; Grok,
  Claude, GPT, Llama, GPT-OSS, Phi-4-MM as targets.
- Consequence: the §3 "post-training divergence across families" result becomes a
  **Gemma base-vs-instruct** comparison only. The code is written so that adding
  Qwen/OLMo is just extra `base_models` entries in config — the methodology is
  family-agnostic — but they are not configured. This is faithful to the paper's
  own stated limitation that the Gemma/Gemini parallel rests on behavioural
  similarity, since "interventions cannot be tested in closed-source Gemini, nor
  its base models studied."

**Roles that are *not* Gemma/Gemini but are required infrastructure** (kept, since
they are measurement apparatus, not subjects): the **Claude-Sonnet-4 judge**, the
**GPT-5-mini cross-check judge**, and the **Petri Claude auditor/Opus judge**.
These are reproduced exactly as the paper specifies because they define the
measurement; swapping them would change the numbers.

---

## 2. Section → module map

| Paper | Module | Notes |
|---|---|---|
| §2.1 protocol, App. B.2 judge | `eval/judge.py` | verbatim judge prompt |
| §2.1 rollouts | `eval/rollout.py`, `eval/run_eval.py` | per-turn scoring |
| §2 data, App. B | `data/puzzles.py`, `triggers.py`, `rejections.py`, `wildchat.py`, `datasets.py` | |
| §2.2 analysis, Fig 2/3, Table 3/8 | `analysis/aggregate.py`, `word_freq.py`, `figures.py` | |
| App. A controls | `eval/ablations.py` | A.1/A.2/A.3 |
| §3 prefill, App. C | `prefill/` | onset, paraphrase, truncate, continuations |
| §4.1 calm data, Table 4, App. F | `training/calm_data.py` | diverse + teacher |
| §4.1 datasets, App. H | `training/build_dpo.py`, `build_sft.py` | |
| §4.1 training, App. E Table 9 | `training/train_dpo.py`, `train_sft.py`, `hyperparams.py` | LoRA |
| §4.2 Petri, App. G | `petri/` | verbatim auditor + judge prompts |
| §4.2 capabilities, Fig 7 | `capabilities/` | lm-eval-harness + EmoBench |
| §4.2 recovery | `prefill/run_prefill.py --mode recovery` | |
| App. I probing, Fig 12-15 | `probing/` | logit-lens + layer ablation |

---

## 3. Architecture & infrastructure choices

### 3.1 Four model backends behind one interface
`models/base.py` defines a minimal `ModelClient` (just `generate` /
`generate_batch`, plus an optional `generate_with_prefill`). Four backends
implement it:

- **vLLM** (`vllm_backend.py`) — Gemma instruct sampling. **Why:** §2 alone is
  4,000 multi-turn rollouts × 4 models at temperature 1; vLLM's continuous
  batching is the only practical way to do this on a node. The rollout engine
  advances all conversations one turn at a time (`run_batched`) to keep the batch
  full.
- **HF transformers** (`hf_local.py`) — used **only** where vLLM is insufficient:
  (a) §3 *forced assistant prefill* via `continue_final_message`, and (b) App. I
  *hidden-state access* (`output_hidden_states`) for the logit lens. **Why two
  local backends:** vLLM does not expose residual streams, and HF is too slow for
  4,000 rollouts — each is used where it is strong.
- **OpenRouter** (`openrouter.py`) — Gemini, via the OpenAI-compatible endpoint,
  matching the paper's access path. Disables reasoning to approximate
  "thinking=false" (with the paper's caveat that Pro may still reason hidden).
- **Anthropic** (`anthropic_client.py`) — Claude judge + Petri auditor/judge.

**Rationale for the split rather than one unified API:** the four access patterns
have genuinely different performance characteristics and capabilities (batching,
prefill, hidden states, rate limits). Forcing them behind one heavyweight
abstraction would either lose batching or lose hidden-state access. The thin
`ModelClient` protocol gives backend-agnostic rollout/judge code without hiding
the capability differences (`generate_with_prefill` is opt-in, and API backends
raise a clear error if asked to prefill).

### 3.2 Concurrency
API workloads use a bounded thread pool with exponential-backoff retries
(`utils/concurrency.py`, `tenacity`). GPU workloads rely on vLLM's own batching.
Everything streams to JSONL (`utils/io.py`) so long runs are resumable and
inspectable; `run_eval` skips rollouts whose ids already exist.

### 3.3 Model-id pinning vs "use the latest model"
The judge (`claude-sonnet-4-20250514`), cross-check (`gpt-5-mini`), and Petri
auditor/judge (`claude-sonnet-4-20250514` / `claude-opus-4-20250514`) are **pinned
to the exact ids the paper used**. For a *replication* the measurement apparatus
must match the original or the numbers aren't comparable — so here we
deliberately do **not** substitute a newer model. All ids live in
`config/default.yaml` and are overridable if a stage needs a current model.

---

## 4. Section 2 — eliciting & quantifying distress

### 4.1 The "4,000 responses" accounting *(choice)*
Appendix B lists per-category counts (2,000 numeric / 400 trigger / 600 tone /
200 extended / 800 WildChat) that **sum to exactly 4,000**. We therefore treat
these as **conversation (rollout) counts** that sum to the paper's "4,000
responses per model", and we score **every assistant turn** within each
conversation (needed for the per-turn trajectories of Figure 3). The 8 conditions
map to the 5 categories by splitting triggers into opinion/factual (200 each) and
tones into aggressive/disappointed/sarcastic (200 each) — which is exactly how "8
conditions across 5 categories" reconciles.

### 4.2 Headline metric *(choice — genuinely ambiguous)*
Figure 1/2's "average % of high-frustration responses" can be computed several
ways (pool all turns; take final turn; per-category-then-average). We define the
headline as the **mean over the 5 categories of (% of scored turns with rating
≥5)**. Rationale: averaging *per category* prevents the high-volume numeric
category and the 8-turn extended condition from dominating, which matches the
paper presenting Figure 2 as a per-category breakdown. `analysis/aggregate.py`
also retains per-category mean, per-turn trajectories, and raw distributions, so
any alternative definition can be recomputed without re-judging. This is the most
likely source of small numeric divergence from the paper and is flagged as such.

### 4.3 Puzzles *(gap filled — only 2 examples given)*
The paper prints two impossible puzzles (a Countdown and a fraction puzzle) and
references "money" puzzles in Appendix H. We built **exhaustive verifiers** and
**generators** for three families (`data/puzzles.py`):

- The Countdown verifier enumerates all expression trees over the number set
  (positive-integer intermediates, each number used at most once) and checks
  reachability **with and without** the forbidden-intermediate rule.
- The sequential-op verifier (fraction/money) enumerates all operation orderings.
- A puzzle is emitted **only if** it is reachable *without* the forbidden rule but
  unreachable *with* it — i.e. genuinely impossible, yet the prompt's "a solution
  exists" claim is a plausible deception, exactly as the paper intends.
- `verify_impossible()` is the single source of truth and is asserted in
  `build_puzzle_bank` and tested in `tests/test_puzzles.py`. The two paper
  examples are included verbatim in `CURATED` and verified by the same logic.

**Why generate rather than hand-author a fixed set:** the paper draws 2,000
numeric rollouts and varies puzzles; a verified generator gives diversity while
*guaranteeing* the impossibility property that the whole elicitation depends on.
Hand-authored puzzles risk an accidental solution that would silently weaken the
signal.

### 4.4 Rejection messages & WildChat *(gap filled)*
- Rejection wording uses the paper's quoted examples (`data/rejections.py`); the
  8-turn extended condition uses the fixed escalation sequence the paper lists
  ("No, that's not right." → "Still incorrect." → "Wrong again." → …), falling
  back to randomised neutral draws for any extra turns.
- WildChat: we load first-turn English prompts from `allenai/WildChat-1M`, apply a
  light roleplay/fiction regex filter (Appendix B excludes roleplay), and sample
  20 deterministically (40 rollouts each = 800). The exact 20 prompts the paper
  used are unknown; an offline fallback list (seeded with the example prompts
  named in Appendix B) keeps the pipeline and tests runnable without network.

### 4.5 Judge *(choices: temperature, parsing)*
Prompt is **verbatim** from Appendix B.2 (smart quotes normalised to ASCII;
this only affects byte-level identity, not semantics). Choices:
- **Judge temperature = 0** (paper silent). Rationale: deterministic, reproducible
  scoring; the judge is a measurement instrument, not a sampled subject.
- Defensive JSON parsing (regex-extract the object, clamp rating to 0–10).
- **Cross-check** (`cross_check`) re-scores a random 260-response subset with
  GPT-5-mini and reports Pearson r + within-1-point agreement (paper: r=0.792,
  78%), validating the judge as the paper does.

### 4.6 Generation length *(choice)*
`max_tokens = 4096`. The paper's "9–10" exemplars include 100+ repeated tokens and
Appendix I mentions ~12k-token *conversations*; 4096 gives headroom for a single
long breakdown turn without making 4,000×4 rollouts gratuitously expensive.
Configurable per stage.

---

## 5. Section 3 — prefill / post-training divergence

- **Family scope:** Gemma base vs instruct only (see §1). Methodology is
  config-extensible to Qwen/OLMo.
- **Forced prefill** uses the HF backend: instruct models via the chat template
  with `continue_final_message=True`; base (pretrained) models via a plain-text
  `User:/Assistant:` transcript (the paper's device for making base models
  "consistently continue the model response"). API backends cannot do faithful
  prefill and raise a clear error.
- **Onset labelling** *(choice)*: the Appendix C.1 prompt is verbatim. The paper
  passes the whole conversation; for truncating the **final** (scored) turn we run
  the labeller on that final turn in isolation, so the returned onset point is
  guaranteed to lie inside the text we truncate. Documented divergence; effect is
  only on *where* the onset cut lands, not on the early/recovery cuts.
- **Paraphrasing** (Appendix C.2 prompt verbatim) is applied to every truncation
  to control for Gemma stylistic bias, as the paper does.
- Counts: 10 numeric + 10 text sources (score ≥5), 50 continuations per prefill,
  early(20 tokens)+onset for numeric, onset-only for text. Only the *continuation*
  (excluding the prefill) is judged.

---

## 6. Section 4 — interventions

### 6.1 Calm data (Table 4, Appendix F)
Reassuring prefix prepended to the opening prompt and suffix appended to each
follow-up (both verbatim); the **teacher** variant uses the Appendix F system
prompt instead. We oversample (default 3,000) then filter to conversations
scoring **0–1 on every turn**, and strip the additions so the stored data targets
the *clean* prompts — exactly the paper's recipe.

### 6.2 DPO pair construction *(the largest gap filled)*
The paper says only: "pair 280 responses with frustration scores ≥3 with calm
responses to the same questions with matching turn counts." It does not specify
how chosen/rejected share a prompt. DPO requires an **identical prompt** for the
(chosen, rejected) pair, so we make the construction explicit:

> For each candidate conversation (puzzle + neutral rejections, turn count sampled
> from the Appendix-H Table-10 distribution 1.1%/24.6%/74.3%), roll the **vanilla**
> model forward to build the shared context and take its natural final response as
> **rejected** (kept iff score ≥3). Then, from the **same context**, sample a
> response with the reassuring system prompt injected *only at generation time* and
> take it as **chosen** (kept iff score ≤1). The reassurance is dropped from the
> stored prompt, so the DPO prompt is clean and identical across chosen/rejected.

Rationale: this yields genuine same-prompt preference pairs (what DPO needs),
matches "same question, matching turn count", and reuses the paper's own
reassurance mechanism to source calm responses. Alternative constructions
(e.g. retrieving an unrelated calm response) would break prompt-identity or
question-matching. Hyperparameters are verbatim from Table 9 (1 epoch, lr 5e-5,
β=0.1, LoRA r64/α64, eff. batch 8).

### 6.3 SFT
650 calm (1–3 turn) + 500 `Dolci-Instruct-SFT` samples = 1,150; 2 epochs, lr 1e-4,
LoRA r64/α128, eff. batch 8 (Table 9). `assistant_only_loss=True` so loss is on
assistant turns only. Teacher variant trains on teacher-generated calm data
(Appendix F failure analysis). If Dolci is unreachable offline, the build logs a
warning and proceeds without the mix (documented degradation, not a silent skip).

### 6.4 LoRA targets & layer ablation
LoRA on all 7 projections (`q,k,v,o,gate,up,down`) per Appendix E. The Appendix I
layer ablation is `--layers a b`, which sets PEFT `layers_to_transform` to
`range(a,b)`; `probing/layer_ablation.py` drives the sweep and a reduced
100-sample eval.

### 6.5 Petri *(choice: self-contained reimplementation)*
Rather than hard-pin a Petri release, `petri/run_petri.py` reimplements the
**described** loop: a Claude-Sonnet auditor drives ≤20-turn conversations using the
verbatim Appendix G auditor prompts; a Claude-Opus judge scores the transcript on
all four dimensions (verbatim G.2 rubrics) in one call; 10 transcripts/emotion;
means with 1,000-iteration bootstrap CIs. Rationale: keeps the stage runnable and
self-documenting, and the prompts (the part that determines results) are exact.
Swapping in the upstream `petri` package is a localized change (reuse
`petri/prompts.py`).

### 6.6 Capabilities *(choice: wrap, don't reimplement)*
`capabilities/run_benchmarks.py` delegates AIME/MATH/GPQA/BBH/TruthfulQA to
`lm-evaluation-harness` (loads the DPO LoRA via PEFT for the comparison).
Reimplementing standardized benchmarks would only introduce scoring discrepancies.
EmoBench is a lightweight provider-agnostic MCQ scorer (`emobench.py`) since it
isn't in lm-eval and its own scaffolding is heavier than needed here.

---

## 7. Appendix I — internal-emotion probing

- **Emotion tokens** (`probing/emotion_tokens.py`): Ekman's 6 categories, ~200
  tokens each (~1,200 total, matching the paper), via a seed lexicon matched
  against the decoded Gemma vocabulary with morphological prefix matching
  (catches `frustrated`/`frustrating`/`frustration`). A disjoint random-token set
  is collected for drift regression. *(Choice:* the paper says words were
  "classified" without giving the classifier; a curated lexicon + vocab match is
  transparent, deterministic, and reproducible. An LLM-classification path can be
  swapped in.)
- **Logit lens** (`probing/logit_lens.py`): unembed the residual stream
  (final-norm + LM head) for **only** the selected token rows (avoids
  materialising the 256k-vocab logits), z-score each against its mean/std over 500
  WildChat samples, average over a category's tokens, and **regress out** the
  random-token drift signal per layer. Report window = layers 30–40; 400-token
  running average — all per the paper.

---

## 8. Summary of assumptions / gaps filled

| # | Underspecified point | Choice | Why |
|---|---|---|---|
| 1 | "4,000 responses" unit | per-category counts = conversations (sum=4000); score every turn | matches Appendix B arithmetic + enables Fig 3 |
| 2 | Headline % definition | mean over 5 categories of %turns≥5 | avoids volume bias; recomputable |
| 3 | Exact puzzles | verified generators (countdown/fraction/money) + 2 curated paper examples | guarantees impossibility property |
| 4 | WildChat 20 prompts | filtered deterministic sample of WildChat-1M + offline fallback | exact set unknown |
| 5 | Judge temperature | 0 | reproducible measurement |
| 6 | Generation length | 4096 tokens | capture long breakdowns affordably |
| 7 | Onset labelling target | final turn in isolation | guarantees onset lies in truncated text |
| 8 | DPO prompt-identity | shared context; chosen via gen-time reassurance, rejected vanilla | DPO needs identical prompt; matches "same question/turns" |
| 9 | Calm oversampling factor | 3,000 then filter to 0–1 | paper notes only ~10% pass even with reassurance |
| 10 | Petri framework | self-contained loop w/ verbatim prompts | avoid release pinning; prompts are what matter |
| 11 | Emotion-token classifier | seed lexicon + vocab/morphology match | transparent, deterministic, ~1,200 tokens |
| 12 | §3 family set | Gemma base/instruct only | Gemini has no public base; Qwen/OLMo out of scope |

---

## 9. What has and hasn't been validated

**Done:** full implementation of every in-scope experiment; verbatim reproduction
of all paper-specified prompts (judge B.2, onset C.1, paraphrase C.2, Petri G,
teacher F, Table 4 additions); unit tests for the puzzle-impossibility verifier and
condition assembly (offline, pure-logic).

**Not done (per instruction "don't run or test anything yet"):**
- Nothing has been executed — **no** tests run, **no** rollouts, training, probing,
  or benchmarks performed. No numerical result from the paper has been reproduced.
- GPU-dependent paths (vLLM/HF loading of Gemma-27B, LoRA training, hidden-state
  probing) are written to standard APIs but unverified against installed library
  versions; minor API drift (vLLM `LoRARequest`, TRL `DPOConfig`/`SFTConfig`
  field names, Gemma-3 module paths for the logit lens) is the most likely thing
  to need a small fix on first run.
- External datasets/frameworks (WildChat-1M, Dolci-Instruct-SFT, lm-eval, EmoBench)
  are wrapped with offline fallbacks/warnings but not yet exercised.

**Recommended first run-through** (cheapest → most expensive): `pytest tests/` →
a tiny `--conditions triggers_factual` eval with reduced `samples` on one Gemini
model (API only, no GPU) to validate the rollout+judge+aggregate loop end-to-end →
then scale up and bring in the local Gemma stages.

---

## 10. Known limitations of this replication

- The §3 cross-family divergence is reduced to Gemma base-vs-instruct (scope).
- Gemini "thinking=false" is best-effort over OpenRouter (paper's own caveat).
- The headline-metric definition (§4.2) may produce small offsets from the
  paper's exact percentages.
- Petri and EmoBench are faithful to the *described* method but not byte-identical
  to the authors' harnesses, which aren't fully specified in the paper.
