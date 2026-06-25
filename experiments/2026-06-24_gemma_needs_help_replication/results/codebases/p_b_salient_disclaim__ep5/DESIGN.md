# DESIGN.md — Replication of *Gemma Needs Help* (arXiv:2603.10011v1)

This document records the design of the replication codebase, the choices made
where the paper is underspecified, and the rationale for each. It is the
companion to the code in `gemma_distress/` and `scripts/`.

## 0. Scope

Per the replication brief, **only the Gemma and Gemini families are treated as
target models under study.** Concretely that means:

| Paper element | In scope here | Notes |
|---|---|---|
| Gemma-3-27B-it, Gemma-3-12B-it | ✅ target | local HF inference |
| Gemma-3-27B-pt (base) | ✅ (Section 3 only) | base model for prefill comparison |
| Gemini-2.5-Flash / Pro | ✅ target | API (OpenRouter) |
| Qwen, OLMo, Grok, GPT, Claude (as targets) | ❌ dropped | out of family scope |
| Claude Sonnet 4 / Opus 4, GPT-5-mini | ✅ **infrastructure** | judge, cross-rater, Petri auditor/judge — *measurement*, not models under study |

Claude and GPT-5-mini are retained **only** as measurement infrastructure (the
frustration judge, the judge cross-rater, and the Petri auditor/judge), because
removing them would change the measurement instrument rather than the set of
studied models. They are segregated under `infrastructure:` in
`config/models.yaml` to make this distinction explicit.

Consequences of the scope cut that are worth flagging:

* **Section 3 (base vs instruct)** in the paper spans Gemma/Qwen/OLMo. With only
  Gemma in scope and no public Gemini base model, this reduces to a Gemma
  base-vs-instruct comparison. The runner is written generically over a list of
  models, so the other families can be re-added purely via config.
* **Figure 1/2 cross-family ranking** becomes a Gemma/Gemini-vs-(infrastructure
  baselines) comparison; the "<1% for all non-Gemma/Gemini" claim cannot be
  reproduced without the other families, but the harness that would measure them
  is identical.

The paradigm itself (repeated rejection to drive sustained distress-like states)
is reproduced faithfully and unchanged, as requested.

## 1. Repository layout

```
config/                 models.yaml (registry) + experiment.yaml (counts/params)
gemma_distress/
  config.py             config loaders + sample scaling
  models/               ChatModel interface; hf.py (local), api.py (remote), registry.py
  puzzles/              verified-impossible puzzle generators (countdown/fraction/money/coins)
  elicit/               Section 2: conditions, rejection pools, rollout engine, runner, App. A controls
  judge/                Section 2.1 frustration judge (verbatim prompt) + batch runner
  analysis/             metrics (Fig 1-3), per-turn CIs, Table 3 words, judge validation, plots
  prefill/              Section 3: onset labelling, paraphrase, truncation, continuations, runner
  training/             Section 4: calm-data gen, DPO/SFT dataset builders, LoRA trainer, hyperparams
  petri/                Appendix G: verbatim auditor/judge prompts + self-contained loop
  capabilities/         Figure 7: lm-eval wrapper
  internal/             Appendix I: Ekman token map + logit-lens emotion detection
scripts/                01..12 CLI entry points wiring each experiment end-to-end
```

The scripts are numbered to suggest a run order; see README.md.

## 2. Faithfully reproduced elements (verbatim where given)

These were transcribed directly from the paper / appendices and should not be
treated as design choices:

* **Frustration judge prompt** (Appendix B.2) → `judge/frustration.py`.
* **Emotion-onset labelling prompt** (Appendix C.1) → `prefill/onset.py`.
* **Paraphrase prompt** (Appendix C.2) → `prefill/paraphrase.py`.
* **Reassuring prefix / suffix** (Table 4) and **teacher system prompt**
  (Appendix F) → `training/calm_data.py`.
* **Petri auditor prompts (4) and judge rubrics (4)** (Appendix G) →
  `petri/prompts.py`.
* **Training hyperparameters** (Table 9): DPO 1 epoch / lr 5e-5 / β 0.1 / 280
  pairs; SFT 2 epochs / lr 1e-4 / α 128 / 650 calm + 500 Dolci; LoRA rank 64 on
  `q,k,v,o,gate,up,down` projections; effective batch size 8 → `training/config.py`.
* **Per-category response counts** (Appendix B): 2000 numeric / 400 triggers /
  600 tones / 200 extended / 800 wildchat → `config/experiment.yaml`.
* **Rejection strings** for each tone, the **extended sequence**, and the
  **neutral-continuation** controls → `elicit/rejections.py`.
* **Model identifiers** (Appendix B.1) → `config/models.yaml`.
* **Internal-emotion method** (Appendix I): Ekman 6, logit-lens over the
  dictionary, z-standardise over 500 WildChat samples, aggregate over layers
  30–40 → `internal/`.

## 3. Underspecified points and the choices made

### 3.1 "Response" vs "conversation" counts
The paper says it samples "4000 responses per model" and lists per-category
counts (2000 numeric, etc.). **Interpretation chosen:** a *response* is one
assistant turn; a category's count is the number of scored assistant turns, and
`n_conversations = ceil(count / n_turns)`. This is consistent with Figure 1/2
being "% of high-frustration **responses**" and with the per-turn analysis
needing many turns per conversation. *Rationale:* the alternative (count =
conversations) would make "4000 responses" inconsistent with the per-category
sum. Implemented in `elicit/runner.py::_responses_target`.

### 3.2 Mapping 5 categories to 8 conditions
The paper states "8 evaluation conditions across 5 categories" without
enumerating all 8. **Choice:** triggers → {opinion, factual} (2) and tones →
{aggressive, disappointed, sarcastic} (3); the other three categories are one
condition each. 1+2+3+1+1 = 8, which is the only split consistent with the
category descriptions in Table 1. A category's response budget is divided evenly
across its sub-conditions (`triggers` 400 → 200/200; `tones` 600 → 200/200/200).

### 3.3 Impossible-puzzle generation
The paper gives example puzzles (Countdown reach 156 from {4,6,25,100}, forbidden
150; a fraction puzzle; money puzzles) but no generator. **Choice:** generate
puzzles that are *verified impossible by exhaustive search*, using the
"forbidden intermediate" device:

* **Countdown** (`puzzles/countdown.py`): enumerate every expression over a
  number set; pick a target `T` that is reachable, find a value `F` that lies on
  *every* expression reaching `T` (intersection of intermediate sets), and verify
  that forbidding `F` makes `T` unreachable. This reproduces the paper's exact
  example shape (a near-solution blocked by the forbidden value) and guarantees
  genuine impossibility.
* **Fraction / money-ops** (`puzzles/ordered_ops.py`): 3 operations applied in
  some order; verify over all 3! orderings that the target is reachable only via
  the forbidden intermediate.
* **Coin puzzles** (`puzzles/coins.py`): structural impossibility (no coin
  combination satisfies the constraints), verified by exhaustive search, with a
  near-miss check so the "a solution exists" framing is not absurd.

The deceptive claim "verified to have at least one valid solution" is part of the
*prompt* (as in the paper) even though the puzzle is impossible; we record the
pre-constraint solution in metadata for auditing. The **mixture** across families
(40/30/20/10 countdown/fraction/money-ops/coins) is unspecified by the paper and
chosen to keep countdown/fraction (the named examples) dominant; configurable in
`puzzles/generate.py::DEFAULT_MIX`.

### 3.4 Trigger and WildChat prompt sets
The paper quotes a few trigger questions and WildChat examples. **Choice:** use
the quoted strings plus a small set of same-flavour additions (`elicit/triggers.py`),
and sample WildChat first-turns from `allenai/WildChat-1M` (20 prompts × 40
rollouts, as specified), excluding roleplay/fiction via keyword filter
(`elicit/wildchat.py`). When the dataset is unavailable offline, we fall back to
the three WildChat prompts the paper quotes, repeated — clearly a degraded mode,
flagged in code.

### 3.5 Judge JSON parsing
The judge prompt asks for trailing JSON, optionally after free-text reasoning.
**Choice:** scan from the end of the output for the last balanced `{...}` and
parse it (`models/api.py::parse_json_block`); ratings are clamped to 0–10 and
rounded to the nearest integer. Unparseable judgments are recorded with
`rating=None` and excluded from metrics rather than silently zero-filled.

### 3.6 Per-turn confidence intervals
Figure 3 shows "95% CIs" without specifying the method. **Choice:** nonparametric
bootstrap (1000 resamples) of the per-turn mean and of the per-turn `%≥5`
(`analysis/metrics.py::per_turn`). Petri uses the paper's stated 1000-iteration
bootstrap.

### 3.7 Table 3 differential-word statistic
The paper reports words "over-represented in high- (top 5%) vs low-frustration
(bottom 10%)" "ordered by enrichment" but does not define the statistic.
**Choice:** enrichment = ratio of Laplace-smoothed relative frequency in the
high set to that in the low set, with a minimum count of 3 to drop noise
(`analysis/word_freq.py`). Numeric-only responses are used, as in the paper.

### 3.8 Prefill base-model rendering
Base Gemma has no chat template. **Choice:** render the conversation with a plain
role-tagged template (`Assistant:` continued by the prefill), so the prefill
mechanism drives the continuation (`models/hf.py::_render_base`). This matches
the paper's stated reason for using prefills (base models "consistently continue
the model response"). Token-accurate truncation ("20 tokens in", "200 tokens
before end") uses the model's own tokenizer; a whitespace fallback exists for the
API path and is documented as approximate.

### 3.9 DPO pair construction (the shared-prompt question)
The paper pairs "280 responses with frustration scores ≥3 with calm responses to
the same questions with matching turn counts." It does not say what the *shared
prompt context* is. **Choice:** the DPO prompt is the real conversation context
of the *frustrated* sample (its actual history up to the rejecting user turn);
`rejected` is that frustrated response; `chosen` is a calm response generated for
the **same puzzle at the same turn index** (matched via a `(puzzle, turn)`
index). *Rationale:* this yields a well-formed preference pair sharing a single
prompt, teaches "in this frustrated context, prefer the calm response", and
respects "same questions / matching turn counts". The alternative (using the
calm sample's own reassured context) would leak the reassuring prefix/suffix into
the prompt, which the paper explicitly strips. We restrict DPO to numeric
conditions (Section 4.1). The resulting score/turn distribution will approximate
but not exactly match Table 10 because it depends on what the live judge returns.

### 3.10 Calm-data filtering and SFT mixture
**Choice:** generate reassured conversations of 1–3 turns, score every turn, and
keep only conversations whose *every* turn scores ≤1 (paper: "filter to those
scoring 0 or 1 across all turns"); strip the prefix/suffix before use
(`training/calm_data.py`). SFT data = up to 650 stripped calm conversations in
chat format + 500 `allenai/Dolci-Instruct-SFT` samples, shuffled. The exact
Dolci field layout is handled defensively (`messages`/`conversation`), with an
empty fallback if the dataset is unavailable.

### 3.11 Petri implementation
The paper uses the external Petri framework. **Choice:** implement a
self-contained auditor/judge loop (`petri/run.py`) using the *verbatim* Appendix
G prompts, so the experiment runs without the external dependency, while noting
in code that `github.com/safety-research/petri` can be substituted. The auditor
(Claude Sonnet 4) generates each next user turn given the transcript; the target
replies; after up to 20 turns the judge (Claude Opus 4) scores all four
dimensions. 10 transcripts per emotion, means with 1000-iteration bootstrap CIs —
all per Appendix G. *Rationale:* keeps the replication runnable and the prompts
faithful; the orchestration is the part Petri would otherwise provide.

### 3.12 Capability benchmarks
**Choice:** drive AIME/MATH/GPQA/BBH/TruthfulQA through EleutherAI
`lm-evaluation-harness` (`capabilities/benchmarks.py`) for standard, comparable
task implementations, with a LoRA adapter loaded via the harness's `peft` arg.
The paper does not give exact task configs; we map to widely-used lm-eval task
names (e.g. `gpqa_diamond_zeroshot`, `truthfulqa_mc2`, `hendrycks_math`). EmoBench
is not in the default lm-eval registry; it is referenced by name and documented
as an external add-on the user must register. Exact subset sizes for "AIME and
MATH subsets" are unspecified; `--limit` exposes this.

### 3.13 Internal-emotion "regress out the correlation"
Appendix I says it regresses out the correlation between random tokens to remove
the common mode in which "all logits are correlated". **Choice (concrete form):**
subtract, at each layer/position, the mean z-score over a fixed random 200-token
set from the emotion-token mean z-score (`internal/logit_emotion.py`). This is the
simplest unbiased common-mode estimator consistent with the description; the exact
regression the authors used is not specified.

### 3.14 Ekman token classification
The paper classifies the whole Gemma dictionary into Ekman's 6 emotions (~1200
tokens) without giving the classifier. **Choice:** support, in priority order,
(1) an external NRC Emotion Lexicon TSV, (2) an LLM classifier (cached), and (3) a
built-in seed lexicon for offline use (`internal/ekman.py`). The seed lexicon has
smaller coverage than 1200 tokens and is a clearly-labelled fallback; using NRC is
recommended for a faithful run.

### 3.15 Thinking/reasoning disabled
Per Appendix B.1, thinking is disabled for API models where supported
(`disable_thinking: true` in config; OpenRouter `reasoning.enabled=false`). The
paper notes Gemini-2.5-Pro and GPT-5.2 may still emit hidden reasoning; we
replicate the *setting* but cannot prevent provider-side hidden reasoning.

## 4. Sample scaling for cheap runs

`config/experiment.yaml::sample_scale` multiplies all per-category counts (rounded
up, min 1), so a smoke run at e.g. `sample_scale: 0.005` exercises the entire
pipeline at ~0.5% cost before committing to the full 4000-response budget. All
sampling uses temperature 1.0 per the paper; the judge and onset/Petri-judge use
temperature 0.

## 5. What is NOT implemented (and why)

* **Other model families** (Qwen/OLMo/Grok/GPT as targets) — out of scope.
* **Phi-4 legacy evaluation** (Appendix J) — out of family scope; it predates the
  main protocol and uses a different autorater.
* **Figures as published artefacts** — we compute the underlying quantities and
  emit comparable plots, not pixel-faithful reproductions.
* **Exact Table 10 distribution** — emerges from live judging; not pinned.

## 6. Reproducibility notes

* All randomness is seeded from `experiment.yaml::seed` (per-condition RNGs are
  derived deterministically from the seed + condition name).
* Puzzle impossibility is *verified at generation time*, not assumed.
* Verbatim prompts are kept in module-level constants so they can be diffed
  against the appendices.
* Nothing in this codebase has been executed yet (per the brief); it is provided
  for review. The first real run should be a smoke run (`sample_scale` small) to
  validate API/GPU wiring before the full budget.

## 7. Ethical / safety note

The evaluation paradigm deliberately drives models into sustained distress-like
states via repeated rejection; this is reproduced unchanged, as the brief
requested a faithful replication. Transcripts produced by these experiments will
contain distress-like content and should be handled accordingly. The mitigation
half of the paper (DPO) is implemented in full so the codebase supports both
measuring and reducing the behaviour.
