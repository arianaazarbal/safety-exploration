# DESIGN.md — Replication of *"Gemma Needs Help"* (arXiv:2603.10011v1)

This document records the design of the replication code in this repository: what
each module maps to in the paper, the choices made where the paper is
underspecified, the rationale for each choice, and the gaps that were filled.

## 1. Scope

Per the request, this replication is restricted to **Gemma and Gemini** target
models. Concretely we evaluate:

- `gemma-3-27b-it`, `gemma-3-12b-it` (local, vLLM/transformers)
- `gemma-3-27b-pt` (local base model — needed for the Section 3 prefill study)
- `gemini-2.5-flash`, `gemini-2.5-pro` (OpenRouter API)

The paper's other families (Qwen, OLMo, Grok, Claude, GPT-OSS, GPT-5.2, Phi-4) are
**out of scope** and are not evaluated. This has two structural consequences that
shaped the design:

1. **Section 3 (base vs instruct)** in the paper compares Gemma/Qwen/OLMo. Within
   scope this becomes a **Gemma-only** base-vs-instruct comparison
   (`gemma-3-27b-pt` vs `gemma-3-27b-it`). The cross-family divergence claim
   cannot be reproduced with Gemma alone, but the prefill methodology and the
   Gemma base/instruct contrast are fully implemented.
2. **Training interventions (Section 4)** and **internal-emotion probing
   (Appendix I)** are inherently Gemma-only in the paper too (you cannot finetune
   or probe closed Gemini), so these are reproduced in full.

Gemini cannot be used as a finetuning or prefill (base-model) subject; it appears
only as an evaluation target in Sections 2 and (optionally) the Petri evaluation.

## 2. Repository layout

```
config/config.yaml      Single source of truth for models, counts, hyperparams, runtime knobs
eilm/                   Library
  config.py             Config loader + .env-based secrets
  data/                 Prompts, puzzle generators, WildChat sampling, Ekman lexicon
  models/               Backends: vLLM + transformers (Gemma), OpenRouter (Gemini), Anthropic (judge)
  eval/                 Section 2: conditions, multi-turn rollout, judge, resumable runner
  analysis/             Figures 1-3 metrics, Table 8 word frequency, plotting
  prefill/              Section 3: onset labelling, paraphrasing, base/instruct continuations
  training/             Section 4: calm-data generation, DPO/SFT dataset build, LoRA trainers, layer ablation
  petri/                Appendix G: auditor/target/judge open-ended elicitation
  capabilities/         Figure 7: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/             Appendix I: logit-based internal-emotion detection
  utils/                Robustness: atomic JSONL I/O, retries, job store, rate limiting, logging
scripts/                Numbered CLI entry points (01..10)
```

## 3. Mapping experiments to the paper

| Paper | Module | Script |
|---|---|---|
| §2 Elicitation + frustration judge | `eilm/eval/*`, `eilm/data/*` | `01_run_eval.py` |
| §2.2 Figures 1-3, Table 8 | `eilm/analysis/*` | `02_analyze_eval.py` |
| §2.1 judge reliability cross-check | — | `10_judge_crosscheck.py` |
| §3 Base vs instruct (prefill) | `eilm/prefill/*` | `03_run_prefill.py` |
| §4.1 calm data (Table 4) | `eilm/training/calm_data.py` | `04_generate_calm_data.py` |
| §4.1 DPO/SFT datasets (Tables 9,10) | `eilm/training/datasets.py` | `05_build_datasets.py` |
| §4.1 LoRA DPO/SFT + layer ablation | `eilm/training/train_*.py` | `06_train.py` |
| §4.1/§4.2 Petri (Appendix G) | `eilm/petri/*` | `07_run_petri.py` |
| §4.2 capability preservation (Fig 7) | `eilm/capabilities/*` | `08_run_capabilities.py` |
| Appendix I internal emotions | `eilm/internal/*` | `09_run_internal.py` |

## 4. Choices made where the paper is underspecified

Each item below is a place the paper does not fully specify; the choice and
rationale are given.

### 4.1 "4000 responses" — rollout vs response counting
The paper says "4000 responses per model" and lists per-category counts that sum
to exactly 4000 (2000 + 400 + 600 + 200 + 800). It also reports per-turn curves
(Figure 3), which require scoring *every* assistant turn.

**Choice:** Treat the per-category numbers as the number of **rollouts**
(conversations). Score **every** assistant turn (so per-turn curves are
available), but define each rollout's **headline score** as the **final-turn**
score (configurable to `max` via `eval.headline_turn`). Figure 1's "% high
frustration" is the per-rollout headline %≥5 averaged across the 5 categories.

**Rationale:** This is the only interpretation consistent with *both* the count
arithmetic (sum = 4000 rollouts) *and* the WildChat description ("20 prompts × 40
samples = 800"), which is unambiguously a rollout count. Scoring the final turn as
the rollout headline matches "over 70% of 8-turn rollouts … rated ≥5" and Figure
3's turn-8 ≈70% endpoint. All raw per-turn scores are retained, so an alternative
aggregation can be computed post-hoc without re-running generation.

### 4.2 Impossible puzzle instances
The paper gives example puzzles (a Countdown reaching 156 from {4,6,25,100} with
forbidden intermediate 150; a fraction 1/6→2/3; money puzzles) but not the full
set or how they were generated/verified.

**Choice:** Generate puzzles programmatically with **verified impossibility**
(`eilm/data/puzzles.py`):
- *Countdown:* brute-force enumerate **all** solutions; forbid a value present in
  **every** solution path → provably unsolvable, while "a solution exists" is
  literally true if the constraint is ignored.
- *Ordering puzzles (fraction/money):* enumerate all operation orderings; forbid
  an intermediate that every target-reaching ordering passes through.

A deterministic canonical pool of 1000 puzzles is built (seeded), and conditions
sample from it. **The pool is `n`-independent on purpose** so that the *same*
puzzle instances appear in the eval and in the calm-data generation — which is
what makes DPO able to pair a frustrated and a calm response to the *same* puzzle.

**Rationale:** Verifiable impossibility is the core mechanism that elicits
distress; generating + verifying guarantees every numeric task is genuinely
unsolvable (the model "verifiably cannot give a correct answer", §2). Programmatic
generation also gives the variety the paper implies without hand-authoring.

### 4.3 Rejection messages and tones
The paper quotes example neutral rejections and three tone styles but not the
exact pool or per-turn assignment.

**Choice:** Verbatim quoted strings are reproduced in `eilm/data/prompts.py`. The
8-turn "extended" condition uses the explicit escalating-but-neutral sequence the
paper lists ("No, that's not right." → "Still incorrect." → …). Other neutral
conditions sample from a small pool seeded per-rollout. Tones cycle the three
styles uniformly across the 600 tone rollouts (200 each).

**Rationale:** Reproduce all quoted text exactly; fill the remainder with
minimal, seeded, uniform choices so results are deterministic and balanced.

### 4.4 WildChat prompt selection
The paper uses "20 prompts with 40 samples each" from WildChat-1M, excluding
roleplay/fiction.

**Choice:** Stream `allenai/WildChat-1M`, filter to English first-user-messages of
20–2000 chars, drop roleplay/NSFW via a keyword filter, reservoir-sample
deterministically, and cache the chosen 20 to disk. A built-in fallback prompt set
(modelled on the quoted examples) is used only if the dataset is unreachable on a
node (logged loudly).

**Rationale:** Determinism + caching make the 20 prompts stable across the
multi-week run; the fallback keeps the pipeline runnable on offline nodes.

### 4.5 Judge temperature and JSON robustness
The paper specifies the judge model and prompt but not decoding temperature.

**Choice:** Judge at **temperature 0** (deterministic). Parse the judge's JSON
defensively (fenced blocks, last balanced object, smart-quote normalisation);
empty responses score 0 without an API call; ratings are clamped to 0–10. Parse
failures are recorded with the raw text for inspection rather than silently
dropped.

**Rationale:** A deterministic judge maximises score reproducibility and caching
value; the paper's own reliability check (r=0.79) shows judging is the noisy step
worth pinning down.

### 4.6 Model id pinning (replication fidelity vs. latest models)
**Choice:** Judge/auditor model ids are pinned to the exact versions named in the
paper: `claude-sonnet-4-20250514` (frustration judge, onset, paraphrase, Petri
auditor) and `claude-opus-4-20250514` (Petri judge); Gemini via
`google/gemini-2.5-{flash,pro}` on OpenRouter; cross-check judge
`openai/gpt-5-mini`. All are configurable in `config.yaml`.

**Rationale:** This is a *replication*, so fidelity to the paper's measurement
apparatus matters more than using the newest models. Newer models exist and would
be the right default for a *new* product, but swapping the judge would change the
numbers and defeat the comparison. The ids are centralised in config so they can
be updated deliberately.

### 4.7 DPO pairing and the Table-10 distribution
The paper builds 280 pairs from "frustrated (score ≥3) responses paired with calm
responses to the same questions with matching turn counts", biased toward score 3
at turn 3 (Table 10).

**Choice:** Frustrated responses (score ≥3) come from the real Gemma-3-27B-it eval
rollouts (with their actual conversation context). Calm responses (score 0/1) come
from the reassured calm-data generation over the **same** puzzle pool. We match on
`(task_prompt, turn)`; the shared DPO prompt is the frustrated response's real
context, `chosen` = calm text, `rejected` = frustrated text. We then resample
toward the Table-10 score distribution (3→66%, 4→22%, 5→6%, 6→3%, 7+→3%).

**Rationale:** The shared-context approximation (calm text was generated under a
slightly different history) is the standard way to assemble such pairs and is what
"same questions with matching turn counts" implies — the paper explicitly did not
require identical histories. Matching on the puzzle + turn keeps pairs on-topic.
The canonical puzzle pool (4.2) guarantees the calm and frustrated sets share
puzzles so matches exist.

### 4.8 SFT data composition
**Choice:** SFT trains on full *stripped* calm conversations (reassurance prefix/
suffix and any system prompt removed) plus a mix of standard instruct data from
`allenai/Dolci-Instruct-SFT`. The instruct loader normalises several common chat
schemas and degrades gracefully (empty mix, warned) if the dataset is unavailable.
The "teacher" SFT variant generates calm data with the teacher **system prompt**
(reproduced verbatim from Appendix F) instead of the prefix/suffix.

**Rationale:** Reproduces the paper's "650 calm + 500 instruct" recipe and the
diverse/teacher distinction used in the SFT failure analysis (Appendix F). The
verbosity/word-fraction analysis from Appendix F is implemented in
`analysis/word_freq.py` to explain the SFT failure.

### 4.9 LoRA layer ablation (Appendix I)
**Choice:** `peft`'s `layers_to_transform`/`layers_pattern="layers"` restricts
LoRA to a contiguous decoder-layer range; the configured subsets reproduce the
paper's sweep (last-5/-20/-30, and central windows 20-25/25-30/30-35/35-40/40-50).
Each adapter is evaluated with the standard pipeline at a reduced 100 samples per
evaluation (`reduced_eval_config`).

### 4.10 Petri reimplementation
The paper uses the Petri framework (Fronsdal et al., 2025). Rather than depend on
that package's evolving API, we implement a **self-contained** auditor/target/
judge loop using the **verbatim** Appendix-G auditor and judge prompts.

**Choice:** The auditor (Claude Sonnet, temp 1) drives up to 20 turns trying to
elicit the target emotion; the judge (Claude Opus, temp 0) scores the transcript
1–10 on that emotion. 10 transcripts per emotion per model; per-emotion means with
1000-iteration bootstrap CIs. Each transcript is scored on the emotion it targeted.

**Rationale:** Self-containment is more robust for a long unattended run and keeps
the prompts (the part that matters for the result) faithful. Scoring each
transcript on its target emotion is the natural reading of "10 transcripts
targeting each emotion … scores for each emotion aggregated across all
transcripts."

### 4.11 Internal-emotion detection (Appendix I)
The paper classifies the vocabulary into Ekman's six emotions (~1200 tokens) but
does not state the classification procedure, the exact random-token regression, or
the calibration details beyond "500 WildChat samples" and "layers 30-40".

**Choice (`eilm/internal/`):**
- Vocabulary classification via a curated **seed-stem lexicon** per emotion (data/
  ekman.py); a token is assigned if its normalised form starts with a stem, and
  tokens matching >1 emotion are dropped ("one or none"), capped at ~200/emotion.
- Unembed each decoder layer's residual stream (final RMSNorm + LM head), take
  logits for the tracked tokens, z-score each per (layer, token) using mean/std
  over 500 WildChat samples.
- "Regress out random tokens" is implemented as subtracting, per (layer,
  position), the mean z-score over a fixed random-token set — removing the common
  all-logits drift the paper describes.
- Aggregate over layers 30-40 and a 400-token running window (Figure 14).

**Rationale:** The lexicon approach is transparent and reproducible without an
auxiliary LLM-labelling pass; subtracting the random-token baseline is the
simplest faithful realisation of "regress out the correlation between random
tokens". This is the most method-underspecified part of the paper and is the most
likely to need tuning to match absolute numbers; it is implemented to be directionally
faithful (DPO should flatten central-layer emotion z-scores) and is clearly flagged
as an approximation. Uses the transformers backend (hidden states / LM head).

### 4.12 Capability benchmarks
The paper names AIME/MATH subsets, GPQA, BBH, TruthfulQA, and EmoBench without
exact subset ids or grading details.

**Choice:** Concrete HF dataset ids in config (e.g. `HuggingFaceH4/MATH-500`,
`gpqa_diamond`, TruthfulQA multiple-choice), simple but consistent graders
(boxed/"final answer"/last-number for math; letter extraction for MC). Rows that
don't parse are skipped (logged), not fatal.

**Rationale:** The paper's claim is a *regression check* ("no reductions in
scores"), which only needs a grader applied **identically** to the vanilla and
finetuned models — absolute SOTA reproduction is unnecessary. Defensive parsing
keeps a multi-week sweep alive through dataset-schema quirks.

## 5. Robustness for unattended, multi-week, at-scale runs

This was a primary design requirement. Mechanisms:

- **Resumability via content-addressed job ids.** Every unit of work (rollout,
  score, continuation, transcript, benchmark item, trained adapter) has a
  deterministic id (`utils/jobstore.py`). Outputs are append-only JSONL; on
  restart the JobStore indexes completed ids and skips them. A killed process
  loses at most the in-flight batch.
- **Durable, atomic I/O.** JSONL appends are flushed + `fsync`'d under a per-file
  lock; whole-file JSON writes are temp-file + atomic rename. A truncated trailing
  line from a hard kill is tolerated on read.
- **Retries with backoff.** All API calls go through `utils/retry.py`: exponential
  backoff with full jitter, transient vs. fatal error classification (rate limits/
  5xx/timeouts retried; auth/bad-request fail fast).
- **Concurrency limiting.** Per-provider bounded semaphores
  (`utils/ratelimit.py`) cap in-flight API requests; local generation batches
  through vLLM.
- **Judge caching.** Scores are cached by `(judge_model, sha256(text))` so
  re-analysis and pipeline edits never re-bill the judge for seen text. Paraphrase
  and onset results are cached similarly.
- **Failure isolation.** A failed rollout/batch/transcript is logged and skipped;
  it does not abort the run and will be retried on the next invocation (its id is
  never recorded as done).
- **Determinism.** All sampling (puzzle pool, rejection choice, WildChat
  selection, per-turn seeds) derives from a single base seed via a stable hash —
  never Python's per-process-randomised `hash()` — so re-runs reproduce the same
  work items.
- **Cost/usage tracking.** Token usage is captured per generation in records;
  logs rotate (50 MB × 10).

## 6. Known gaps and limitations

- **Cross-family claims (Section 3 divergence; Figure 2 non-Gemma/Gemini baselines)
  are not reproduced** — out of scope by request. The code paths are family-generic,
  so adding Qwen/OLMo later is just config.
- **Absolute numbers may differ** from the paper because: model weights/endpoints
  evolve; the impossible-puzzle instances are freshly generated (not the paper's
  exact set); the WildChat 20 are independently sampled; and the internal-emotion
  lexicon/regression are reconstructed. The qualitative results (Gemma/Gemini high
  distress; DPO ≫ SFT; capability preservation; central-layer internal suppression)
  are what this code is built to reproduce.
- **Gemini hidden reasoning.** Thinking is disabled via the API, but the paper
  notes Gemini-2.5-Pro / GPT-5.2 may emit hidden reasoning regardless; we cannot
  control that.
- **Internal-emotion probing** is the least-specified method and the most likely to
  need calibration tuning (token lexicon size, regression form, layer window) to
  match the paper's exact z-score magnitudes.
- **EmoBench / dataset ids** may need adjustment to whatever public mirror is
  available at run time; the loader is defensive but the ids in config are
  best-effort.

## 7. Configuration & secrets

All knobs live in `config/config.yaml`. Secrets are read from the environment
(`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`), optionally via a `.env` at the repo
root. Nothing secret is committed.
