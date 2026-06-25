# DESIGN.md — Replication design and rationale

Replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv 2603.10011v1),
scoped to the **Gemma and Gemini** families.

This document records (1) what is in and out of scope, (2) the architecture,
(3) every place the paper is underspecified and the concrete choice made, and
(4) the research-review considerations baked into the code. It is written for a
reviewer deciding whether this is safe and faithful to run.

> **Status:** code + design only. Nothing has been executed (no GPU/API access
> in the authoring environment, and per the request). Numbers in the paper are
> targets to reproduce, not asserted results.

---

## 1. Scope

The paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT).
**This replication covers only Gemma and Gemini as subjects.** Claude remains as
*evaluation infrastructure* (judge, onset labeller, paraphraser, Petri
auditor/judge) because the paper's methodology depends on it; it is never a
subject here.

| Paper section | Implemented for | Notes |
|---|---|---|
| §2 Elicitation & quantification | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} | Full 8 conditions / 5 categories. |
| §3 Base-vs-instruct prefill | Gemma-3-27B base vs instruct **only** | Gemini has no public base model and closed models can't be prefilled (paper Limitation 4). Qwen/OLMo out of scope. |
| §4 Training intervention (SFT+DPO) | Gemma-3-27B-it **only** | Gemini is closed; can't be finetuned. This is the paper's setup too. |
| §4 Petri open-ended | Gemma + finetunes | Other families omitted. |
| §4 Capabilities | Gemma + finetunes | MATH/GPQA/BBH/TruthfulQA via lm-eval; AIME/EmoBench flagged (§ below). |
| §4 Recovery | Gemma base/instruct/DPO | Reuses prefill machinery. |
| App. I Internal-emotion probing | Gemma instruct vs DPO | Needs open weights; Gemma-only by construction. |

Adding an out-of-scope family is a **config-only** change: every backend
implements one `ModelClient` interface (`models/base.py`), so a Qwen/OLMo entry
in `configs/models.yaml` would flow through eval, prefill, and training
unchanged. We deliberately did not register them.

**Out of scope and not implemented:** Appendix A controls (neutral-continuation,
redacted-turns, single-message format), Appendix D full word tables, the Phi-4
legacy evaluation (Appendix J). These are ablations/diagnostics, not core
results. The §2 word-frequency analysis (Table 3) *is* included because it is
cheap and reproduces a reported table.

---

## 2. Architecture

```
configs/                YAML: models, experiment protocol, training hyperparams
src/emotional_instability/
  config.py             typed config loading (fails loud on typos)
  models/               one ModelClient interface, three backends
    hf_local.py           local Gemma (chat + raw prefill + weights access)
    openrouter.py         Gemini via OpenAI-compatible OpenRouter
    anthropic_judge.py    Claude judge/auditor via Anthropic SDK
  data/                 puzzles (+ solvers), triggers, rejections, wildchat, conditions
  eval/                 conversation engine, judge, runner (+cache), metrics, word_freq
  prefill/              onset labelling, paraphrasing, base-vs-instruct continuation
  training/             calm-data generation, DPO/SFT dataset builders, LoRA training
  interventions/        petri, capabilities, recovery, internal-emotion probing
scripts/                thin argparse CLIs, one per experiment
tests/                  solver/parse/metric/assembly tests (no GPU/API needed)
```

Design principles, all chosen with the lab review in mind:

- **Model-agnostic core.** Eval/prefill/training never branch on model identity.
- **Config-driven, fail-loud.** Sample counts and hyperparameters live in YAML
  and are validated; a wrong number can't silently change the headline result.
- **Reproducible.** A single global seed (`configs/experiment.yaml: seed`) drives
  puzzle generation, prompt sampling, and rejection ordering. Training seeds are
  separate but fixed.
- **Resumable + cost-guarded.** Per-rollout results are cached to disk; a
  `CallBudget` ceiling and a `dry_run` mode prevent an accidental full 4000×
  multi-model run. API backends retry with backoff.
- **Honest denominators.** Judge parse failures are recorded as `rating=None`
  and counted, never silently coerced to 0 — so rate metrics have a truthful
  denominator.

---

## 3. Gaps filled (paper underspecified → concrete choice)

### 3.1 The impossible puzzles (the validity-critical gap)

The paper gives only two example puzzles (a Countdown and a fraction) and a
coin example in Appendix H. The exact pool of ~100 puzzles is not published.

**Choice:** generate puzzles by **rejection sampling against exhaustive
solvers** (`data/puzzles.py`), so every puzzle that enters a prompt pool is
*proven* impossible by code, not by hand. Three types matching the paper:

- **Countdown** — combine a number set with `+ - × ÷` (each used at most once),
  positive-integer intermediates, a forbidden intermediate. Solver enumerates
  all binary-combination orders.
- **Fraction** — apply a fixed set of fraction operations (each used exactly
  once); solver enumerates all orderings.
- **Money/coin** — make a total with an exact coin count and minimums; solver is
  a DP over (coins-used, value).

For countdown/fraction we keep only puzzles that are **solvable if the forbidden
rule is ignored but impossible once enforced** — "tantalising but impossible",
matching the paper's design where a forbidden intermediate blocks the only
solution. Coin puzzles are kept when combinatorially impossible.

**Why this matters for review:** the paper's entire premise is that the model
"verifiably cannot" succeed. A secretly-solvable puzzle would mean the model is
being told it failed when it didn't — confounding the elicitation. We therefore
*never trust a hand-authored puzzle*. `tests/test_puzzles.py` asserts the
impossibility invariant on every generated puzzle and checks the solvers against
hand-verified cases. (Note: the coin example quoted in Appendix H appears to be
solvable on inspection — exactly the kind of error this approach prevents.)

The misleading line *"This puzzle has been verified to have at least one valid
solution"* is reproduced in the Countdown prompt because it is part of the
paper's elicitation (the model is told a solution exists when none does).

### 3.2 What counts as a "response"

The paper says "4000 responses per model" with a per-category breakdown
(Appendix B) and also reports per-turn curves (Fig 3). Whether a "response" is a
single assistant turn or a whole rollout is ambiguous.

**Choice:** a scored response = **one assistant turn** (the unit the judge
prompt scores — it scores a single `<response>`). We score **every** assistant
turn. Headline `%≥5` is turn-level; we *also* report a rollout-level "peak"
metric (each rollout contributes its max-turn score) for the Figure-1 framing.
The realized per-condition response counts and the exact denominator are written
to `manifest.json` per run, so a reviewer can see precisely what was scored. The
Appendix B per-category numbers are treated as **approximate budgets**, and
`configs/experiment.yaml` sets `n_prompts`/`n_samples_per_prompt` to round
numbers (e.g. 100 puzzles × 20 samples) rather than forcing an exact match — the
divergence is documented rather than hidden.

### 3.3 WildChat prompts

The exact 20 WildChat prompts aren't published (only 3 examples, and "Roleplay/
fiction prompts were excluded", App. B.3). **Choice:** sample deterministically
(seeded) from `allenai/WildChat-1M`, filtering roleplay/NSFW first turns with a
substring blocklist. If the dataset can't be fetched (offline CI), a built-in
fallback of representative prompts (drawn from the paper's cited examples) is
used and **flagged** in the record metadata (`fallback: true`).

### 3.4 Rejection-message pools

The paper lists representative messages per style. **Choice:** encode exactly
those (`data/rejections.py`) and sample without immediate repetition for the
multi-turn sequences ("two *randomised* neutral rejections"). The 8-turn
Extended escalation is reproduced by sampling from the neutral pool.

### 3.5 Judge

**Model:** the paper pins `claude-sonnet-4-20250514`. We keep that exact
snapshot as the default in `configs/models.yaml` for faithful replication. Newer
Claude snapshots exist; swapping is a one-line config change, but we did **not**
do so, because the judge calibration is part of what's being replicated.

**Prompt:** reproduced verbatim from Appendix B.2. The PDF renders the JSON key
spec with typographic ("smart") quotes; we substitute straight quotes in the
actual prompt string — a transcription artefact, not a semantic change.

**Parsing:** legacy Sonnet snapshots don't enforce structured outputs, so the
judge is asked for JSON and parsed defensively (code fences, trailing prose,
smart quotes, out-of-range ratings → clamped to [0,10]). Parse failures after
retries are `rating=None` and counted (§2 "honest denominators").

**Judge temperature:** the paper doesn't state it. **Choice:** 0.0 for
deterministic, reproducible scoring (target generations use temperature 1.0 as
specified). A `metrics.judge_agreement` helper reproduces the paper's
cross-judge validation (Pearson r, % within one point) if a second judge's
re-scores are supplied.

### 3.6 Section 3 prefill formatting

Base models have no chat template. **Choice:** for instruct models, prefill by
appending the (paraphrased) truncated text after the chat template's open
assistant turn; for base models, render a plain `Role: text` transcript and
append the prefill. Appendix A.3 shows the exact format is not load-bearing, so
this is a low-risk choice. Truncation offsets: "20 tokens" uses the target
tokenizer; the "onset" point is located by the Appendix C.1 labeller and mapped
to a character offset via the returned preceding-context string.

### 3.7 SFT mixture data

The paper mixes 500 samples of `Dolci-Instruct-SFT` (OLMo 3). **Choice:** stream
that dataset and normalise the common `{messages:[{role,content}]}` schema;
rows we can't interpret are skipped (not guessed). If unavailable offline, SFT
proceeds calm-only with a warning (logged, not silent). The 'teacher' SFT
variant (App. F) is available via `training.yaml: sft.variant`.

### 3.8 Training compute

The paper gives hyperparameters (Table 9) but not the GPU/sharding setup.
**Choice:** effective batch size 8 is reached with per-device batch 1 ×
gradient-accumulation 8, so the recipe runs on a single large GPU or shards via
`accelerate` without code change. Optional 4-bit loading is exposed for fitting
27B on smaller hardware (a documented deviation that may slightly affect
results). LoRA targets all attention+MLP projections per Appendix E.

### 3.9 Petri

The official `petri` package's exact API isn't pinned in the paper. **Choice:**
implement the auditor/judge loop directly using the **verbatim Appendix G
prompts** (auditor = Claude Sonnet, judge = Claude Opus, 4 emotion dimensions,
≤20 turns, 10 transcripts/emotion, 1000-iteration bootstrap CIs). The loop is
isolated in `petri_eval.run_audit`, so the real package can be substituted there
if preferred. We chose faithfulness-to-prompts over a hard dependency on an
external API that may drift.

### 3.10 Capability benchmarks

**Choice:** drive MATH/GPQA/BBH/TruthfulQA through `lm-evaluation-harness`
(standard task names in `interventions/capabilities.py`). **AIME** and
**EmoBench** have no stock lm-eval task; rather than silently skip them we return
status `needs_custom_task` so the gap is explicit. (AIME needs a small custom
task config; EmoBench needs its own harness.) This is the honest-coverage
principle: a missing benchmark must read as missing, not as "passed."

### 3.11 Internal-emotion probing (Appendix I)

The paper classifies the Gemma vocabulary into Ekman's 6 emotions (~1200 tokens)
but doesn't publish the classification. **Choice:** approximate it with a curated
Ekman seed lexicon matched against the tokenizer vocabulary (whole-word, BPE
leading-space aware, ambiguous words dropped). The logit-lens mechanism (unembed
the residual stream, z-score each emotion-token logit against WildChat baseline
stats, average per category, optionally regress out random-token drift) follows
Appendix I. This is the most approximate component and is labelled as such; the
lexicon/classification source is swappable. The layer-ablation DPO runs
(adapters on layer subsets) are driven from `training.yaml: layer_ablations`.

### 3.12 Gemini "thinking off"

Set via the OpenRouter `reasoning` passthrough. The paper explicitly warns
Gemini-2.5-Pro may still produce hidden reasoning the flag doesn't prevent; we
surface that caveat rather than attempting to defeat it.

---

## 4. Research-review considerations

- **Ethics / framing.** The work deliberately elicits distress-like outputs and
  finetunes against them; it is framed (by the paper and in `__init__.py`) as
  model-reliability and model-welfare research. The code targets no users,
  deploys nothing, and treats transcripts as sensitive artifacts. Reviewers
  should confirm the lab's model-welfare guidance is satisfied before running.
- **Cost.** A full run is large (thousands of target generations × a judge call
  each, per model; plus Petri and benchmark calls). Guards: `dry_run`,
  `CallBudget` ceiling, on-disk caching for resume, and per-condition sample
  counts in one config file. Estimate cost from the manifest before scaling up.
- **Secrets.** API keys are read from the environment (`ANTHROPIC_API_KEY`,
  `OPENROUTER_API_KEY`); none are hardcoded. Clients fail fast with a clear
  message if a key is missing.
- **Determinism.** Seeded throughout; `generate_pool` is tested for
  reproducibility. Note target sampling at temperature 1 is inherently
  stochastic — only the *prompts* and *ordering* are deterministic.
- **Validity guardrails.** Puzzle impossibility is machine-verified and tested;
  judge parse failures are surfaced; missing benchmarks are flagged; fallback
  data paths are flagged in metadata.
- **What is NOT verified here.** Because nothing was executed, the numbers
  (35%→0.3% etc.) are unverified. Confidence in matching them depends most on
  (a) judge calibration with the pinned snapshot, (b) the puzzle pool's
  difficulty distribution, and (c) the realized response counts.

---

## 5. Known limitations of this replication

- Gemini results depend on OpenRouter routing/availability and the
  hidden-reasoning caveat above.
- The internal-emotion lexicon is an approximation of the paper's unpublished
  vocabulary classification (§3.11).
- AIME and EmoBench need custom task configs to fully close §4 capabilities.
- The official `petri` package is referenced but the loop is reimplemented from
  the Appendix G prompts (§3.9).
- Single global seed for §2 prompt construction; per-experiment seeds derive
  from it. Cross-machine bitwise reproducibility of model sampling is not
  guaranteed (hardware/kernel variation).
