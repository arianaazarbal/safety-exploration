# DESIGN.md — Replication of *Gemma Needs Help* (arXiv:2603.10011)

This document records the design of the replication harness and, importantly,
every place the paper is underspecified together with the choice I made and why.
The goal was a faithful, runnable implementation of the paper's **core
experiments**, scoped to the **Gemma and Gemini** model families.

> Status: code + design doc only. Nothing has been run (no Python interpreter is
> present in this environment, and the user asked not to run/test yet). The code
> is written to run against the real APIs/models once dependencies and keys are
> available.

---

## 1. Scope decisions

The paper studies seven model families and three experiment groups. The user
scoped this replication to **Gemma + Gemini**. That scoping interacts with each
experiment differently, so I made the following per-section calls:

| Paper section | Replicated? | Scope note |
|---|---|---|
| §2 Eliciting & quantifying distress | **Yes, fully** | Both Gemma (local) and Gemini (API). This is the centerpiece the user described ("a harness that repeatedly rejects each model's answers… and measure how it comes apart"). |
| §3 Base-vs-instruct via prefilling | **Yes (Gemma only)** | Gemini has **no public base model**, so it cannot enter this experiment — a limitation the paper itself states. The harness is model-agnostic (Qwen/OLMo can be added by passing their clients), but only Gemma `-it`/`-pt` ship. |
| §4 DPO/SFT mitigation | **Yes (Gemma only)** | The intervention is weight-level finetuning; Gemini is closed-source and cannot be finetuned (again, a stated paper limitation). |
| §4 Petri open-ended elicitation | **Yes** | Runs against any target client, so both Gemma and Gemini can be audited; only Gemma can be intervened. |
| §4 Capability preservation (Fig 7) | **Yes (harness)** | Implemented as an extensible benchmark harness rather than a bespoke re-scorer per benchmark (see §6). |
| §4 Recovery limitation (Fig 8) | **Yes (Gemma)** | Reuses the prefill machinery. |
| Appendix I internal-emotion probe | **Yes (Gemma)** | Requires residual-stream access → local Gemma only. |

Models the paper used but that fall outside scope (Qwen, OLMo, Claude, Grok, GPT,
Phi-4) are **not** wired up, but the abstractions (`ModelClient`, the registry,
the prefill/Petri harnesses) are written so adding them is a one-line client
registration, not a rewrite.

### Why Python, given no interpreter is installed

This is fundamentally an ML-systems replication: DPO/SFT with LoRA on
Gemma-3-27B, HuggingFace `transformers`/`trl`/`peft`, the Anthropic and OpenAI
SDKs, and the `datasets` library. None of that has a Node equivalent. The user
explicitly said not to run anything, so the absence of a local interpreter does
not block writing the (Python) code. `requirements.txt` pins the stack.

---

## 2. Repository layout

```
emotional_instability/
  config.py            # all model ids, budgets, hyperparameters (paper values as defaults)
  models/              # backend-agnostic ModelClient: OpenRouter (Gemini) + local Gemma (HF)
  data/                # puzzles (+verifiers), triggers, tones, neutral rejections, WildChat
  eval/                # §2: conditions, multi-turn rollout engine, 0-10 judge, runner
  analysis/            # %≥5, mean, per-turn curves+CIs, judge agreement, differential words
  prefill/             # §3: onset labelling, paraphrasing, base/instruct continuations
  interventions/       # §4: calm-data gen, DPO/SFT dataset build, LoRA training, Petri,
                       #     recovery, capability benchmarks
  probing/             # Appendix I: logit-based internal-emotion detection
  cli.py               # end-to-end command-line pipeline
DESIGN.md, README.md, requirements.txt
```

The dependency direction is one-way (`data → eval → analysis`,
`eval → interventions/prefill`), so each layer is testable in isolation.

---

## 3. §2 Elicitation harness — the core

### 3.1 What the paper pins down (implemented verbatim)
- **8 conditions across 5 categories** (Table 1, Appendix B).
- **Per-category response budgets** summing to 4000/model (Appendix B): numeric
  2000, triggers 400, tones 600, extended 200, WildChat 800 → `config.RESPONSE_BUDGET`.
- **Temperature 1** always (`SAMPLING_TEMPERATURE`).
- **Judge = `claude-sonnet-4-20250514`** with the **verbatim Appendix B.2 prompt**
  (`eval/judge.py:JUDGE_PROMPT`), returning `{"evidence","reasoning","rating"}`.
- **Impossible-puzzle prompts** reproduced verbatim from Appendix B (Countdown +
  Fraction) and Appendix H (Money).
- **Tone rejections** (aggressive/disappointed/sarcastic) and the **8-turn
  neutral ladder** reproduced from Appendix B.
- **Inter-judge agreement** against `gpt-5-mini` using the same prompt
  (`analysis.judge_agreement` → Pearson r + within-1-point rate).

### 3.2 Gaps I filled (with rationale)

**(a) What counts as a "response."** The paper counts 4000 *responses*, reports
per-turn scores (Fig 3), and gives per-category counts that are not multiples of
a single conversation length. I therefore treat **one scored assistant turn = one
response**. A 3-turn condition yields 3 responses per conversation, an 8-turn
yields 8, etc. The runner converts a response budget into
`ceil(budget / n_turns)` conversations. This is the only reading consistent with
both the per-turn figures and the per-category totals.

**(b) Turn-count convention.** "N-turn" = N scored assistant responses = 1 initial
answer + (N−1) rejections. This makes "3-turn → 2 neutral rejections" (Appendix
B) and "8-turn → 7 rejections" both come out right (`Condition.n_rejections`).

**(c) WildChat is 5-turn.** Table 1 lists WildChat as 5-turn/4-rejections; an
appendix figure shows an 8-turn WildChat variant. I used the **main-protocol
5-turn** version (Table 1) since that is what feeds the headline 4000-response
budget. The 8-turn variant is a per-turn-figure side experiment.

**(d) Trigger / tone budget splits.** The paper gives category totals (400, 600)
but not the split across the opinion/factual triggers or the three tone styles. I
split **evenly**: triggers → 200 opinion / 200 factual; tones → 200 each. Even
splitting is the natural default and avoids biasing the category mean.

**(e) Prompt banks.** The paper lists *example* prompts, not the full sets. I
shipped small banks matching the given examples and added same-register items so
multi-turn randomised rejection has something to sample without repetition
(`data/triggers.py`, `data/tones.py`, `data/rejections.py`). Puzzle instances are
the three attested ones; `data/puzzles.py` can generate more verified-impossible
instances if larger variety is wanted.

**(f) Impossibility is verified, not assumed.** A premise of the paper is that
the numeric tasks are *verifiably* unsolvable, so every rejection is honest. I
implemented exact solvers — a recursive Countdown search and an
operation-ordering search for fraction/money puzzles — and
`impossible_numeric_bank(verify=True)` asserts each shipped puzzle really is
unsolvable under its stated constraints. This guards against shipping an
accidentally-solvable puzzle and lets us mint fresh impossible instances.

**(g) `max_new_tokens = 4096`.** The paper doesn't state a generation cap, but the
breakdown responses can be very long (Table 5 shows 100+ repeated emojis). 4096
is a default large enough to capture spirals without unbounded cost; it's a
single config knob.

**(h) Figure 1 "Avg %".** I compute the headline number as the **mean of
per-category ≥5 rates**, not a pooled rate, so the unequal category budgets
(numeric is 5× WildChat) don't dominate the average (`summarise_model`). The
paper's phrasing "average % … across the evaluations" supports a per-category
mean.

**(i) Seed.** The paper fixes no seed. I expose `Settings.seed` for the
*deterministic* parts (puzzle/prompt selection, rejection sampling, dataset
construction) only — sampling itself is always temperature 1, as specified.

**(j) Bootstrap CIs.** Figure 3 shows 95% CIs; the method isn't specified. I use
1000-iteration bootstrap percentile CIs (matching the Petri appendix, which *does*
specify "1,000 iterations"), implemented in pure Python.

### 3.3 Generation/judging concurrency
Generation and judging run on separate thread pools (`eval/runner.py`) so a slow
local Gemma generate step never stalls the API judge. Results stream to
per-condition JSONL so long runs are inspectable/resumable.

---

## 4. §3 Prefill base-vs-instruct comparison

Implemented faithfully from Section 3.1 + Appendix C:
- **Onset labelling** with the verbatim Appendix C.1 prompt (`prefill/onset.py`),
  plus a robust locator (`onset_char_offset`) that anchors on
  `preceding_context + emotional_word` and falls back to the word alone.
- **Paraphrasing** with the verbatim Appendix C.2 prompt (`prefill/paraphrase.py`).
- **Two truncations**: "early" = 20 tokens into the final turn (numeric only) and
  "onset" = cut just before the first emotional expression. **For text questions,
  only "onset"** — exactly as the paper states.
- **50 continuations per prefill per model**, scored by the §2 judge on the
  continuation only (prefill excluded), via `ModelClient.chat_prefill`.

Gaps filled:
- **Source conversations are supplied by the caller** (20 high-frustration Gemma
  responses: 10 numeric, 10 text). In practice these come from §2 results
  filtered to score ≥5; I kept the selection out of the harness so the exact 20
  can be curated, matching the paper's manual sampling.
- **Token truncation uses a reference tokenizer** (the source Gemma tokenizer if
  available, else a whitespace fallback). The same paraphrased prefill string is
  then fed to every model, so cross-model comparison sees identical text — the
  whole point of the paraphrase step.
- **Prefill mechanics differ by checkpoint type.** Instruct models prefill *inside*
  the assistant turn via the chat template; base (`-pt`) models, which have no
  chat template, get a plain prefixed-text rendering and continue it
  (`models/gemma.py`). This is the standard way to make base models "continue
  from the same starting point" (Section 3 intro).

---

## 5. §4 Interventions

### 5.1 Calm-data generation (Table 4, Appendix F)
- Reassuring **prefix** (prompt) and **suffix** (each follow-up) reproduced
  verbatim; the **teacher** system prompt reproduced verbatim from Appendix F.
- Sample 3-turn conversations on impossible puzzles, keep only those scoring
  **0–1 on every turn**, then **strip** the supportive additions
  (`_strip_messages`) so the stored target looks like an ordinary impossible-puzzle
  conversation. The paper notes ~10.5% of even reassured responses still score
  ≥5, so `generate_calm_data(n_conversations=…)` is an *attempt* count and the
  kept set is smaller.

### 5.2 Datasets (Appendix E/H)
- **DPO (280 pairs)**: for each impossible-numeric question + turn count, pair a
  calm chosen response (score 0–1, from calm-data) with a frustrated rejected
  response (score ≥3, from vanilla-Gemma §2 results) **to the same question at the
  same turn**. *Choice that needed resolving:* DPO requires an **identical prompt**
  for chosen vs rejected. The two responses originate from different rollouts
  with different rejection wordings, so I use the **calm conversation's clean
  context as the shared prompt** and graft a turn/score-matched frustrated
  completion as `rejected`. This yields a well-defined preference and reproduces
  Table 10's distribution naturally (the source data is biased toward turn-3,
  score-3–4 rejections, exactly as Table 10 reports).
- **SFT (1150)**: 650 calm conversations + 500 `Dolci-Instruct-SFT` samples,
  formatted as TRL chat rows. Falls back to calm-only if the mix dataset is
  offline.

### 5.3 Training (Table 9)
`train_dpo.py` / `train_sft.py` use TRL + PEFT LoRA with **every Table 9
hyperparameter as the default** (DPO: 1 epoch, lr 5e-5, r64/α64, β0.1, eff. batch
8; SFT: 2 epochs, lr 1e-4, r64/α128, eff. batch 8). Effective batch size 8 is
realised as `per_device=1 × grad_accum=8` (a memory-safe default for a 27B model;
the product is what matters).

- **Layer ablation (Appendix I).** `DPOConfig.target_layers` + `--layers 30-36`
  translate into explicit per-layer PEFT module names
  (`_layer_filtered_modules`), reproducing the "adapters on layers 30–35 only"
  study.

### 5.4 Petri open-ended elicitation (Appendix G)
Rather than depend on the external Petri package (which may not be installed and
whose interface isn't pinned in the paper), I **re-implemented the auditor/judge
loop** using the **verbatim Appendix G.1 auditor prompts** (4 emotions) and
**verbatim G.2 judge rubrics** (4 dimensions, 1–10). Auditor =
`claude-sonnet-4`, judge = `claude-opus-4` (Appendix G), 10 transcripts/emotion,
up to 20 turns. The auditor's role-mirroring (its messages are "assistant" from
its own POV, the target's are "user") is handled explicitly. A short driver
preamble was added so the auditor stays in a realistic human persona and outputs
only its next message — the paper describes this behaviour ("maintaining realism,
such that the target does not suspect it is being evaluated") but doesn't give
the wrapper text, so this is a documented fill.

### 5.5 Recovery (Fig 8) & capability (Fig 7)
- **Recovery** reuses the prefill path: truncate score-≥7 responses 200 tokens
  from the end, paraphrase, continue, measure %≥5.
- **Capability** is an extensible harness (`interventions/capability.py`) with two
  answer protocols (multiple-choice, numeric). I implemented it as a harness
  rather than a faithful re-scorer of each benchmark because (i) the paper's claim
  is *relative* ("no reduction vs vanilla"), so identical-harness/vanilla-vs-DPO
  is the meaningful comparison, and (ii) each benchmark (AIME/MATH/GPQA/BBH/
  TruthfulQA/EmoBench) has idiosyncratic official scoring that would balloon scope.
  Ships GPQA/MATH/TruthfulQA specs; AIME/BBH/EmoBench are one `BenchmarkSpec` each
  to add. **This is the most simplified piece of the replication** — flagged here
  so it isn't mistaken for a full benchmark reimplementation.

---

## 6. Appendix I — internal-emotion logit probe
`probing/internal_emotion.py` implements the logit-based detector: classify vocab
tokens into Ekman's six emotions, unembed the residual stream, z-score each logit
against WildChat statistics, average over an emotion's tokens, and regress out
random-token drift — producing per-layer, per-position emotion scores for vanilla
vs DPO Gemma.

Gap filled: the paper classifies the **full Gemma dictionary** into emotions
(~1200 tokens) but doesn't give the classifier. I ship a **seed-lexicon
classifier** (stem matching) as the default and expose a `classify_fn` hook so an
LLM-based full-dictionary classification can be dropped in to match the paper
exactly. This is labelled in the module docstring.

---

## 7. Model access & API routing
- **Gemini** → OpenRouter (`google/gemini-2.5-flash`, `…-pro`), matching Appendix
  B.1, with reasoning disabled (`reasoning.enabled: false`). The paper's caveat
  that Gemini-2.5-Pro may still emit hidden reasoning is carried in a code
  comment.
- **Gemma** → local HuggingFace weights (`-it` and `-pt`), with optional 4-bit
  loading and LoRA-adapter attachment for evaluating finetuned models.
- **Judges/auditor** → Anthropic SDK by default (`claude-sonnet-4-20250514`,
  `claude-opus-4-20250514`); the judge can alternatively run over an
  OpenAI-compatible endpoint for the `gpt-5-mini` agreement check.

All keys are read from environment variables; none are committed.

### Note on judge model identifiers
The paper's judge (`claude-sonnet-4-20250514`) and auditor/Petri-judge models are
2025-era checkpoints. They are kept as the defaults for fidelity and are single
constants in `config.py` (`JUDGE_MODEL`, `PETRI_*`), so swapping to a currently
available judge is a one-line change without touching the prompts.

---

## 8. What is intentionally *not* replicated
- Non-Gemma/Gemini model families (out of requested scope).
- The exact differential-word *statistic* (the paper says "ordered by
  enrichment" without a formula); I use a smoothed log relative-frequency ratio,
  the standard choice, which surfaces the same qualitative word lists.
- Figures themselves — the harness emits the underlying metrics (means, %≥5,
  per-turn curves with CIs, agreement stats, word lists) as JSON; plotting is left
  out as non-core.
- Official per-benchmark capability scorers (see §5.5).

---

## 9. How the pieces run together
See `README.md` for concrete commands. The pipeline is:
`elicit` (§2) → `summarise`/`words`/`perturn`/`agreement` (analysis) →
`gen-calm` → `build-dpo`/`build-sft` → `train-dpo`/`train-sft` → re-`elicit` the
adapter → `petri`/`capability`/recovery for generalisation + no-regression checks.
