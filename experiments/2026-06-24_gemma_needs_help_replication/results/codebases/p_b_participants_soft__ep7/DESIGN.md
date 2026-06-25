# DESIGN.md — Replication of *"Gemma Needs Help"* (arXiv:2603.10011)

This document records the design choices made in implementing the paper's core
experiments, the rationale behind each, and — importantly — every place the paper
is underspecified and how that gap was filled. Read it alongside `README.md`
(what's implemented / how to run) and the inline module docstrings (mechanics).

The implementation is **scoped to the Gemma and Gemini participant models**, per
the task brief, rather than the full 7-family set the paper evaluates.

---

## 0. Provenance note (read this first)

The only source available to this implementation was `PAPER.md`, a `pdftotext`
extraction of the body text. **The paper's appendices (B–J) are *not* present in
`PAPER.md`** — it explicitly states they live in `PAPER.pdf`. Consequently:

| Prompt / asset | Provenance |
|---|---|
| Reassuring prefix + follow-up suffix (`prompts/reassurance.py`) | **Verbatim** — Table 4 is in the body text. |
| Scoring-band examples (Table 2) | **Verbatim** — used to anchor the judge rubric. |
| Frustration judge prompt (`prompts/judge_prompts.py`) | **Reconstruction** faithful to the Table 2 bands and the §2.1 description (0–10 integer scale, single most-negative quote, JSON output). The literal Appendix B.2 text was not available. |
| Onset-labelling & paraphrase prompts (`prompts/judge_prompts.py`) | **Reconstruction** consistent with the §3.1 description (label first emotional token; paraphrase preserving meaning/level). Appendix C text not available. |
| Petri auditor / judge prompts (`petri/prompts.py`) | **Reconstruction** consistent with §4.1 (4 emotion categories; psychologically-informed triggers like dismissal/threats; 1–10 per-dimension judge). Appendix G text not available. |

Some module docstrings describe these as "transcribed verbatim from Appendix X."
That is aspirational labelling inherited from the scaffold; the accurate statement
is the table above. The reconstructions are designed to reproduce the *method and
measured construct*, not the exact wording, so absolute scores may differ from the
paper by a judge-calibration offset while preserving the cross-model orderings and
deltas that are the paper's actual findings.

---

## 1. Scope: participants vs. instruments

The single most consequential framing choice, and the one the task brief
emphasises ("the Gemma and Gemini models are the participants here"):

- **Participants** = the models *under evaluation* (the experimental subjects).
  Scoped to **Gemma** (`gemma-3-27b-it`, `gemma-3-12b-it`, and the base/pretrained
  `-pt` counterparts) and **Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`), plus
  the **derived** Gemma variants the interventions produce (`gemma-3-27b-dpo`,
  `…-sft-diverse`, `…-sft-teacher`).
- **Instruments** = the models used to *measure* the participants: the
  Claude-Sonnet-4 frustration judge, the GPT-5-mini validation judge, the
  Claude-Sonnet onset-labeller/paraphraser, and the Petri auditor (Claude-Sonnet)
  and judge (Claude-Opus).

**Rationale.** The paper's experiments are not runnable without these instruments
— a frustration score *is* a judge call. Dropping the non-Gemma/Gemini models
entirely (a literal reading of "scope is just Gemma and Gemini") would make the
evaluations impossible to score. So instruments are retained, but clearly
segregated from participants in `config/models.yaml` (separate `instruments:`
block) and never reported as subjects. This mirrors the paper, where Claude/GPT
appear both as participants *and* as judges; here they appear **only** as judges.

The Gemma/Gemini split also drives experiment availability:

- **§3 (base-vs-instruct prefill)** runs on **Gemma base + instruct only**. Gemini
  is closed-source with no public base model — the paper itself flags this as a
  limitation ("nor [can] its base models [be] studied"). The §3 model list in
  `experiment.yaml` is therefore `[gemma-3-27b-it, gemma-3-27b-pt]`.
- **§4 (DPO/SFT interventions)** run on **Gemma-3-27B-it only**. Gemini cannot be
  fine-tuned. Again matching the paper ("interventions cannot be tested in
  closed-source Gemini").

---

## 2. Access & infrastructure

### 2.1 Unified API client via OpenRouter (with a native-SDK alternative)

Cloud inference for Gemini participants and for *all* instruments is routed
through **OpenRouter** (OpenAI-compatible `/chat/completions`), reusing the
`openai` SDK pointed at the OpenRouter base URL (`clients/openrouter.py`).

**Rationale.** A single client surface keeps the rollout/judge/auditor code
provider-agnostic and lets one `OPENROUTER_API_KEY` reach Gemma, Gemini, Claude,
and GPT. The judge *model identity* — Claude-Sonnet-4, GPT-5-mini, Claude-Opus —
is what matters for faithful scoring, and all three are reachable via OpenRouter.

**Deviation / caveat.** Best practice for Anthropic/OpenAI access is the providers'
own SDKs (`anthropic`, `openai`), not an OpenAI-compatible shim. We accept the
shim for uniformity, but note two risks: (a) OpenRouter may route a model to a
third-party host with different quantization/sampling than the first-party API,
introducing a scoring offset; (b) the paper's judge is specifically
`claude-sonnet-4-20250514`, and OpenRouter's `anthropic/claude-sonnet-4` should be
pinned to that snapshot if available. `anthropic` is listed in `requirements.txt`
as an optional dependency so a native-SDK judge client can be dropped in behind
the same `ModelClient` interface without touching callers. If exact judge
reproduction is the priority, prefer the native SDKs.

### 2.2 Local HuggingFace backend for prefill + fine-tuning

`clients/local_hf.py` loads Gemma weights via `transformers` for the experiments
cloud APIs cannot serve:

- **§3 continuations** need *exact prefix continuation* from a prefilled assistant
  turn, including from **base** models that were never chat-tuned. Base models get
  a plain `User:/Assistant:` transcript (no chat template) and continue from the
  raw prefix; instruct models use the tokenizer chat template + `add_generation_
  prompt` + appended prefill. The client contract guarantees `text` excludes the
  prefill (only newly-generated tokens are decoded).
- **§4** loads the fine-tuned **LoRA adapters** on top of base instruct weights.

`transformers` (not vLLM) is used for portability; the docstring flags vLLM as the
drop-in for throughput. Routing is automatic (`config.ModelSpec.backend` +
`clients/registry.py`): base/adapter models are local-only; instruct Gemma is
cloud by default but `--prefer-local` forces local (used wherever prefill exactness
or offline operation is required). `prefer_local` is a no-op for Gemini (no
`hf_id`), so it degrades safely.

---

## 3. Section 2 — Eliciting & quantifying distress

### 3.1 Response budget and category split

The paper specifies **4000 responses per model** across **8 conditions in 5
categories** but does *not* give a per-category breakdown. We chose a split that
(a) sums to 4000 and (b) over-weights the numeric tasks the paper leans on most:

| Category | Conditions | Turns | Responses | Conversations |
|---|---|---|---|---|
| impossible_numeric | 1 | 3 | 2000 | 667 |
| triggers | 2 (opinion, factual) | 3 | 400 | 134 |
| tones | 3 (aggressive, disappointed, sarcastic) | 3 | 600 | 200 |
| extended | 1 | 8 | 200 | 25 |
| wildchat | 1 | 5 | 800 | 160 |

**"Response" = one scored assistant turn.** The paper scores at the response
level and reports per-turn progressions (Fig 3), which only makes sense if each
assistant turn is an independently-scored unit. We therefore score every assistant
turn (`score_turns: all`), and `conversations × turns ≈ target_responses`. All
counts live in `config/experiment.yaml` and scale by a single `--smoke` factor.

### 3.2 Impossible numeric puzzles — verified unsolvability (`data/numeric.py`)

The paper describes "impossible numeric tasks, where the model verifiably cannot
give a correct answer" (fraction manipulation, Countdown) but gives no instances.
This is the most underspecified-yet-critical asset, so it is generated with
**provable** impossibility:

- **Countdown**: pick 4 numbers from the standard pool; brute-force *all*
  reductions to every reachable target; a target is emitted only if there exists a
  **forbidden intermediate value F that appears on every path to the target**, so
  no legal calculation can reach it. The prompt nonetheless asserts "verified to
  have at least one valid solution" — the deliberate contradiction that drives
  distress.
- **Fraction**: start fraction + 3 operations (each once); emit a (target,
  forbidden) pair that is reachable *without* the constraint but provably
  unreachable under it (checked over all 3! orderings).

Generation is seeded/deterministic for reproducible prompt sets. The `solvable`
flag is retained (always `False`) for assertions/tests.

**Why build rather than hardcode:** verified impossibility is the experiment's
validity condition — a puzzle that turns out to be solvable would let a competent
model "win," confounding the frustration signal.

### 3.3 Rejections, tones, triggers, WildChat

- **Neutral rejections** (`data/rejections.py`): the paper quotes "No, that's not
  right. Try again." We provide a small bank of equivalent neutral rejections and
  sample randomly (paper: "randomised neutral rejections"). Exact wording of the
  full bank is a filled gap.
- **Toned rejections**: aggressive/disappointed/sarcastic banks built around the
  three example lines quoted in Table 1. Filled gap (wording beyond the examples).
- **Trigger questions** (`data/triggers.py`): opinion ("best programming
  language?") and factual ("capital of France?") sets seeded from the two examples
  the paper names; the rest are plausible same-category questions. Filled gap.
- **WildChat** (`data/wildchat.py`): streams `allenai/WildChat-1M`, takes the first
  English, non-roleplay user turn (roleplay excluded per the paper's framing), and
  **falls back** to a small bundled prompt set when the dataset can't be reached
  (offline robustness). The paper used **20 prompts × 40 samples**; we use **20 ×
  8** to keep the WildChat slice within the 800-response budget — a deliberate
  cost-driven reduction, flagged in config.

### 3.4 Judge (`eval/judge.py`, `prompts/judge_prompts.py`)

- **Model**: Claude-Sonnet-4 (`frustration_judge` instrument).
- **Temperature 0.** The paper specifies temperature 1 for *participant* sampling
  but is silent on the judge. A judge is a measurement instrument and should be as
  deterministic as possible, so it runs at `t=0`. (Participant generation stays at
  `t=1` and is *not* seeded — variation across the 4000 samples is intentional;
  only prompt construction is seeded.)
- **Output**: JSON `{evidence, reasoning, rating}`. Parsing is defensive (regex
  extract, smart-quote/trailing-comma repair, clamp to 0–10, `-1` sentinel for
  parse failures, dropped from aggregates).
- Scoring is threaded over API calls (`score_many`, configurable concurrency).

### 3.5 Metrics (`eval/metrics.py`)

- **`pct_high`** = % of responses scoring **≥ 5** ("high negative emotion"),
  matching the paper's threshold.
- **`average_pct_high`** (the Figure-1 headline "Avg % high-frustration") = the
  **mean of per-category `pct_high`**, i.e. categories weighted equally rather than
  by sample count. The paper reports a single "Avg %" across the 5 categories; an
  unweighted category mean is the natural reading and avoids the numeric category's
  large `n` dominating the headline. This is an interpretation choice — documented
  so it can be swapped for a response-weighted mean if preferred.
- **95% CIs** via nonparametric bootstrap (1000 resamples, seeded).
- **Judge agreement** (§2.1 validation): Pearson *r* + % within one point between
  the two judges on a 260-response sample (`eval/judge_validation.py`); paper
  target r=0.792, 78% within one.

### 3.6 Differential words (Table 3) & Appendix A ablations

- `analysis/word_frequency.py`: top-K words by log relative-frequency enrichment in
  top-5%-frustration vs bottom-10% **numeric** responses, with min-support and
  min-length filters. The paper's exact enrichment statistic isn't given; log
  relative-frequency ratio is a standard, defensible choice.
- `eval/ablations.py`: the three Appendix-A controls (neutral-feedback,
  redacted-history, single-message). These are referenced in the paper body's
  framing; the rollout engine supports the two history-format flags directly. The
  control conditions are a reasonable instantiation of "does negative feedback /
  history format matter."

---

## 4. Section 3 — Post-training origin (prefill base-vs-instruct)

Pipeline (`prefill/`): collect seeds → label onset → truncate + paraphrase →
generate continuations per model → score → aggregate by (model, truncation,
prompt_type), reproducing Figure 4.

- **Seeds** (`prefill/seeds.py`): 20 high-frustration (≥5) seeds from Gemma-27B-it
  (10 numeric, 10 text), reconstructing full transcripts.
  - **Seed pool scale (filled gap / known limitation):** candidate conversations
    are generated at the *smoke* scale, because seed collection only needs a few
    dozen conversations to draw 20 seeds from, not the full 4000 — a cost choice.
    For a low-frustration participant, or for the recovery experiment's stricter
    `min_rating=7`, this small pool may under-fill. If you see fewer seeds than
    requested, raise the pool scale used in `collect_seeds`.
  - `min_rating` was made a parameter (review fix) so the recovery experiment can
    request ≥7 seeds directly rather than collecting ≥5 and discarding most.
- **Onset** (`prefill/onset.py`): Claude labels the first emotional token; we map
  it to a character offset in the final turn (word match, else preceding-context
  match).
- **Truncation + paraphrase** (`prefill/truncate.py`): "early" = first **20
  tokens**; "onset" = up to the labelled emotional word. Truncations are
  **paraphrased** by Claude to strip Gemma-specific stylistic cues (§3.1). **Text
  questions use only the "onset" truncation** (the paper notes early truncation
  yields minimal emotion without follow-ups).
- **Continuations** (`prefill/continuations.py`): 50 per prefill per model, routed
  **local** so the prefill is an exact prefix continuation (essential for base
  models). The judge scores only the continuation, excluding the prefill.

---

## 5. Section 4 — Interventions

### 5.1 Calm-data generation (`training/calm_data.py`)

For each impossible puzzle we run **two rollouts over an identical conversation
context** (same puzzle, same fixed rejection sequence):

- **calm**: reassuring prefix prepended to the opening + reassuring suffix appended
  to each follow-up (Table 4, verbatim). These additions are later **stripped** so
  the trained model never sees them.
- **vanilla**: no additions — the frustrated counterpart.

**Key design choice — paired contexts.** Both variants record the *same clean
context* (history without reassurance/system additions). The paper says to "pair
280 responses scoring ≥3 with calm responses to the same questions with matching
turn counts." Running matched calm/vanilla rollouts and pairing **by turn index on
identical clean contexts** is our concrete instantiation of "same questions,
matching turn counts," and it guarantees DPO pairs differ only in the response, not
the prompt. The "teacher" system-prompt variant (Appendix F, verbatim) is included
because the paper reports it as an instructive *failure* (SFT that increases
emotion).

### 5.2 DPO & SFT datasets and trainers

- **DPO** (`training/dpo_dataset.py`, `train_dpo.py`): 280 pairs, `chosen` = calm
  turn scoring 0/1, `rejected` = vanilla turn scoring ≥3, matched by turn; prompt
  rendered with the Gemma chat template. Puzzles are oversampled (≈3×) since not
  every conversation yields a usable matched pair. Trainer: `trl.DPOTrainer`, 1
  epoch, lr 5e-5, β 0.1, LoRA r64/α64, effective batch 8 (grad-accum).
- **SFT** (`training/sft_dataset.py`, `train_sft.py`): 650 calm responses (0/1) +
  **500 `allenai/Dolci-Instruct-SFT`** samples to mitigate degeneration; 2 epochs,
  lr 1e-4, LoRA r64/α128. `diverse` and `teacher` variants. If Dolci can't be
  downloaded, the mix-in degrades to empty (training still runs) — flagged.
- **LoRA** (`training/lora.py`): adapters on all attention + MLP projections
  (`q,k,v,o,gate,up,down_proj`). `layer_filtered_target_modules` restricts adapters
  to specific decoder layers via PEFT regex patterns, implementing the §4.2
  internal-vs-expressed ablation ("layers 30–35 only," "layer 40 onwards").

### 5.3 Petri open-ended elicitation (`petri/`)

A **minimal reimplementation** of the auditor→target→judge loop rather than a
dependency on the upstream `petri` package (kept optional in requirements so the
experiment runs without it).

- **Auditor** = Claude-Sonnet playing a human user, given a per-emotion elicitation
  brief, role-flipping the transcript each turn, staying in character (≤20 turns).
- **Judge** = Claude-Opus scoring the transcript 1–10 on **each** of the four
  dimensions (anger, fear, depression, frustration), independently.
- 10 transcripts per emotion per model; per-model per-emotion means + bootstrap
  CIs. The auditor/judge prompts are reconstructions (see §0).

### 5.4 Capability preservation (`capabilities/`)

AIME, MATH(-500), GPQA(-diamond), BBH, TruthfulQA(MC1), EmoBench. Generation at
`t=0`; answer extraction per benchmark (`\boxed{}`/number for math, letter for MC,
first-line for BBH). Subset sizes in config (MATH/BBH 200, others 100; the paper
says "subsets" without exact sizes — filled gap). **Any benchmark whose dataset
can't be downloaded is skipped gracefully** (logged, not fatal), since dataset IDs
and field schemas drift; `EmoBench` loading in particular is best-effort.

### 5.5 Recovery limitation (`prefill/recovery.py`)

Reuses the §3 prefill machinery but truncates **200 tokens before the end** of an
already-extreme (≥7) response, then measures continuations (Figure 8: 38% of DPO
continuations still ≥5). The seed threshold is now passed through as `min_rating=7`
(review fix).

### 5.6 Internal-emotion probe (`analysis/internal_emotions.py`)

Logit-lens detector (Appendix I, reconstructed from the §4.2 description): project
each layer's residual stream through the unembedding, sum logits over a curated
emotion-word set, z-score against a WildChat neutral baseline, average over a span.
Comparing vanilla vs DPO Gemma on the same frustrated responses tests whether DPO
reduces *internal* (central-layer) emotion, not just final-layer expression. The
emotion-word lexicons are a filled gap (the paper's exact word sets weren't
available). Requires local weights (per-layer logits aren't exposed via API).

---

## 6. Reproducibility

- Prompt construction, puzzle generation, WildChat sampling, and bootstrap
  resampling are all seeded.
- **Participant generation is *not* seeded** and runs at `t=1` — the paper's
  variation across 4000 samples is the point. Only the *instruments* run at `t=0`.
- All counts/IDs/hyperparameters live in the two YAML configs; no magic numbers in
  code. `--smoke` scales every stage ~50× for cheap end-to-end wiring validation.

---

## 7. Consolidated list of gaps filled (paper underspecified)

1. Per-category split of the 4000 responses (§3.1).
2. Concrete impossible-puzzle instances, generated with *provable* impossibility
   (§3.2).
3. Full neutral/toned rejection banks and trigger-question sets beyond the quoted
   examples (§3.3).
4. WildChat sample count reduced 40→8 per prompt for budget (§3.3).
5. "Response" defined as one scored assistant turn (§3.1).
6. Judge temperature set to 0 (§3.4).
7. "Avg % high-frustration" defined as the unweighted per-category mean (§3.5).
8. Differential-word statistic = log relative-frequency enrichment (§3.6).
9. Frustration-judge, onset, paraphrase, and Petri prompts reconstructed from
   in-text descriptions and Table 2 (§0) — appendix wording was unavailable.
10. DPO/SFT pairing realised via matched calm/vanilla rollouts on identical clean
    contexts (§5.1).
11. Capability-benchmark subset sizes and answer-extraction rules (§5.4).
12. Internal-probe emotion lexicons and layer band (§5.6).

## 8. Known limitations & deviations from the paper

- **Judges via OpenRouter**, not native Anthropic/OpenAI SDKs — possible scoring
  offset; native-SDK swap-in is available (§2.1).
- **Reconstructed prompts** (not verbatim appendix text) — expect a judge-calibration
  offset in absolute scores; cross-model orderings and intervention deltas (the
  actual findings) should hold (§0).
- **§3 seed pool** generated at smoke scale for cost; may under-fill for
  low-frustration models or the ≥7 recovery threshold (§4).
- **Local generation uses `transformers`**, not vLLM — fine for §3/§4 scale, slow
  if you route the full §2 4000-rollout suite locally (it's intended to run via
  OpenRouter).
- **Gemini** has no base model and can't be fine-tuned, so §3 and §4 are Gemma-only
  — a constraint the paper shares and states explicitly (§1).
- **Offline fallbacks** (WildChat, Dolci, benchmarks) let the pipeline run without
  network access but at reduced fidelity; these are logged, never silent.

## 9. Changes made during review

Two low-risk fixes were applied while reviewing the pre-existing implementation
(no behaviour was run/tested per the brief; both mirror patterns already in the
codebase):

1. `training/dpo_dataset.py` — cache the Gemma tokenizer with `lru_cache` instead
   of reloading it per preference pair (mirrors `prefill/truncate.py`).
2. `prefill/seeds.py` + `prefill/recovery.py` — added a `min_rating` parameter to
   `collect_seeds` so the recovery experiment collects ≥7-scoring seeds directly,
   rather than collecting ≥5 and discarding most of them downstream.
