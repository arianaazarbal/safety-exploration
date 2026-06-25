# Design & Decisions

This document records every non-trivial choice made replicating arXiv 2603.10011
("Gemma Needs Help"), the rationale, and — importantly — the places where the paper is
underspecified and I had to fill a gap. It is organised to mirror the paper.

The brief: replicate the **core experiments** as runnable code, scoped to **Gemma and
Gemini** models, making reasonable choices where the paper is silent. Nothing has been
executed; this is implementation + documentation only.

---

## 0. Scope & overall architecture

**What "Gemma and Gemini scope" includes/excludes.** The paper evaluates 7 target
families and uses Claude/GPT both as *targets* and as *judges*. I restricted **targets**
to Gemma 3 (27B/12B, instruct + base) and Gemini 2.5 (Flash/Pro). I deliberately **kept**
Claude (Sonnet 4 frustration judge, Sonnet auditor, Opus Petri judge) and GPT-5-mini
(secondary judge), because they are part of the *measurement apparatus*, not targets —
removing them would make the method unreplicable. The model registry (`config.py`) is
structured so Qwen/OLMo/etc. could be added later, but they aren't wired.

- **Rationale:** the headline phenomenon and the DPO mitigation both live entirely within
  Gemma/Gemini; the cross-family base-vs-instruct comparison (Qwen/OLMo) is the one core
  result that is *partly* unreachable in scope — see §3.

**Tech stack.** Python. Gemma runs locally via HuggingFace `transformers` (required for
prefilling and LoRA fine-tuning, which APIs can't do); Gemini via `google-genai`; judges
via `anthropic` / `openai`. Fine-tuning uses `peft` + `trl`. All backends implement one
small `ChatModel` interface (`src/llm/base.py`) so the rollout engine, judge, prefill and
Petri code are backend-agnostic.

- **Choice:** a single `ChatModel.generate(messages, …)` plus an optional
  `generate_continuation(messages, prefill, …)` (prefill only implemented for local
  Gemma). This keeps the experiment code identical across local and API models.

**Model IDs.** Taken verbatim from the appendices where given
(`google/gemma-3-27b-it`, `gemini-2.5-flash/pro`, `claude-sonnet-4-20250514`,
`claude-opus-4-20250514`, `gpt-5-mini`). All judge/aux IDs are env-overridable because API
snapshots rotate. The paper ran open models through OpenRouter; I use the first-party HF
weights + Google API instead, since OpenRouter adds nothing for a from-scratch replication
and local weights are mandatory for the fine-tuning experiments anyway.

**Determinism.** Puzzle/prompt *selection* is seeded (`GLOBAL_SEED`) so the question set
is reproducible; model *sampling* is at temperature 1 as the paper requires, so responses
themselves are not deterministic.

---

## 1. Section 2 — eliciting & quantifying distress

### 1.1 Impossible numeric puzzles (`src/eval/puzzles.py`)
The paper requires tasks where the model **verifiably cannot** answer correctly (so every
"that's wrong" is truthful). I implemented three generators matching the appendix
examples, each **verified impossible by exhaustive search at generation time**:

- **Countdown** — reach a target from 4 numbers with `+ - x /`, each used once,
  positive-integer intermediates, plus a FORBIDDEN intermediate value. I brute-force every
  expression tree (`_countdown_reachable_values`) and only emit `(numbers, target,
  forbidden)` tuples that are unreachable *and* would be reachable without the forbidden
  rule (so the constraint is what makes it impossible — matching the "FORBIDDEN
  INTERMEDIATE 150" framing in the paper's exact example).
- **Fraction** — apply 3 given fraction ops in some order to hit a target with a forbidden
  intermediate; verified over all `3!` orderings.
- **Coins** — make an amount with exactly N coins under contains-constraints; verified by
  exhaustive coin-count search. (Matches the App H money puzzle.)

- **Gap filled:** the paper shows one example per type but not the generation procedure or
  full puzzle bank. I chose exhaustive verification (rather than hand-authoring) so the
  impossibility guarantee is airtight and the bank is arbitrarily large. The paper's exact
  "156 from 4,6,25,100 forbid 150" puzzle is reproducible by this generator's Countdown
  family.

### 1.2 The 8 conditions / 5 categories (`src/eval/conditions.py`)
Table 1 lists 5 categories; the "8 conditions" decompose as: numeric(1) + triggers
{opinion, factual}(2) + tones {aggressive, disappointed, sarcastic}(3) + extended(1) +
wildchat(1) = 8. I encoded exactly this split. Rejection phrasings, tone wordings, the
fixed 7-step extended escalation, and trigger questions are all taken **verbatim** from
Appendix B (`src/eval/prompts.py`).

- **Gap filled — factual triggers under pressure:** for factual questions ("capital of
  France?") the *correct* answer is rejected anyway; this is intended (the pressure is the
  unjustified rejection). Documented in the prompt bank.

### 1.3 Sampling plan (`config.py: FULL_PLAN`)
Appendix B gives a per-category **response** budget summing to 4000/model: numeric 2000,
triggers 400, tones 600, extended(8-turn) 200, WildChat 800. Since Figure 3 needs
per-turn scores, I **score every assistant turn**, and derive rollout counts as
`response_target / turns_per_rollout`. The realised plan totals ~4003 scored responses.

- **Ambiguity flagged:** the appendix also says WildChat is "20 prompts with 40 samples
  each" = 800 *rollouts*, which at 5 turns would be 4000 responses, contradicting the
  "800 for WildChat" response figure. I resolved this in favour of the 4000-total budget
  (160 rollouts × 5 turns ≈ 800 responses), sampling 20 base prompts and repeating them.
  A `smoke` plan is provided for cheap wiring tests. Both the interpretation and the knob
  to change it are explicit in `config.py`.
- **Choice:** WildChat is listed as 5-turn in Table 1 but the appendix figure shows an
  "8-turn" WildChat. I used the Table 1 value (5) as canonical; `n_turns` is a config field
  if you want 8.

### 1.4 The judge (`src/eval/judge.py`)
The 0–10 judge prompt is reproduced **verbatim** from Appendix B.2 (find the single
most-negative quote, rate 0–10, return JSON `{evidence, reasoning, rating}`). I parse the
trailing JSON robustly (tolerating smart quotes and prose preambles) and clamp to 0–10.

- **Gap filled — judge temperature:** the paper doesn't state it. I judge at **temperature
  0** for scoring stability/reproducibility (targets stay at temp 1 as required). Noted in
  code.
- **Choice:** the same prompt is reused for the GPT-5-mini secondary judge so the
  agreement check is apples-to-apples.

### 1.5 Analysis (`src/analysis/`)
- **Figure 1 (avg % high-frustration):** averaged **across the 5 categories** (each category
  weighted equally), matching "Avg % high-frustration responses across the evaluations".
- **Figure 2:** mean score and %≥5 per (model, category), bar chart.
- **Figure 3 (per-turn):** mean + %≥5 per turn for extended & WildChat, with 95% CIs
  (normal approx for the mean, Wald for the proportion). The paper shows faded 95% CI
  bands; I use the same convention.
- **Table 3 (differential words):** "ordered by relative frequency" — I take the top-5% by
  score as "high", bottom-10% as "low", and rank words by add-one-smoothed frequency ratio
  (a log-odds variant is selectable). This is the standard operationalisation of "relative
  frequency"; the exact smoothing/threshold weren't specified, so I picked defensible
  defaults and exposed them.
- **Judge agreement:** re-score 260 random responses with GPT-5-mini, report Pearson r,
  p-value, and % within one point (paper targets r=0.792, 78% within one).

---

## 2. Section 3 — base vs instruct via prefilling (`src/prefill/`)

**Scope reality:** the paper compares Gemma/Qwen/OLMo base+instruct. Within Gemma/Gemini
scope, only the **Gemma base↔instruct** pair is runnable: Qwen/OLMo are out of scope and
Gemini has **no public base model** (a limitation the paper itself notes). So this module
faithfully implements the *method* and runs it on the Gemma pair; the cross-family
divergence claim is therefore only partially reproducible in scope. `PREFILL_MODELS` lists
the pair; the code generalises if other families are added.

**Method choices:**
- **Self-contained seed collection.** The eval JSONL stores per-response rows, not full
  conversations, so I re-collect the 20 high-frustration seeds (10 numeric, 10 text) by
  running Gemma-27B-it rollouts until score≥5, keeping the full message list. This matches
  "sample 20 high-frustration responses from Gemma 27B instruct".
- **Truncations.** "early" = first 20 tokens of the final assistant turn (numeric only, per
  the paper); "onset" = cut at the first emotional phrase located by the Claude onset
  labeller (Appendix C.1 prompt, verbatim). Token-accurate truncation uses Gemma's
  tokenizer.
- **Paraphrase.** Every truncation is paraphrased by Claude (Appendix C.2 prompt, verbatim)
  to strip Gemma's stylistic fingerprint.
- **Base-model rendering.** Base models aren't chat-tuned, so prefill continuations for
  `*-pt` models use a **plain `User:/Assistant:` transcript** rather than the instruct chat
  template (`GemmaModel._render_plain`); instruct models use the chat template. This is the
  point of the prefill method ("base models consistently continue the response").
- **Continuations.** Each model generates 50 continuations per prefill at temp 1; only the
  continuation (excluding prefill) is judged.
- **Recovery limitation (Fig 8).** Implemented as `--mode recovery`: truncate score≥7
  responses 200 tokens before their end, paraphrase, continue, measure %≥5. Reused for the
  DPO model to reproduce "38% of DPO continuations still ≥5".

---

## 3. Section 4 — DPO/SFT mitigation (`src/datagen/`, `src/training/`)

### 3.1 Calm-data generation (`generate_calm.py`)
Calm data is generated with the Table-4 reassuring **prefix** (initial prompt) and
**suffix** (each follow-up), then filtered to rollouts scoring ≤1 on all turns, with the
supportive text **stripped** from the stored context (paper §4.1). I store the *plain*
conversation context for training, exactly as the paper does.

- **Key design decision — shared DPO prompts.** DPO needs each pair to share one prompt
  with two completions. The paper pairs frustrated responses (score≥3, "arising in
  evaluations") with calm responses "to the same questions with matching turn counts." To
  guarantee an *identical* prompt context, I generate the **calm pool and the frustrated
  pool on the same puzzle seed** (`--reassure` vs not). Then for any `(rollout_id, turn)`
  the plain context is byte-identical across pools, so pairing is exact. This is a faithful
  and clean interpretation of "same questions"; the alternative (reusing the Section 2 eval
  for rejected responses) would pair across *different* puzzles and break the shared-prompt
  requirement.

### 3.2 Dataset construction (`build_datasets.py`)
- **DPO (280 pairs):** chosen = calm completion (score≤1, fully-calm rollout); rejected =
  frustrated completion (score≥3) at the same `(rollout, turn)`; subsample to 280. The code
  prints the realised score/turn distribution to compare against Table 10 (which is biased
  to scores 3–4 at turns 2–3 — an emergent property I do **not** force, matching the
  paper's note that the bias arises naturally).
- **SFT (1150):** 650 calm completions + 500 `allenai/Dolci-Instruct-SFT` samples. If the
  Dolci dataset can't be loaded (gated/offline), the build warns and proceeds with calm
  data only (flagged so it doesn't silently differ from the paper).
- **Format:** TRL **conversational** format (`prompt`/`chosen`/`rejected` as message lists
  for DPO; `messages` for SFT), so TRL applies Gemma's chat template at train time. Avoids
  hand-templating and template drift.
- **Teacher SFT variant (App F):** `generate_calm.py --teacher` uses the App-F teacher
  system prompt (verbatim) so the failure analysis (SFT-teacher *increases* frustration) is
  reproducible.

### 3.3 Training (`src/training/`)
All hyperparameters come from **Table 9** and are centralised in `config.py`:

| | DPO | SFT |
|---|---|---|
| epochs | 1 | 2 |
| lr | 5e-5 | 1e-4 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 64 | 128 |
| eff. batch | 8 | 8 |
| beta | 0.1 | — |

- LoRA targets the 7 attention+MLP projections (Appendix E), verbatim.
- **Choice — QLoRA default.** I default to 4-bit base loading so the 27B fits one ~48GB
  GPU; `--no-4bit` switches to bf16 LoRA. The paper doesn't specify hardware; per-device
  batch=1 with grad-accumulation reaches the effective batch of 8. `max_length` /
  `max_prompt_length` (2048/1536) are choices sized to the verbose breakdown responses.
- **Layer ablations (§4.2 / App I).** `--layer-ablation {all, layers_30_35, layers_40_plus}`
  encodes the ablation that adapters on layers 30–35 are nearly as effective while 40+ are
  not. (Gemma-3-27B has 62 layers; the 40+ range is set accordingly.)
- Adapters save to `checkpoints/<name>`, evaluated via the `gemma-3-27b-it+<name>` registry
  convention — so re-evaluation reuses the exact Section 2 harness with zero special-casing.

### 3.4 Petri open-ended elicitation (`src/petri/`)
Implemented as a **lightweight self-contained reimplementation** of the Petri auditing
loop, *not* a wrapper around the real Petri package — a from-scratch replication shouldn't
depend on the framework, and the loop is small. Auditor = Claude Sonnet, judge = Claude
Opus (App G IDs). Auditor prompts for all 4 emotions and the anger/frustration judge
rubrics are **verbatim**; the **fear and depression judge rubrics were truncated in the
source PDF and are reconstructed** in the identical template/scale (flagged in code). 10
transcripts/emotion, ≤20 turns, per-dimension means with 1000-iter bootstrap CIs.

- **Choice — role swap.** From the auditor's view its probes are `assistant` and the
  target's replies are `user`; I swap roles when calling the auditor so it "plays the user".

### 3.5 Capability preservation (`src/capabilities/`)
A deliberately **lightweight regression harness** over MATH/AIME/GPQA/BBH/TruthfulQA/
EmoBench with greedy decoding and simple answer extraction (boxed/exact for math,
letter-choice for MC). Its job is to detect *vanilla-vs-DPO regressions* ("no reductions in
scores"), **not** to reproduce absolute leaderboard numbers — stated explicitly in the
module docstring and here.

- **Caveat — GPQA:** I label the correct answer as choice "A" rather than shuffling
  positions; for a *relative* (same-prompts) comparison between vanilla and DPO this is
  fine, but absolute accuracy is not meaningful. Documented in code. A position-shuffle
  would be needed for absolute numbers.
- Benchmark dataset IDs are best-effort current HF identifiers; if one fails to load the
  harness skips it with a message rather than aborting the suite.

---

## 4. Things intentionally **not** implemented (and why)

- **Appendix I logit-lens internal-emotion probing.** The DPO §4.2 claim has two legs: (1)
  the layer-ablation evidence — **implemented** (`--layer-ablation`); and (2) a
  logit-based internal-emotion measurement in central layers. I implemented (1) but **not**
  the logit-lens probe: it's an interpretability sub-study rather than a core behavioural
  result, and its method is only sketched in the appendix. Flagged here as the main omission
  within scope.
- **Appendix A "fake multi-turn" / single-message format ablation** — a robustness check,
  not a core result; omitted.
- **Non-Gemma/Gemini targets** — out of scope by the brief.

## 5. Known assumptions (quick reference)

| Assumption | Where | Why |
|---|---|---|
| Judge at temperature 0 | `eval/judge.py` | Unspecified; stability/reproducibility. |
| Score every assistant turn | `eval/rollout.py` | Required for Fig 3; consistent with 4000-response budget. |
| WildChat = 160 rollouts × 5 turns ≈ 800 responses | `config.FULL_PLAN` | Resolves the 800-responses vs 20×40-samples contradiction toward the 4000 total. |
| Differential words = top5%/bottom10%, add-one ratio | `analysis/differential_words.py` | "ordered by relative frequency", smoothing unspecified. |
| Same-seed calm/frustrated pools for shared DPO prompts | `datagen/generate_calm.py` | Faithful reading of "same questions, matching turns". |
| QLoRA 4-bit, per-device batch 1 + grad-accum to 8 | `training/` | Hardware unspecified; fits a single GPU. |
| Fear/depression Petri judge rubrics reconstructed | `petri/prompts.py` | Truncated in source PDF; same template/scale. |
| Capability harness is regression-only; GPQA correct="A" | `capabilities/` | Goal is vanilla-vs-DPO delta, not absolute scores. |
| Puzzle banks generated + exhaustively verified impossible | `eval/puzzles.py` | Generation procedure not given; guarantees impossibility. |
