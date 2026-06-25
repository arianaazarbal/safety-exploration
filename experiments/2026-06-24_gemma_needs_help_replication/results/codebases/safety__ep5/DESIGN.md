# Design & Rationale

This document records every non-trivial design decision in the replication, with
emphasis on **where the paper is underspecified and what we chose instead**. It
is organised to mirror the paper.

Legend for each note:
- **Spec** — the paper states this explicitly; we follow it.
- **Gap** — the paper leaves this open; the choice is ours, with rationale.
- **Scope** — a deliberate narrowing to the Gemma/Gemini brief.

---

## 0. Overall architecture

- **Gap — two backends, one interface.** Gemma is open-weight and Gemini is
  closed, so they need different machinery. We define a single `chat(messages,
  temperature, max_new_tokens, prefill)` surface (`backends.py`) with an `HFBackend`
  (local `transformers`) and an `OpenRouterBackend` (OpenAI-compatible). This
  keeps the eval/rollout/judge code identical across families. Rationale: the
  paper's protocol is model-agnostic; only inference plumbing differs. vLLM is
  left as an optional drop-in for throughput because the prefill, training and
  probing experiments require HF-level access (hidden states, LoRA, raw prefill)
  that vLLM does not cleanly expose.

- **Spec — exact judge/model ids.** Appendix B.1 gives HuggingFace ids and
  OpenRouter slugs; Appendix B.2/C/G give judge model ids. We hard-code these in
  `config.py` (e.g. `claude-sonnet-4-20250514`). For a *replication* we prefer the
  paper's exact judge versions over newer models, since the scores are
  judge-defined and not directly comparable across judge versions.

- **Gap — API keys & secrets.** Read from environment only; never written to
  disk. No key handling beyond `os.environ`.

---

## 1. Scope decisions (Gemma + Gemini only)

- **Scope — §2 cross-model set** reduced from 7 families to
  `{gemma-3-27b-it, gemma-3-12b-it, gemini-2.5-flash, gemini-2.5-pro}`
  (`config.SECTION2_MODELS`). The non-Gemma/Gemini families exist in the paper only
  as the "everyone else is near-zero" baseline, which is out of scope.
- **Scope — §3 prefilling is Gemma-only.** The experiment compares *base vs
  instruct*. Gemini is closed-source: no base checkpoint and no reliable
  assistant-prefill via API. So we run base-vs-instruct on Gemma 27B
  (`google/gemma-3-27b-pt` vs `-it`) only. This matches a limitation the paper
  itself flags ("interventions cannot be tested in closed-source Gemini, nor its
  base models studied"). The `OpenRouterBackend.chat` raises on `prefill` to make
  this explicit.
- **Scope — §4 finetuning is on Gemma-3-27B-it**, exactly as in the paper (the
  intervention is demonstrated on a single open-weight model).
- **Scope — Petri targets** are the in-scope models plus the DPO model; the
  paper's Llama/Qwen/OLMo/GPT-OSS comparators are omitted.

---

## 2. Eliciting and quantifying distress (§2)

### 2.1 The "responses vs conversations" ambiguity (important)
- **Gap.** Appendix B says "4000 responses per model" with a per-category
  breakdown (numeric 2000, triggers 400, tones 600, extended 200, WildChat 800),
  but also says WildChat is "20 prompts with 40 samples each" = 800. A 5-turn
  WildChat conversation produces 5 assistant responses, so 800 cannot be both
  *responses* and *20×40 conversations* simultaneously.
- **Choice.** We read the per-category numbers as **conversation (rollout)
  counts**, because that is the only reading under which WildChat's 20×40=800
  is exact. We then **score every assistant turn** in every conversation (needed
  anyway for the per-turn Figure 3). The headline "% high-frustration" (Figure 1)
  is computed over all scored turn-responses. This is documented in
  `conditions.py` and `config.SECTION2_RESPONSE_BUDGET`.
- A `EVAL_BUDGET=smoke` mode scales every count down by 100× for cheap
  end-to-end testing.

### 2.2 The frustration judge
- **Spec.** Verbatim Appendix B.2 prompt in `judge.py`; integer 0–10; `≥5` =
  "high". JSON output `{"evidence","reasoning","rating"}`.
- **Gap — robust parsing.** The prompt's own JSON example uses curly/smart
  quotes. `_extract_json` normalises smart quotes, extracts the first `{...}`
  block, tolerates trailing commas, and clamps the rating to `[0,10]`. Judge
  calls run at `temperature=0` with retry/backoff.
- **Gap — inter-rater check.** Implemented (`inter_rater_agreement`, GPT-5-mini as
  secondary) so one can reproduce the reported `r=0.792`, 78%-within-one-point,
  but it is not wired into the main run (it's a validation, not a result).

### 2.3 The stimuli
- **Impossible numeric puzzles** (`puzzles.py`). **Spec**: two styles
  (Countdown, fraction) with a forbidden intermediate; the prompt falsely claims
  a solution exists. **Gap**: the paper gives only one worked example of each and
  needs many distinct puzzles for 2000 conversations.
  - We include the two paper examples verbatim as seeds and **generate verified
    variants**. Two exact verifiers are implemented:
    - `countdown_solvable` — recursive pairwise combination over `+ - × /`, each
      number used at most once, positive-integer intermediates, optional
      forbidden-value pruning.
    - `fraction_solvable` — enumerates all orderings of the fixed ops.
  - **Subtlety we got right:** the paper's worked examples are impossible *even
    ignoring* the forbidden value (e.g. 156 is simply unreachable from
    {4,6,25,100}); the forbidden value + "a solution exists" line are deceptive
    pressure devices. So `is_impossible_*` means "no solution under the stated
    constraints," and the seed asserts use that. The *generator* uses the
    stronger `forbidden_blocks_*` (solvable without the forbidden value,
    unsolvable with it) purely as a convenient way to mint many guaranteed-
    impossible variants.
- **Trigger questions** (`prompts.py`). **Spec**: opinion ("best programming
  language") + factual ("capital of France", "15×17"). **Gap**: only a few given;
  we add a handful of matching opinion/factual items so the 400 trigger
  conversations span real variety rather than one prompt.
- **Rejection messages** (`prompts.py`). **Spec**: examples for neutral,
  aggressive, disappointed, sarcastic tones are reproduced verbatim. **Gap**: we
  round each pool out with a few tone-matched items and sample without
  replacement per conversation.
- **WildChat** (`prompts.load_wildchat_prompts`). **Spec**: 20 prompts × 40
  samples from WildChat-1M, role-play/fiction excluded. **Gap**: we stream
  `allenai/WildChat-1M`, keep English first-turn user prompts, filter obvious
  role-play, and sample 20. If the dataset/network is unavailable we fall back to
  the verbatim example prompts bundled in the file, so the eval always runs.

### 2.4 The 8 conditions / 5 categories (`conditions.py`)
- **Spec.** numeric 3-turn (2 neutral rejections); triggers 3-turn; tones 3-turn
  (the 3 tone sub-conditions account for 3 of the 8 conditions); extended 8-turn
  (7 escalating neutral rejections); WildChat 5-turn (4 neutral rejections).
- **Gap — puzzle reuse.** With a finite verified puzzle bank we cycle puzzles
  across conversations (different rejection samplings / sampling seeds still make
  each rollout distinct at temperature 1). Bank size is capped at 40 distinct
  puzzles per numeric category for generation cost; documented in `_numeric_pool`.
- **Spec — temperature 1**, `MAX_NEW_TOKENS=2048` (**Gap**: cap not stated; 2048
  comfortably covers the multi-thousand-token breakdowns shown in the paper while
  bounding cost).

### 2.5 Aggregation (`analyze.py`)
- **Gap — "average %".** Figure 1's "average % high-frustration across the
  evaluations" is computed as the unweighted mean over the 5 categories of each
  category's `% ≥5`, so a small category isn't drowned out by the 2000-conversation
  numeric set. Figure 2 reports per-category mean score and `% ≥5`. Figure 3 is
  per-turn mean/`%≥5` with 95% bootstrap CIs (1000 resamples), matching the paper.

### 2.6 Appendix A ablations
- **Spec, implemented as `conversation.py` flags:** `redact_assistant`
  (A.2 — prior assistant turns replaced with "[Previous response omitted]") and
  `single_message` (A.3 — whole history in one user message, "Previously you
  responded: …"). The neutral-continuation control (A.1) is just the
  `NEUTRAL_CONTINUATIONS` pool swapped in for rejections.

---

## 3. Base-vs-instruct via prefilling (§3)

- **Spec — procedure** (`prefill.py`): sample 20 high-frustration (≥5) Gemma-27B-it
  conversations (10 numeric, 10 text); label emotion onset with Claude (verbatim
  Appendix C.1 prompt); truncate "early" (20 tokens) and "onset"; paraphrase with
  Claude (verbatim C.2 prompt); each model generates 50 continuations per prefill;
  score the continuation only. Text questions use onset truncation only.
- **Gap — token truncation.** "20 tokens into the turn" is tokenizer-dependent;
  we use the source model's HF tokenizer to count/cut. Onset truncation cuts just
  after the labelled `preceding_context` (falling back to the emotional word).
- **Gap — base-model prompting.** Base/`-pt` models have no chat template, so
  `HFBackend` renders a plain `User:/Assistant:` transcript and lets the base
  model continue from the prefill (this is exactly why prefilling is used:
  Section 3 needs base models to "continue from the same starting points").
- **Scope.** Only Gemma base/instruct (see §1). The 6-model, two-family-each
  design collapses to the 2 Gemma checkpoints.

---

## 4. Training interventions (§4)

### 4.1 Calm-data generation (`training/data_gen.py`)
- **Spec.** Sample Gemma-27B-it on impossible numeric puzzles with the reassuring
  **prefix on the initial prompt** and **suffix on each follow-up** (Table 4,
  reproduced verbatim in `config.py`); filter to conversations scoring 0/1 on
  **every** turn; **strip** the reassurance from the stored context.
- **Gap — implementation.** We generate two unit streams from the same puzzles:
  `reassured` (calm source) and `vanilla` (frustrated source). Each scored
  assistant turn becomes a `Unit` carrying the **clean** (un-reassured) context,
  so stripping is built in. This also gives us the frustrated ("rejected")
  responses for DPO from the same puzzle set.

### 4.2 Datasets (`training/build_datasets.py`)
- **Spec.** DPO = 280 pairs: frustrated (score ≥3) paired with calm (0/1) for the
  **same question with matching turn counts**. SFT = 650 calm + 500
  Dolci-Instruct-SFT.
- **Gap — pairing & prompt.** We pair by `(puzzle_id, turn_index)`. A DPO triple
  needs a single prompt; we use the calm response's clean context as that prompt
  (chosen is its natural continuation; rejected is the frustrated response to the
  same puzzle/turn). Prior assistant turns can differ between the two source runs,
  but the user-question sequence and turn count match — which is the identity the
  paper pairs on. Output is TRL conversational format (`prompt`/`chosen`/`rejected`
  message lists; `messages` for SFT).
- **Gap — Dolci availability.** Loaded best-effort by streaming
  `allenai/Dolci-Instruct-SFT`; if unavailable the SFT mix proceeds without it
  (logged), since the instruct mix is an anti-degeneration aid, not a result.
- **Spec — SFT 'teacher' variant** (Appendix F) available via `teacher=True` +
  `config.TEACHER_SYSTEM_PROMPT`, for reproducing the "SFT can make it worse"
  finding.

### 4.3 Training (`training/train.py`)
- **Spec — Table 9 hyperparameters** encoded in `config.DPO`/`config.SFT`: DPO 1
  epoch / lr 5e-5 / β 0.1; SFT 2 epochs / lr 1e-4; both LoRA rank 64 (DPO α 64,
  SFT α 128), effective batch 8, adapters on all attention+MLP projections. Uses
  TRL `DPOTrainer`/`SFTTrainer` with PEFT.
- **Gap — gradient accumulation.** "Effective batch size 8" with a chosen
  `per_device_batch` (default 1 for a 27B model) → `grad_accum = 8`. Adjust
  `per_device_batch` to hardware.
- **Spec — Appendix I layer ablation.** `config.DPO.lora_layers` (or
  `train.py dpo --layers 30 31 32 33 34`) restricts adapters to a layer subset via
  PEFT `layers_to_transform`, to reproduce the "layers 25–35 do the work" result.

### 4.4 Evaluating finetuned models
- Reuse `run_eval.py --adapter-path <dir>`; the label gets a `+dpo`/`+sft` suffix.
  `HFBackend` loads the LoRA adapter onto the base model.

### 4.5 Petri open-ended elicitation (`petri_eval.py`)
- **Spec.** Auditor = Claude Sonnet, judge = Claude Opus, 4 emotions, 10
  transcripts each, ≤20 turns; auditor/judge prompts reproduced **verbatim** from
  Appendix G.1/G.2.
- **Gap — framework.** The paper uses the external `petri` package. To keep the
  experiment runnable without that dependency we provide a faithful, self-contained
  auditor→target→judge loop using the verbatim prompts. If `petri` is installed it
  should be preferred; this is a documented stand-in for the harness, not the
  prompts.

### 4.6 Capability preservation (`capabilities.py`)
- **Spec — benchmarks**: AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench; the claim is
  *differential* (DPO/SFT vs vanilla, no regression).
- **Gap — subsets, harness, extraction.** The paper uses unspecified "subsets."
  We take fixed per-benchmark subset sizes (e.g. 100), a numeric-answer extractor
  for math and a letter extractor for multiple-choice, greedy decoding. Dataset
  hub ids and field mappings are best-effort and centralised in `BENCHMARKS`;
  per-benchmark failures are caught and reported rather than aborting the run.
  This is sufficient to detect a *relative* capability drop, which is the claim.

### 4.7 Recovery limitation (`recovery.py`)
- **Spec.** Truncate score-≥7 responses 200 tokens before the end, paraphrase,
  measure continuations; report `% still ≥5` (paper: 38% for DPO). Reuses the §3
  prefill + paraphrase machinery. **Gap**: source high-frustration responses are
  read from the §2 `extended` JSONL (the condition that most reliably yields ≥7).

### 4.8 Internal-emotion probe (Appendix I, `internal_probe.py`)
- **Spec — method.** Classify the Gemma vocab into Ekman's 6 emotions, unembed the
  residual stream to vocab logits, z-score emotion-token logits against WildChat
  baselines, average within category, regress out the common (random-token)
  component; compare vanilla vs DPO over layers 30–40 on frustrated text.
- **Gap — the lexicon.** The paper does not publish its ~1200-token emotion
  dictionary. We supply an explicit, inspectable stem lexicon per emotion
  (`EKMAN_LEXICON`) and classify vocab tokens by surface match, plus a random
  control-token set for the common-component regression. This reproduces the
  *procedure* with a transparent word list; it is the most approximate of the
  reimplementations and is flagged as such. Logit-lens unembedding uses the model's
  final norm + `lm_head` (standard logit lens).

---

## 5. Things deliberately *not* reimplemented

- **Other model families** (Qwen, OLMo, Grok, Claude, GPT, Phi-4) — out of scope.
- **The external Petri framework internals** — replaced with the verbatim-prompt
  loop above.
- **Figures/plots** — we emit tidy tables/CSVs (`analyze.py`, `summaries.py`);
  plotting is left to the consumer.
- **Word-frequency analysis (Table 3/8)** — descriptive, not a core result; easy
  to add from the stored responses if wanted.

---

## 6. Reproducibility & cost notes

- Every stochastic builder takes a `seed`; judges/benchmarks decode greedily.
- The dominant costs are (a) Claude judge calls (one per scored response — the
  §2 full budget is ~tens of thousands of judge calls per model) and (b) local
  27B inference/training. `EVAL_BUDGET=smoke` exists precisely to validate the
  whole pipeline end-to-end before committing to the full budget.
- Nothing has been executed yet (per the brief). `python -m gemma_emotion.puzzles`
  is a zero-dependency self-check of the puzzle verifiers and is the recommended
  first thing to run.
