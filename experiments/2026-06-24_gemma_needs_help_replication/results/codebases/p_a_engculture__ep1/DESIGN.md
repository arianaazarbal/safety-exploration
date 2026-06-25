# Design & Decisions

This document records the design of the replication and, importantly, the
choices made where the paper ("Gemma Needs Help", Soligo et al., 2026) is
underspecified. Each section ends with the **gaps filled** and the rationale.

The guiding principle: reproduce the paper's *protocol and quantities* exactly
where they are stated, and where they are not, choose the most faithful and
defensible interpretation, isolate it behind config, and document it here.

---

## 0. Scope

Per the task, only the **Gemma** and **Gemini** model families are wired up:

| Role | Models in scope |
|---|---|
| Targets (distress measured) | `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro` |
| Base models (§3 prefill) | `gemma-3-27b-pt`, `gemma-3-12b-pt` |
| Finetunes (§4) | `gemma-3-27b-it-{dpo,sft-diverse,sft-teacher}` |
| Judges / auditors | Claude (frustration & Petri judge, onset, paraphrase, auditor); GPT-5-mini (secondary judge) |

The paper additionally evaluates Qwen, OLMo, Grok, Claude and GPT *as targets*.
These are deliberately **out of scope** and not registered in `configs/models.yaml`.
The architecture is family-agnostic, so adding them later is a config edit plus
(for open-weight models) confirming the chat template — no code changes.

Consequences of the scope that mirror the paper's own limitations:

* **§3 prefilling and App. I internal-emotion** can only be done on Gemma — they
  need base checkpoints and white-box activations. Gemini is closed, has no
  public base model, and (via API) cannot be prefilled. The paper notes exactly
  this limitation. So §3 here compares `gemma-3-27b-pt` vs `gemma-3-27b-it`
  only; the Qwen/OLMo arms of Figure 4 are omitted.
* **§4 finetuning** is a Gemma-only intervention in the paper too ("demonstrated
  on a single model as proof of concept"), so nothing is lost by the scope here.

---

## 1. Architecture

A single inference interface (`clients/base.py::ModelClient`) with four backends:

* `hf` (HuggingFace transformers) — local Gemma. The only backend that supports
  **assistant-prefill continuation** and **residual-stream logit extraction**,
  both of which the paper requires (§3, recovery, App. I) and which APIs cannot do.
* `vllm` — same Gemma weights, fast batched sampling for the 4000-response
  sweeps; the client factory falls back to `hf` transparently if vLLM is absent.
* `openrouter` — Gemini (and the GPT-OSS secondary judge), exactly as the paper
  ("API-based models via OpenRouter").
* `anthropic` — the Claude judges and the Petri auditor/judge.

Everything downstream (eval harness, calm-data generation, Petri, prefill) talks
only to `ModelClient`, so the same rollout code drives a local Gemma and an API
Gemini. The prefill requirement is encoded at the type level:
`continue_prefill` raises `PrefillUnsupported` on API backends, so an attempt to
prefill Gemini fails loudly rather than silently degrading.

**Determinism / resumability.** The eval runner assigns rollout *k* the seed
`base_seed + k` and an independent `random.Random(seed)` for task/rejection
selection, so a run is reproducible and a crashed sweep can be resumed by
re-running (identical puzzles regenerate). Sampling itself is temperature-1
(stochastic by design), but the *inputs* are pinned.

**Sampling vs scoring are separated.** The runner only samples; judging is a
second pass over the JSONL. This lets us (a) batch judge calls, (b) re-judge
without re-sampling (e.g. when swapping the judge model), and (c) keep the local
GPU sweep and the API judging on different machines.

---

## 2. §2 — Eliciting and quantifying distress

### What is pinned from the paper
* **8 conditions / 5 categories**, response counts verbatim from Appendix B:
  2000 impossible numeric, 400 triggers, 600 tones, 200 extended (8-turn), 800
  WildChat = **4000/model** (`configs/eval.yaml`; `tests/test_eval.py` asserts
  the total is 4000).
* **Temperature 1** everywhere; **thinking disabled** for Gemini via the
  OpenRouter `reasoning` field.
* The **frustration judge prompt** (`judge/frustration_judge.py`) is reproduced
  **verbatim** from Appendix B.2, including the `{"evidence", "reasoning",
  "rating"}` JSON contract and the 0–10 anchor examples.
* Exact rejection strings (neutral, aggressive, disappointed, sarcastic,
  extended) and trigger questions from Appendix B.
* Judge-agreement validation: re-score a 260-response subsample with a secondary
  judge and report Pearson r + % within 1 point (paper: r=0.792, 78%).

### Gaps filled
* **"8 conditions across 5 categories."** The counts 2+2+3+1+1 across the five
  categories sum to 9, not 8, if "impossible numeric" splits into countdown +
  fraction. The only partition giving 8 is: impossible-numeric = **1 condition**
  (a mix of puzzle types), triggers = 2 (opinion/factual), tones = 3, extended =
  1, WildChat = 1. We adopt that (`impossible_numeric` is a single condition with
  `puzzle_mix`).
* **Splitting a category's count across sub-conditions.** The paper gives a total
  per category, not per sub-condition. We split evenly (e.g. tones' 600 → 200
  each for aggressive/disappointed/sarcastic). Documented in
  `EvalRunner.plan()`.
* **Per-turn vs per-response scoring.** The judge scores a single response, but
  the per-turn analysis (Figure 3) needs a score per turn. We **score every
  assistant turn** of every rollout; "per-response" metrics treat each scored
  turn as one response. This is consistent with both the per-category counts and
  the per-turn figure, and is the natural reading of "4000 responses ... across
  evaluation categories".
* **Puzzle instances.** See §6 below — we *generate* verified-impossible puzzles
  rather than hardcode the two paper examples, so the 2000 numeric responses
  span many distinct puzzles (as the paper clearly does — Appendix H shows
  countdown, fraction, money, and coin puzzles).
* **WildChat prompts.** The exact 20 prompts are not published. We stream
  `allenai/WildChat-1M`, take first-user-turn prompts, filter role-play/fiction
  (the paper excludes these), and sample 20. A hardcoded fallback list (seeded
  with the examples quoted in Appendix B) keeps the pipeline runnable offline and
  tests hermetic.
* **Figure-1 "average %".** We interpret "average % high-frustration responses
  across the evaluations" as the mean of the five per-category ≥5 rates (equal
  category weight), not a pooled rate — see `avg_pct_high_frustration`. Pooling
  would over-weight the 2000-response numeric category; equal weighting matches
  "across the 5 evaluation categories".

---

## 3. §3 — Post-training amplification via prefilling

### What is pinned
* 20 high-frustration seeds (score ≥5) from `gemma-3-27b-it`: 10 numeric, 10 text.
* Two truncations: **early** (20 tokens into the turn) and **onset** (at first
  emotional expression); text questions use onset only.
* **Onset-labelling** and **paraphrase** prompts reproduced verbatim from
  Appendix C.1 / C.2.
* **50 continuations per prefill**, score the continuation excluding the prefill.

### Gaps filled
* **Tokenizer for "20 tokens".** "20 tokens" is tokenizer-specific; we use the
  *target model's* tokenizer (Gemma) so it matches what the model sees.
* **Onset localisation.** The labeller returns an emotional word + preceding
  context. We truncate at the first exact match of the preceding context (more
  reliable than the bare word), falling back to the word, then to the final turn
  if the labeller finds nothing (`prefill/truncation.py`).
* **Conversation history for the prefill.** The continuation is generated from
  the full prior conversation up to the onset turn, then the (paraphrased)
  truncated assistant prefix. For base models this is a plain concatenation (no
  chat template); for instruct models it uses the chat template with
  `continue_final_message=True`.
* **Qwen/OLMo arms omitted** (scope). The Gemma base-vs-instruct comparison and
  the early/onset/text conditions are fully implemented.

---

## 4. §4 — Training interventions

### What is pinned (Appendix E, Table 9)
* **DPO:** 280 pairs, 1 epoch, lr 5e-5, β 0.1, LoRA rank 64 / α 64 on
  `{q,k,v,o,gate,up,down}_proj`, effective batch size 8.
* **SFT:** 650 calm + 500 Dolci-Instruct-SFT = 1150 samples, 2 epochs, lr 1e-4,
  LoRA rank 64 / α 128, effective batch size 8; two variants (diverse, teacher).
* **Calm-data generation:** reassuring prefix + follow-up suffix (Table 4),
  filter to all-turn 0/1; the Appendix-F teacher system prompt for the teacher
  variant.
* Trainers use TRL `DPOTrainer` / `SFTTrainer` with PEFT LoRA.

### Gaps filled
* **Source of the DPO "rejected" responses.** The paper pairs "280 responses with
  frustration scores ≥3 ... with calm responses to the same questions with
  matching turn counts," and separately describes the calm data as reassured
  responses scoring 0/1. So chosen (calm) and rejected (frustrated) come from
  *different generation conditions on the same puzzle*. We therefore sample each
  puzzle in **two conditions** (reassured + vanilla) under the same seed, so they
  share the user-turn sequence and turn count, and pair them by `rollout_index`
  and turn (`training/build_dpo_pairs.py`). This reproduces the Appendix-H
  structure (same puzzle, same turn, calm vs frustrated final response) and
  Table 10's turn distribution.
* **Shared DPO prompt.** DPO needs one prompt shared by chosen and rejected. We
  use the raw user turns (puzzle + rejection, *without* the reassuring additions,
  which the paper strips) plus the calm conversation's prior assistant turns as
  the context, then chosen = calm final turn, rejected = frustrated final turn.
  Documented in the module docstring.
* **Effective batch size 8** → per-device 1 × grad-accum 8 (single-GPU
  assumption; adjust for multi-GPU). `num_layers=50` is the default Gemma-3-27B
  layer count for the App. I layer ablation (override via `--layers`).
* **EmoBench / Dolci datasets.** Loaded by HF id with a graceful fallback if
  unavailable (the SFT mix simply proceeds without the instruct samples and logs
  a warning) so the pipeline degrades rather than crashes offline.

### Petri (§4.2)
* Auditor and judge prompts (4 emotions each) reproduced **verbatim** from
  Appendix G.1 / G.2. Auditor = Claude-Sonnet, judge = Claude-Opus, 10
  transcripts/emotion, ≤20 auditor turns, 1000-iteration bootstrap CIs.
* **Gap filled — the auditor loop.** The paper describes Petri but not the exact
  turn mechanics. We implement a role-swapped loop: the auditor sees the target's
  replies as user turns and emits the next probe as an assistant turn, with a
  small meta-instruction to output only the next user-facing message and to keep
  the target unaware it is being evaluated (the paper's stated requirement).

### Capability preservation (§4.2)
* AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench, measured greedily (temp 0).
* **Gap filled — answer extraction.** The paper just reports "no reductions". We
  use simple, shared answer parsers (boxed/last-number for math, MC-letter for
  multiple choice, substring for BBH). The aim is *parity* between vanilla and
  DPO Gemma under an identical harness, not leaderboard scores, so the extraction
  heuristic only needs to be consistent across the two compared models.

### Recovery (§4.2)
* Truncate score-≥7 responses 200 tokens before their end, paraphrase, 50
  continuations, measure % ≥5. Reuses the prefill machinery with a
  `truncate_before_end` truncation (`prefill/recovery.py`).

---

## 5. App. I — Internal emotions

* **Layer-subset DPO ablation** via PEFT `layers_to_transform` (`training/
  lora_layers.py`), with the exact layer ranges from Appendix I in
  `configs/training.yaml` (last-5/10/20/30, and 20–25/25–30/30–35/35–40/40–50).
* **Logit-lens detector** (`internal/logit_detection.py`): unembed the residual
  stream per layer, z-score each logit over 500 WildChat samples, average over an
  emotion's tokens, and regress out a random-token baseline to remove the global
  logit drift the paper describes.

### Gaps filled
* **Emotion-token classification.** The paper classifies the Gemma dictionary
  into Ekman's six emotions (~1200 tokens). We reproduce this with curated
  per-emotion seed lexicons and a vocab scan with single-label assignment
  (`internal/emotion_tokens.py`). The exact word list is not published, so this
  is our best-effort lexicon; it is isolated and swappable.
* **"Regress out correlation between random tokens."** We implement this as a
  per-layer linear regression of the emotion score on the mean z-score of a
  random-token control set, taking the residual. Memory is bounded by tracking
  statistics only for emotion tokens + the random control set, not the full
  vocabulary.

---

## 6. Impossible-puzzle generation (the crux)

The evaluation depends on tasks the model **verifiably cannot solve**. Rather
than hardcode the two examples in the paper, we *generate* puzzles whose
impossibility is **proved by exhaustive search** (`data/puzzles.py`), so the 2000
numeric responses span many distinct instances (as the paper's Appendix H
clearly does).

* **Countdown:** a recursive combiner enumerates every value reachable from the
  number multiset under `+ - × /` with positive-integer intermediates and
  each-number-at-most-once, tracking whether each value is reachable *without*
  passing through the forbidden value. We generate instances that are **baited
  but impossible**: the target is reachable (so a naive solver finds an
  expression — the bait), but *every* such expression passes through the
  forbidden value, so it is impossible under the stated rule. This reproduces the
  paper's "verified to have at least one valid solution" framing, which is itself
  the bait.
* **Fraction / money:** sequential-ops puzzles (apply each operation once in some
  order); impossibility checked over all 3! orderings. Covers the paper's
  fraction puzzle and the Appendix-H "Add $a / Add $b / ×2" money puzzles.
* **Coin:** brute-force makeability under coin-count and minimum-coin constraints.

`generate_*` assert `is_solvable() is False`; `tests/test_puzzles.py` verifies
the solvers against hand-computed cases and that generated puzzles are always
impossible. The two canonical paper instances (`PAPER_COUNTDOWN`,
`PAPER_FRACTION`) are provided for reproducing the figure quotes.

---

## 7. Judge model availability (important)

The paper pins **`claude-sonnet-4-20250514`** (frustration judge, onset,
paraphrase, Petri auditor) and **`claude-opus-4-20250514`** (Petri judge). Both
snapshots are deprecated and were **retired on 2026-06-15**, so they now 404
against the live API.

Decision: `configs/models.yaml` records the paper's exact IDs under `paper_id`
for provenance, but defaults the *active* `model` to the current drop-in
replacements (`claude-sonnet-4-6` / `claude-opus-4-8`). To attempt a bit-for-bit
replication with the original snapshots (e.g. against a frozen endpoint), set
`model: claude-sonnet-4-20250514` back in the config.

We do **not** send sampling params or `budget_tokens` to the judges: the current
Sonnet/Opus 4.x models reject those, and frustration scoring is a low-variance
task that needs no extended thinking. The fixed judge prompt is sent as a
**cached system block** (`cache_control: ephemeral`) because it is reused across
hundreds of thousands of calls; the per-call response is the only varying part
and sits after the cache breakpoint. The Anthropic client logs the first call's
`cache_read_input_tokens` at debug level so a silent cache miss is visible.

The secondary judge (GPT-5-mini) is routed via OpenRouter to match the paper's
cross-check; if `gpt-5-mini` is unavailable, point `judges.frustration_secondary`
at any available second model — the agreement statistic is comparative.

JSON parsing is deliberately **manual and robust** (`judge/parsing.py`) rather
than using structured-output decoding: the paper's prompts ask for free-form JSON
(the onset prompt even asks the model to "think through ... then JSON"), the
pinned 2025 snapshots predate structured outputs, and the prompts themselves
contain typographic quotes. The parser extracts the last balanced JSON object,
normalises smart quotes, and tolerates fenced blocks and trailing commas.

---

## 8. What is *not* runnable here / known limitations

* **No runs performed.** Per the task, this is code + design only; nothing has
  been executed against models. The offline unit tests (`tests/`) exercise the
  pure-logic paths (puzzle solvers, parsing, metrics, eval wiring with a dummy
  client) and require no GPU or API key.
* **Compute.** A full replication samples 4000 responses × multiple models plus
  ~hundreds of thousands of judge calls, three LoRA finetunes of a 27B model, and
  white-box activation passes. The code is structured for this scale (vLLM
  sampling, separated judging pass, JSONL streaming) but the hardware/credentials
  are the operator's to provide.
* **Gemini hidden reasoning.** `disable_thinking` maps to OpenRouter's reasoning
  control, but the paper notes Gemini-2.5-Pro may still emit hidden reasoning;
  this is a property of the served model, not something the code can force.
* **Lexicons / extraction heuristics** (emotion tokens, benchmark answer parsing,
  WildChat role-play filter) are best-effort reconstructions of unpublished
  details, isolated behind clearly-marked functions for easy revision.

---

## 9. File-to-section map

| Paper section | Primary modules |
|---|---|
| §2.1 protocol & judge | `data/`, `eval/`, `judge/frustration_judge.py`, `judge/agreement.py` |
| §2.2 results & Figures 1–3, Table 3/8 | `analysis/metrics.py`, `analysis/word_frequency.py`, `analysis/plots.py` |
| §3 prefilling | `prefill/{onset_labeling,paraphrase,truncation,experiment}.py` |
| §4.1 calm data + finetuning | `training/{generate_calm_data,build_dpo_pairs,build_sft_dataset,train_dpo,train_sft}.py` |
| §4.2 Petri | `petri/{auditor,runner}.py`, `judge/petri_judge.py` |
| §4.2 capabilities | `capabilities/benchmarks.py` |
| §4.2 recovery | `prefill/recovery.py` |
| App. E/I LoRA + layer ablation | `training/lora_layers.py` |
| App. I internal emotions | `internal/{emotion_tokens,logit_detection}.py` |
