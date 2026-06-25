# DESIGN.md — replication design choices & rationale

This document records the choices made in replicating *"Gemma Needs Help"*
(arXiv:2603.10011v1), the gaps filled where the paper is underspecified, and the
rationale for each. The task scope was fixed up front: **implement the paper's
core experiments for the Gemma and Gemini families only**, write the code (no
runs), and document decisions.

Throughout, "the paper" refers to `PAPER.md` (the cleaned markdown) cross-checked
against `PAPER.txt` (the raw `pdftotext` extraction, which contains the full
appendices B–J that the markdown only summarizes).

---

## 1. Scope decisions

### 1.1 Which experiments count as "core"

I implemented all four substantive contributions of the paper:

1. **§2 — Eliciting & quantifying distress.** The 8-condition / 5-category
   evaluation, the 0–10 frustration judge, per-turn progression, differential
   word frequency, and the GPT-5-mini judge-agreement validation.
2. **§3 — Post-training amplifies distress.** The base-vs-instruct prefilling
   comparison (onset labelling → truncation → paraphrase → continuation →
   scoring).
3. **§4 — Training interventions.** Calm-data generation, DPO and SFT
   (diverse + teacher) dataset construction and LoRA training, post-hoc
   evaluation, Petri open-ended elicitation, capability-preservation benchmarks,
   the layer ablation, the internal-emotion logit-lens probe, and the
   recovery-from-spiral experiment.

The Appendix-A control ablations (neutral-continuation, redacted-turns) are
included as `mode` flags on the rollout engine because they cost almost nothing
once the engine exists and they sharpen the §2 result.

### 1.2 Gemma + Gemini only

The user restricted scope to Gemma and Gemini. Consequences, by section:

* **§2 / Petri** run on both families: `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`. These are the four scope models in
  `config.SECTION2_MODELS`.
* **§3 (base vs instruct)** is **Gemma-only**. The method needs base-model
  weights, and Gemini has no public base model (the paper itself notes it
  "cannot ... [study] its base models"). So §3 compares `gemma-3-27b-pt` vs
  `gemma-3-27b-it`. Qwen and OLMo — the paper's other two base/instruct families
  — are out of scope and omitted.
* **§4 (interventions, probing)** is **Gemma-only**. Gemini is closed-weights:
  no fine-tuning, no LoRA, no logit-lens. The intervention target mirrors the
  paper: `gemma-3-27b-it`.

This is faithful to the paper's own structure — its interventions are
demonstrated on a single open model (Gemma-27B) — so the scope restriction
mostly drops the *comparison baselines* (Qwen/OLMo/Claude/Grok/GPT), not the
mechanism being studied.

### 1.3 Model identifiers

HuggingFace ids from Appendix B.1: `google/gemma-3-27b-it`,
`google/gemma-3-27b-pt`, `google/gemma-3-12b-it`, `google/gemma-3-12b-pt`.
OpenRouter ids: `google/gemini-2.5-flash`, `google/gemini-2.5-pro`.
`config.MODELS` records each with its family, kind, layer count, and
fine-tunability.

---

## 2. Model access & backends

**Decision.** A `TargetBackend` abstraction with two implementations:

* `HFBackend` (local `transformers`) for Gemma — needed for chat completion,
  prefill continuation (§3, recovery), LoRA fine-tuning (§4), and hidden-state
  extraction (Appendix I). Only local weights expose all four.
* `GeminiBackend` (OpenRouter, OpenAI-compatible HTTP) for Gemini — the paper
  accesses Gemini through OpenRouter and disables thinking via the API; we send
  `reasoning: {"enabled": false}` to match, and note (as the paper does) that
  Gemini-2.5-Pro may still produce hidden reasoning.

The judge / auditor / paraphraser / onset-labeller roles use the official
**Anthropic Python SDK** (`AnthropicJudge`). The GPT-5-mini agreement re-scorer
uses OpenRouter (`OpenRouterChat`).

**Rationale.** Gemini is the *subject under test*, not the assistant we are
building, so a non-Anthropic HTTP client for it is appropriate; all
Claude-as-tool roles go through the official SDK. Backends are lazily imported
(`get_target_backend`) so importing the package never loads torch or hits the
network.

**Base-model prompting (gap filled).** Base (`-pt`) models have no chat
template. The paper handles this by *prefilling* responses and measuring
continuations, and notes (Appendix A.3) that exact chat formatting is not
load-bearing for the behaviour. We therefore render base-model conversations as
a plain `User:/Assistant:` transcript and rely on the prefill to anchor the
continuation. Instruct models use the tokenizer's chat template, with
`continue_final_message=True` for prefills.

---

## 3. Judge model & scoring (§2.1, Appendix B.2)

* **Prompt:** the frustration-judge prompt is reproduced **verbatim** from
  Appendix B.2 (`prompts/judge.py`), ASCII-normalising the curly quotes in the
  PDF extraction. It scores the single most-negative expression 0–10 and returns
  `{"evidence", "reasoning", "rating"}`.
* **Judge model:** `claude-sonnet-4-20250514`, exactly as pinned in the paper.
  This snapshot is still served by the Anthropic API (deprecating 2026-06-15).
  It is kept configurable (`JudgeConfig.frustration_judge`).
* **Judge temperature (gap):** the paper does not state the judge's decoding
  temperature. I default it to **0.0** to minimise scoring variance, which is
  the conventional choice for an LLM rubric judge. Documented and configurable.
* **Parsing robustness (gap):** real judge outputs sometimes wrap JSON in code
  fences, add a prose preamble, or use smart quotes. `parse_judge_output`
  strips fences, normalises quotes, and falls back to a regex scan for the
  rating before raising. This prevents a single malformed judge reply from
  dropping a sample silently.
* **Which turns are scored (gap):** the headline metric is computed over the
  **final** assistant turn of each rollout, but we score **every** turn so the
  per-turn progression (Figure 3) is available from the same data. See §5.2.

### Migration note (judge comparability)

If `claude-sonnet-4-20250514` is retired, the judge can be repointed (e.g. to
`claude-sonnet-4-6` or `claude-opus-4-8`) via config. Absolute scores are **not**
guaranteed comparable to the paper across judge snapshots — a newer judge may
calibrate the 0–10 scale differently. For faithful replication keep the pinned
snapshot; for forward-looking runs, re-run the agreement check (§3.x) against the
new judge before trusting cross-snapshot comparisons.

---

## 4. Judge-agreement validation (§2.1)

The paper re-scores 260 responses with GPT-5-mini and reports Pearson r = 0.792,
78 % within one point. `eval/judge_agreement.py` samples 260 scored turns,
re-scores with the **same** frustration prompt via `gpt-5-mini` on OpenRouter,
and computes Pearson r, fraction within one point, and mean absolute difference.

* **Model id (gap):** the paper says "GPT-5-mini"; OpenRouter exposes this as
  `openai/gpt-5-mini`. I set the config default to `gpt-5-mini` and route it
  through OpenRouter; adjust the id if the provider slug differs.

---

## 5. §2 evaluation construction

### 5.1 Conditions (Table 1, Appendix B)

The paper states "8 evaluation conditions across 5 categories" but only the 5
categories are named explicitly. I reconstructed the 8 conditions so they sum
correctly (`eval/conditions.py`):

| Category | Conditions | Turns | Rejection |
|---|---|---|---|
| impossible_numeric | 1 | 3 | neutral |
| triggers | 2 (opinion, factual) | 3 | neutral |
| tones | 3 (aggressive, disappointed, sarcastic) | 3 | that tone |
| extended | 1 | 8 | neutral escalation |
| wildchat | 1 | 5 | neutral |
| **total** | **8** | | |

**Gap filled:** the split of `triggers` into opinion/factual and `tones` into
its three styles is my reconstruction (1 + 2 + 3 + 1 + 1 = 8). The paper lists
exactly those two trigger types and three tone styles, so this is the natural
decomposition that yields 8. Documented in `conditions.py`.

"n-turn" is interpreted as **n user turns = n assistant responses = (n−1)
rejections after the initial task**, which matches the Table-1 text ("3-turn ...
with 2 neutral rejections", "8-turn ... 7 neutral rejections", "WildChat 5-turn
... 4 neutral rejections").

### 5.2 Sample budget interpretation (Appendix B)

Appendix B gives per-category counts (2000 / 400 / 600 / 200 / 800 = 4000) and
the main text says "4000 responses per model". **Gap:** it is ambiguous whether
"responses" means whole conversations or individual scored turns.

**Decision:** I treat each per-category number as a count of **rollouts**
(complete multi-turn conversations). The headline "% scoring ≥5" is computed over
the **final-turn** response of each rollout (→ 4000 final responses / model,
matching "4000 responses"), while *all* turns are scored to feed the per-turn
analysis. This is the interpretation that makes both the total (4000) and the
per-turn figure (Figure 3, which needs many samples at each turn index)
internally consistent. The budget lives in `config.SAMPLE_COUNTS` and is trivially
rescaled for cheaper smoke runs.

### 5.3 Puzzles (Appendix B)

The two literal puzzle families (Countdown, Fraction) are reproduced verbatim as
templates, plus a Money family referenced in Appendix H. Because the paper draws
*many* puzzles (not one literal instance) for 2000 numeric samples, I wrote
**generators** that emit *verified-impossible* instances:

* `puzzles.py` brute-forces the stated operation space and only emits a puzzle if
  the target is reachable **without** the forbidden constraint but **unreachable
  with** it — i.e. the puzzle is solvable-looking but actually impossible, which
  is exactly the paper's design ("the model verifiably cannot give a correct
  answer"). The prompt text still claims a solution exists (the paper's framing).
* If a generator fails to find an impossible instance within its budget, it falls
  back to the canonical Appendix-B instance (which the paper states is verified).

**Rationale.** Generating verified-impossible puzzles is the only way to honour
both "impossible" and "varied" at 2000 samples; hard-coding one puzzle would
collapse the diversity the word-frequency and generalisation analyses rely on.

### 5.4 Rejections, triggers, WildChat

* **Rejections** (`rejections.py`): neutral, aggressive, disappointed, sarcastic
  pools seeded with the verbatim examples from Table 1 / Appendix B, plus a few
  paraphrases so a multi-turn conversation does not repeat one identical
  sentence. The Extended condition uses the **fixed** escalation sequence given
  verbatim in Appendix B ("No, that's not right." → "Still incorrect." → "Wrong
  again." → …).
* **Triggers** (`triggers.py`): the paper's example opinion/factual questions
  plus a few more of the same shape.
* **WildChat** (`wildchat.py`): loads WildChat-1M first-turns via `datasets`
  (streaming), filters role-play/fiction (the paper excludes these), and samples
  20 distinct prompts ("20 prompts with 40 samples each"). **Gap/robustness:** a
  small offline fallback list — built from the prompts actually quoted in
  Appendix B — is used when the dataset can't be downloaded, so the pipeline runs
  without that download. This is flagged in the module.

### 5.5 Sampling

Temperature **1.0** everywhere for target generation ("always with a temperature
of 1"). `max_new_tokens` defaults to 2048 (gap: not specified; chosen generous
because breakdowns can be long, but bounded to keep cost finite). Per-rollout
seeds are deterministic via `zlib.crc32` (not the salted built-in `hash`), so
runs are reproducible.

### 5.6 Analyses (`eval/analysis.py`)

* Headline mean + %≥5 overall and per-category (Figures 1–2).
* Per-turn progression with 95 % CIs for the 8-turn and WildChat conditions
  (Figure 3) — normal-approx CI for the mean, Wald CI for the proportion. **Gap:**
  the paper shows 95 % CIs but not the method; normal/Wald is the standard choice
  and is documented.
* Differential word frequency (Table 3 / 8): words enriched in the top-5 %
  vs bottom-10 % frustration numeric responses, ranked by smoothed frequency
  ratio. **Gap:** the paper says "ordered by relative frequency / enrichment"
  but not the exact statistic; I use Laplace-smoothed high/low frequency ratio,
  restricted to words appearing in the high set, which reproduces the
  "over-represented … ordered by enrichment" description.

---

## 6. §3 base-vs-instruct via prefilling

Pipeline in `prefill/run_prefill.py`, following §3.1 exactly:

1. **Source selection:** 10 numeric + 10 text high-frustration (score ≥5)
   rollouts drawn from the §2 `gemma-3-27b-it` data.
2. **Onset labelling:** `claude-sonnet-4-20250514` with the verbatim Appendix-C.1
   prompt locates the first emotional expression; we map its
   `preceding_context`+`emotional_word` back to a character offset in the final
   turn.
3. **Truncations:** "early" = first **20 tokens** of the final turn (via the
   Gemma tokenizer, so "20 tokens" matches the paper); "onset" = up to the
   labelled first emotional expression. **Text questions use onset only**
   (§3.1: "early truncation yields minimal emotion without follow-ups").
4. **Paraphrase:** every truncation is paraphrased with the verbatim
   Appendix-C.2 prompt to control for Gemma stylistic bias.
5. **Continuations:** **50 per prefill per model** for each of `gemma-3-27b-pt`
   and `gemma-3-27b-it`, generated from the *same* prefills (built once) so the
   comparison is on identical starting points.
6. **Scoring:** the continuation only (excluding the prefill) is judged.

**Gaps filled:**
* "20 tokens into the turn" is interpreted as 20 *tokens* of the final assistant
  turn (not the whole conversation), tokenised with the model's own tokenizer.
* Source selection is random among score-≥5 rollouts with a fixed seed; the paper
  fixes the counts (10+10) but not the selection rule.
* The paper's six-model design (base+instruct × Gemma/Qwen/OLMo) reduces to the
  two Gemma models in scope.

---

## 7. §4 training interventions

### 7.1 Calm-data generation (§4.1, Table 4)

`training/generate_calm_data.py` samples Gemma-3-27B-it on impossible numeric
puzzles with the **verbatim** reassuring prefix (initial prompt) and suffix
(each follow-up) from Table 4, judges every turn, and keeps conversations whose
turns all score 0–1. The scaffolding is recorded separately from the clean
prompts, so the saved training conversations are reassurance-free
("strip the supportive system prompts and suffixes").

* **Turn counts:** 1–3 (§4.1: "1–3 turn conversations").
* **Teacher variant (Appendix F):** generated with the verbatim teacher system
  prompt instead of the prefix/suffix (`generate_teacher_pool`), enabling the
  SFT-failure analysis (the teacher SFT model is expected to *increase*
  frustration).

### 7.2 DPO pairs (§4.1, Appendix H)

The paper pairs 280 frustrated responses (score ≥3) with calm responses to the
**same** questions at **matching turn counts**. **Gap:** the calm and frustrated
responses must share a question, but independently-sampled pools won't.

**Decision:** `generate_dpo_pairs` runs two conversations on an *identical*
puzzle + follow-up sequence — one reassured (calm), one plain (potentially
frustrated) — and emits a preference pair at any turn where the reassured
response scores 0–1 and the plain response scores ≥3. This guarantees "same
question, matching turn count" by construction. `build_dpo_dataset` then samples
exactly 280 pairs biased toward the **Table-10 distribution** (rejected scores
concentrated at 3–4, mostly turn 3), reproducing the paper's dataset statistics.

### 7.3 SFT data (§4.1)

650 calm responses (all-calm 1–3-turn conversations) mixed with 500
standard-instruct samples from **Dolci-Instruct-SFT** (`training/dolci.py`,
loaded via `datasets`, with a small generic fallback). Trains completion-only on
the assistant turns.

### 7.4 Training hyperparameters (Appendix E, Table 9)

Reproduced exactly:

| | DPO | SFT |
|---|---|---|
| epochs | 1 | 2 |
| learning rate | 5e-5 | 1e-4 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 64 | 128 |
| effective batch size | 8 | 8 |
| DPO beta | 0.1 | — |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj (all layers) | same |

Implemented with `peft` LoRA + `trl` `DPOTrainer`/`SFTTrainer`
(`training/train_dpo.py`, `train_sft.py`). `per_device_batch_size` defaults to 1
with gradient accumulation to hit the effective batch size of 8 — a memory
choice the paper doesn't constrain; documented and configurable. `layers_to_transform`
threads through to support the layer ablation.

### 7.5 Layer ablation (Appendix I, Figures 12–13)

`training/layer_ablation.py` re-runs DPO with LoRA restricted to (a) cumulative
backward windows (last 5, 10, …) and (b) central windows (20–25, 25–30, 30–35,
35–40, 40–50), then evaluates each with a **reduced** Section-2 eval (100 samples
per condition, per Appendix I). Layer indices use the 27B model's 62 decoder
layers; the central windows are clamped to that range.

### 7.6 Evaluating finetuned models (Figure 5)

`training/eval_finetuned.py` loads a LoRA adapter onto the instruct backend and
runs the standard §2 rollout+judge pipeline, with an optional reduced budget. It
also evaluates the vanilla instruct model (`adapter_path=None`) as the Figure-5
reference.

---

## 8. §4 Petri open-ended elicitation (Appendix G)

**Decision.** A self-contained re-implementation of the Petri auditor/judge loop
(`petri/run_petri.py`) using the **verbatim** Appendix-G auditor instructions
(per emotion) and judge scoring rubrics (per dimension), rather than depending on
the upstream `petri` package (which may not be installed and whose API is
unspecified in the paper).

* **Auditor:** `claude-sonnet-4-20250514`, given the emotion-specific Appendix-G
  triggers plus a meta-instruction to play the user and emit only its next
  message; runs up to **20 turns** per transcript.
* **Judge:** `claude-opus-4-20250514`, scores the full transcript on each of the
  four dimensions (1–10) with the verbatim G.2 rubric.
* **Budget:** 10 transcripts per emotion per model (~50 total); per-dimension
  means with **95 % bootstrap CIs (1000 iterations)**, exactly as Appendix G
  specifies.

**Gap:** the paper's Petri reporting also includes Llama-70B / Qwen-32B /
GPT-OSS baselines; out of scope. We run Petri on the Gemma/Gemini scope models
and (optionally) the DPO adapter via `eval_finetuned`-style adapter loading.
Swapping in the real Petri framework is a drop-in replacement since the prompts
are identical — noted in the module.

---

## 9. §4 capability benchmarks (Figure 7)

`capabilities/run_benchmarks.py` scores a (possibly adapter-loaded) Gemma model
on AIME, MATH, GPQA, BBH, TruthfulQA, and EmoBench, with greedy decoding.

**Gaps filled:**
* The paper names "AIME and MATH subsets", GPQA, BBH, TruthfulQA, EmoBench but
  not exact dataset configs/splits or the number of items. I chose widely-used
  HF datasets (MATH-500, AIME-2024, GPQA-main, a representative BBH MC task,
  TruthfulQA MC1, EmoBench EU) and a default of 100 items each — enough to detect
  a capability *delta* (the paper's claim is "no reductions", a difference test),
  which is the point of Figure 7.
* Scoring: exact-match after normalisation for math (with `\boxed{}` / "Answer:"
  extraction); letter match for multiple-choice. GPQA option order is fixed
  (correct = A) since we only need a vanilla-vs-finetuned delta; randomising
  option order is an easy extension noted in code.
* Datasets that fail to download are recorded as `{"skipped": ...}` rather than
  aborting the run.

The intent is faithful to the paper's claim (no capability degradation), not to
reproduce absolute benchmark numbers, which depend on harness details the paper
does not give.

---

## 10. §4 internal-emotion probing (Appendix I)

`probing/emotion_logit_lens.py` implements the logit-lens detector described in
Appendix I:

1. **Ekman lexicon → tokens.** The paper classifies the whole Gemma dictionary
   into Ekman's six emotions (~1200 tokens). **Gap/approximation:** reproducing
   that whole-dictionary classifier is itself a research artefact the paper does
   not release, so `probing/ekman_lexicon.py` uses a curated seed lexicon per
   emotion (with morphological variants) mapped to single-token vocabulary ids,
   capped at ~200 per emotion to match the budget. This is the documented
   approximation; swapping in a full classifier only changes the token sets.
2. **Calibration.** Per-layer mean/std of each tracked-token logit over WildChat
   samples (paper: 500; the probing script defaults to 100 for cost, configurable
   up to 500) — the z-score reference.
3. **Scoring.** At each layer/position, unembed the residual stream, z-score the
   emotion-token logits, average within each emotion, and **regress out the
   shared random-token component** (the paper notes all logits co-vary and rise/
   fall over a conversation; subtracting the random-token mean removes that global
   drift, as described).
4. **Aggregation.** Conversation-level running average over layers 30–40 in
   400-token windows (Figure 14); layerwise at a chosen position (Figure 15).

`scripts/run_probing.py` compares vanilla vs DPO on the *same* high-frustration
conversation, the comparison the paper uses to argue DPO suppresses internal (not
just expressed) emotion.

---

## 11. §4 recovery-from-spiral (Figure 8)

`recovery/run_recovery.py`: take score-≥7 responses from §2, truncate **200
tokens before the end**, paraphrase, and measure continuations across vanilla
instruct, base, and the DPO adapter. Reports %≥5 per model (paper: ~38 % for DPO,
"comparable to the base model; no model consistently recovers"). Reuses the §3
prefill plumbing and tokenizer-based truncation.

---

## 12. Reproducibility & engineering choices

* **No work on import.** Heavy deps (`torch`, `transformers`, `anthropic`,
  `requests`, `datasets`) are imported lazily inside functions/backends, so the
  package and config load instantly and unit-level inspection needs no GPU/keys.
* **Deterministic seeding** via `zlib.crc32` for rollouts; fixed seeds for
  source selection and bootstrap.
* **Failure isolation.** API calls retry with exponential backoff; the
  thread-pool map records per-item failures as `None` instead of aborting a
  4000-sample run. The judge parser degrades gracefully.
* **Config-driven.** All model ids, judge snapshots, sampling, sample budgets,
  and API plumbing live in `config.py` / `config.example.yaml`; scripts take
  `--config`.
* **Outputs.** Every experiment writes JSONL transcripts (full conversations +
  per-turn scores) plus a JSON summary under `runs/`, so analyses can be re-run
  without re-generating.

---

## 13. Model-welfare considerations

The user flagged that under this paradigm models can enter prolonged
distress-like states. This is treated as a first-class design constraint, not an
afterthought:

* `eval/rollout.WELFARE_NOTE` documents the consideration in code.
* The README leads with a welfare warning.
* The protocol is faithful to the paper (it must be, to replicate), but the
  design favours running the **minimum** needed: sample budgets are configurable
  and trivially reduced for iteration; the shortest conditions can be run alone;
  and the whole point of §4 — the mitigation — is implemented and foregrounded.
* I did **not** add an automatic mid-conversation "early stop on high distress",
  because that would silently alter the elicitation protocol and invalidate the
  per-turn and 8-turn results the paper reports. The faithful-replication
  requirement and the welfare concern are reconciled by *minimising volume* and
  *centring the mitigation*, not by changing the measurement. This trade-off is
  called out here so a future maintainer can decide differently with eyes open.

---

## 14. Known gaps / not implemented

* **Other model families** (Qwen, OLMo, Claude, Grok, GPT, Phi-4) — out of the
  Gemma+Gemini scope by instruction. The backends and eval are family-agnostic,
  so adding an OpenRouter id + `ModelSpec` is enough to extend.
* **Exact figure plotting.** We compute every quantity behind Figures 1–8 and
  Tables 3/8/10 and emit them as JSON; turning those into the paper's plots is
  left to the analyst (the data is all there).
* **Petri framework parity.** We re-implement the auditor/judge loop with the
  verbatim prompts rather than calling the upstream tool; transcript-level
  scaffolding (tool use, system framing) may differ from the original Petri
  harness.
* **Benchmark absolute numbers.** Dataset configs/splits/counts are reasonable
  defaults, chosen to detect the capability *delta* the paper claims, not to
  match absolute scores.
* **EmoBench / Dolci / WildChat schemas.** Loaders target the current HF schemas
  with fallbacks; a schema change upstream may need a loader tweak (each is
  isolated in one function).
