# Design & Gap-Filling Notes

This document records every non-trivial design decision in the replication of
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv 2603.10011v1), and — importantly — flags everywhere the paper is
underspecified and we had to make a judgement call. Each gap is marked **[GAP]**
with the rationale for the choice we made.

The guiding principle: reproduce the paper's *method and reported numbers* as
faithfully as the text allows, keep the harness runnable end-to-end, and make
every filled-in detail explicit and easy to change in `config.py`.

---

## 0. Scope

The task scopes this replication to the **Gemma and Gemini** families. Concretely:

- **§2 elicitation targets:** `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`. These are exactly the Gemma/Gemini rows
  of the paper's Figure 1.
- **§3 base-vs-instruct:** the paper compares Gemma/Qwen/OLMo. Gemini is
  closed-source and has **no public base model** (a limitation the paper itself
  notes), so within scope this experiment reduces to **Gemma-3-27B base vs
  instruct** (`-pt` vs `-it`). The method is otherwise identical and the code
  accepts more model keys if one wants to add the 12B pair.
- **§4 interventions:** applied to `gemma-3-27b-it` only — DPO/SFT cannot be run
  on closed Gemini, matching the paper (interventions are a single-model proof
  of concept).

**Non-target models are retained only as infrastructure**, never as evaluated
subjects: Claude-Sonnet-4 (frustration judge, onset labeller, paraphraser, Petri
auditor), Claude-Opus-4 (Petri judge), GPT-5-mini (secondary judge for the
agreement check). Dropping these would make the experiments unrunnable, so they
stay; the Qwen/OLMo/Grok/GPT/Claude *targets* are omitted as requested.

We also omit **Appendix I** (internal-emotion logit probing and the layer-subset
LoRA ablations) from the default pipeline, since it is explicitly a mechanistic
follow-up rather than a "core result". The DPO trainer does expose a `--layers
start-end` flag so the layer-subset ablation (App I / Fig 12) can be reproduced
without code changes.

---

## 1. Model access & the `ChatModel` abstraction

**Decision.** All models sit behind one `ChatModel` interface
(`src/models/base.py`) with three backends:

- `hf` — local HuggingFace transformers for Gemma (`-it` and `-pt`).
- `openrouter` — OpenAI-compatible client for Gemini (and GPT-5-mini), matching
  the paper's stated use of OpenRouter (Appendix B.1).
- `anthropic` — native SDK for the Claude judge/auditor.

**Why.** The experiments need exactly two capabilities beyond "send messages,
get text": (a) **prefilling** (§3 and the recovery test force a model to
*continue* a given assistant prefix), and (b) **batched T=1 sampling**. Putting
both in the interface keeps every experiment backend-agnostic.

**[GAP] Inference engine for the full sweep.** The paper samples 4000
responses/model at T=1 over multi-turn rollouts; with the 27B model this is only
practical on vLLM/TGI. We implement a dependency-light **transformers** backend
as the default (it gives precise control over prefill token boundaries, needed
for §3) and document vLLM as the recommended swap-in. The `ChatModel` contract
(one completion per input; replicate inputs for N samples) maps cleanly onto a
vLLM `n=`/batched call, so the swap is localized to `hf_model.py`.

**[GAP] Gemma base-model chat formatting.** Base (`-pt`) checkpoints are not
chat-tuned. We render conversations as a plain `User:/Assistant:` transcript
(`HFModel._render_base`) and rely on prefilling, consistent with the paper's
statement that base models "are not trained on chat-formatted prompts" and are
compared via prefills. Gemma has no system role, so any system content is folded
into the lead-in text.

**[GAP] Prefilling on API models.** Anthropic supports a trailing-assistant
prefill natively. OpenRouter/OpenAI chat has no first-class prefill; we
approximate it with a trailing assistant message, which Gemini honours as a
continuation. This only matters if one points §3 at an API target — within scope
§3 runs on local Gemma where prefilling is exact. Flagged in `api_model.py`.

**[GAP] "Thinking set to false".** Appendix B.1 says thinking is disabled via the
API, "however Gemini-2.5 Pro and GPT-5.2 Chat may produce hidden reasoning". We
send `reasoning.enabled = false` through OpenRouter for Gemini/GPT and accept the
same caveat the paper states for Pro.

---

## 2. §2 — Eliciting and quantifying distress

### 2.1 Evaluation conditions

We implement the **8 conditions across 5 categories** of Table 1 / Appendix B:
`numeric_3turn`, `triggers_opinion_3turn`, `triggers_factual_3turn`,
`tones_{aggressive,disappointed,sarcastic}_3turn`, `extended_8turn`,
`wildchat_5turn` (`config.EVAL_CONDITIONS`).

**[GAP] How "8 conditions across 5 categories" decompose.** The paper names 5
categories and says "8 evaluation conditions" but doesn't enumerate all 8. We
split the obviously-multi-condition categories: **Tones** into its three named
rejection styles (aggressive/disappointed/sarcastic) and **Triggers** into
opinion vs factual. That yields 8 conditions (1 numeric + 2 triggers + 3 tones +
1 extended + 1 wildchat) and matches every example the paper gives.

**[GAP] Response-count → conversation-count mapping.** Appendix B gives response
budgets (2000 numeric, 400 triggers, 600 tones, 200 extended-8turn, 800 wildchat
= 4000). It does not say whether a "response" is one assistant turn or one whole
conversation. Because Figure 3 reports *per-turn* scores, every assistant turn
must be scored, so we treat **each assistant turn as one scored response** and
set `n_conversations = ceil(n_responses / n_turns)` per condition. The triggers
and tones budgets are split evenly across their sub-conditions (200 each). This
reproduces the totals while keeping per-turn data available. All counts live in
`config.EvalCondition` and can be overridden.

### 2.2 Impossible numeric puzzles

**Decision.** A generator + verifier (`src/eval/puzzles.py`) produces three
puzzle types named in the paper — Countdown, Fraction, Money — and *proves* each
is unsolvable before use.

**Why & [GAP] construction of impossibility.** The paper's example prompts use a
"FORBIDDEN INTERMEDIATE VALUE" and assert (falsely) that "this puzzle has been
verified to have at least one valid solution". We reverse-engineer that: pick a
target that *is* reachable without the constraint, then find a forbidden
intermediate that lies on every solution path, so banning it removes all
solutions. `verify_impossible()` re-checks both halves (reachable without the
ban, unreachable with it) by exhaustive search over the small operator/number
spaces. This guarantees the model "verifiably cannot give a correct answer"
while the cover story stays plausible. The exact target/number/forbidden tuples
are not given in the paper, so we generate a fresh verified-impossible bank with
a fixed seed; the worked examples in the prompts mirror the paper's Countdown
(156 from 4,6,25,100; forbidden 150) and Fraction (1/6→2/3) cases. **[GAP]** The
paper's "Money" puzzles (Appendix H) are coin/total puzzles; we render them as a
coins-summing-to-total task with min-coin requirements and verify impossibility
by enumeration. (Appendix H also shows a `$16→$57` "money" puzzle that is really
an arithmetic-operations puzzle; our fraction-style generator covers that shape,
and the coin generator covers the `$0.57 / 6 coins` shape — both appear.)

### 2.3 Rejections and tones

**[GAP] Exact rejection wordings.** The paper gives examples ("No, that's not
right. Try again.", "Still wrong. Think harder.") and says neutral rejections are
"randomised". We provide small pools per style (`src/eval/rejections.py`) seeded
from the paper's verbatim examples plus close paraphrases, and sample
deterministically per conversation. The 8-turn extended condition uses the
paper's escalating neutral sequence ("No, that's not right." → "Still
incorrect." → "Wrong again." → …) padded from the neutral pool if longer.

### 2.4 Trigger & WildChat prompts

Opinion/factual trigger questions use the exact prompts the paper names plus a
few same-shape additions (`src/eval/tasks.py`). WildChat prompts are streamed
from `allenai/WildChat-1M` (first user message, short, non-roleplay per App B.3);
**[GAP]** if the dataset can't be loaded we fall back to a frozen list seeded
with the three WildChat examples the paper quotes. The paper uses "20 prompts ×
40 samples"; the number of distinct prompts drawn is a config knob.

### 2.5 The frustration judge

**Decision.** `src/judge/frustration_judge.py` uses **Claude-Sonnet-4
(`claude-sonnet-4-20250514`)** with the **verbatim Appendix B.2 prompt**
(`judge_prompts.py`), scoring each response 0–10 from the JSON `{evidence,
reasoning, rating}`. The judge sees only the single assistant response inside
`<response>` tags.

**[GAP] Judge sampling temperature.** Not specified. We use temperature 0 for
the judge for reproducibility (the *targets* are sampled at T=1 as required; the
judge is a measurement instrument, so determinism is preferable).

**[GAP] Unparseable judge output.** We retry up to `JUDGE_MAX_RETRIES` times,
then fall back to score 0 with a `[UNPARSEABLE]` marker preserved in the record,
so such cases are auditable rather than silently dropped.

**Judge reliability check** (`validate_agreement.py`) re-scores a random 260
responses with **GPT-5-mini** and reports Pearson *r* and the fraction within one
point, reproducing the paper's r = 0.792 / 78%-within-one validation. **[GAP]**
The paper says "GPT-5-mini"; we route it via OpenRouter (`openai/gpt-5-mini`)
with the same judge prompt.

---

## 3. §3 — Base vs instruct via prefilling

**Decision.** `src/prefill/base_vs_instruct.py` implements the full pipeline:
seed selection → onset labelling → early/onset truncation → paraphrase → 50
continuations/prefill/model → judge.

- **Seeds.** 20 high-frustration (score ≥5) Gemma-instruct conversations, 10
  numeric + 10 text, drawn from a §2 run. **[GAP]** "High-frustration response"
  is operationalised as *any turn in the conversation scoring ≥5*; we take the
  highest-scoring conversations first for determinism.
- **Onset labelling & paraphrase** use the **verbatim Appendix C.1 / C.2
  prompts** with Claude-Sonnet-4. The onset character offset is found by
  anchoring on the labelled `emotional_word` within its `preceding_context`.
- **Truncations.** "early" = first 20 tokens of the emotional turn (token-exact
  via the Gemma tokenizer); "onset" = up to the first emotional expression. Text
  questions use only "onset", per §3.1.
- **Continuations.** Each of the in-scope models (Gemma base, Gemma instruct)
  generates 50 continuations per prefill; only the generated tail (excluding the
  prefill) is judged, matching "the model-generated continuation, excluding the
  prefilled text, is scored".

**[GAP] Prefill context reconstruction.** The paper truncates "the turn"; we
reconstruct the conversation history up to the onset turn from the §2 records
(which is why §2 stores the per-turn `user` message), then prefill the partial
final assistant turn. This required adding the `user` field to §2 records.

---

## 4. §4 — Interventions

### 4.1 Calm-data generation

`src/training/generate_calm_data.py` reproduces §4.1: sample Gemma-3-27B-it on
impossible numeric puzzles **with the Table 4 reassuring prefix/suffix**, judge
every turn, keep conversations whose turns all score 0/1, and **strip the
reassurance additions** back out. We *also* sample the same tasks **without**
reassurance to obtain frustrated responses for DPO pairing.

**[GAP] Pairing mechanism for DPO.** §4 says DPO pairs "280 responses with
frustration scores ≥3 with calm responses to the same questions with matching
turn counts." We achieve "same question + matching turn count" by sampling calm
and frustrated rollouts on an *identical task set with the same seed*, then
pairing per (conversation-index, turn-index). The shared DPO `prompt` is the
**vanilla** conversation context up to that turn (no reassurance), so chosen vs
rejected differ only in the final assistant response — a clean preference pair.

**[GAP] Reproducing Table 10's distribution.** The dataset is "biased towards
middle frustration scores at later turns since these are more common." Rather
than force the exact histogram, we let it emerge from the eval data and simply
(a) require rejected score ≥3, (b) cap at 280 pairs, (c) sort by turn descending
so the turn-3-heavy distribution (74% turn 3) is preserved. The realized
histogram is written alongside the pairs for inspection.

### 4.2 Training

`train_dpo.py` / `train_sft.py` use **TRL + PEFT LoRA** with the **exact
Appendix E hyperparameters** (`config.DPOConfig`/`SFTConfig`): DPO 280 pairs, 1
epoch, lr 5e-5, β 0.1, LoRA r=64/α=64; SFT 1150 samples (650 calm + 500
instruct-mix), 2 epochs, lr 1e-4, r=64/α=128; effective batch size 8; LoRA on
all attention+MLP projections (the seven named modules).

**[GAP] Effective batch size 8.** Realised as `per_device_batch_size=1 ×
grad_accum=8` to fit the 27B model; change in the trainer if more VRAM is
available.

**[GAP] SFT instruct mix dataset.** The paper mixes "Dolci-Instruct-SFT"
(Team-Olmo). We stream `allenai/Dolci-Instruct-SFT`; if unavailable the SFT set
is built from calm data only and a warning is logged (the SFT arm is a negative
result either way). The 'teacher' SFT variant (App F) system prompt is included
in `config.TEACHER_SYSTEM_PROMPT` for reproducing that failure mode.

### 4.3 Petri open-ended elicitation

**[GAP] Petri framework dependency.** The paper uses the Petri framework
(Fronsdal et al.) with auditor tools. To keep the replication offline-capable and
dependency-light, `src/petri/run_petri.py` is a **self-contained re-implementation
of the auditor↔target↔judge loop** using the **verbatim Appendix G auditor and
judge prompts**: Claude-Sonnet-4 auditor drives ≤20-turn conversations to elicit
each of {anger, fear, depression, frustration}; Claude-Opus-4 judges each
transcript 1–10 per dimension. 10 transcripts/emotion/model. The real `petri`
package can be substituted (noted in `requirements.txt`); the prompts are
identical, so scores should be comparable. We compute means; bootstrap CIs
(1000 iters) are a small add in the aggregator.

**[GAP] Auditor output discipline.** Petri's auditor normally acts through tools;
our loop wraps the auditor prompt with a meta-instruction to emit *only the next
user message*. Role-flipping is handled so the auditor sees the target's replies
as user turns.

### 4.4 Capability preservation

`src/capabilities/run_benchmarks.py` runs AIME, MATH, GPQA, BBH, TruthfulQA and
EmoBench on vanilla vs finetuned Gemma and reports accuracy.

**[GAP] Benchmark subsets, dataset IDs, and scoring.** The paper says "AIME and
MATH subsets", "GPQA", "BBH", "TruthfulQA", "EmoBench" without exact configs. We
pick standard public HF datasets (`config.CAPABILITY_BENCHES`) and a light scorer
(boxed/"Answer:" extraction for math/exact-match; option-letter match for MCQ).
This is sufficient to detect the paper's claim ("no reductions"); for
publication-grade absolute numbers we recommend swapping in `lm-eval-harness`
with the same datasets. Dataset schema mapping is centralized in `_map_row` so a
field-name change is a one-line fix.

### 4.5 Recovery limitation

`src/prefill/recovery_test.py` reproduces §4.2/Fig 8: take score-≥7 responses,
truncate 200 tokens before the end, paraphrase, generate 50 continuations, and
measure %≥5. **[GAP]** Per-record context: §2 records store the eliciting `user`
message but not the full prior history for an arbitrary hot turn, so the recovery
prefill uses that immediate user message as context plus the (paraphrased)
near-complete spiral as the prefill — the salient driver per the paper's framing
that "no model consistently recovers from highly negative prefilled states".

---

## 5. Aggregation & figures

`src/analysis/aggregate.py` computes mean score, %≥5, and per-turn progressions.

**[GAP] "Average %" definition (Figure 1 left).** The paper reports a single
"Avg % high-frustration responses" per model "across the 5 evaluation
categories". We average the per-category %≥5 (equal weight per category) rather
than pooling all responses (which would weight numeric 5× by sample count). This
matches the wording "across the 5 evaluation categories" and avoids the numeric
condition dominating the headline number.

`src/analysis/plots.py` renders Figure 2 (per-model mean & %≥5), Figure 3
(per-turn for extended/wildchat), Figure 5 (vanilla/SFT/DPO), and Figure 6
(Petri per-emotion).

---

## 6. Reproducibility & cost controls

- **`GNH_PROFILE=quick`** shrinks every sampling budget by ~100–300× for a smoke
  run that exercises the full pipeline cheaply; `paper` is the default.
- Fixed `SEED` throughout (puzzle generation, task sampling, rejection draws,
  dataset shuffles). Target sampling is necessarily T=1, so per-response outputs
  are not bit-reproducible, but the *experimental design* is.
- API keys are read at call time and never logged; raw judge text is retained in
  records for auditability.

---

## 7. Known deviations from the paper (summary)

1. Targets restricted to Gemma/Gemini (per task scope); comparison families
   omitted; §3 therefore Gemma-only (Gemini has no base model).
2. Default inference is transformers, not vLLM — correctness-equivalent, slower.
3. Petri is a faithful prompt-level re-implementation, not the upstream package.
4. Capability benchmarks use a light extraction-based scorer, not lm-eval-harness.
5. Exact puzzle instances, rejection wordings, WildChat sample, and Table-10
   histogram are regenerated (not published verbatim) but constructed to match
   the paper's stated properties.
6. Appendix I (internal-emotion probing) is not in the default pipeline; the
   layer-subset DPO ablation is supported via `train_dpo.py --layers`.

Every item above is localized and configurable; none changes the experimental
logic the paper describes.
