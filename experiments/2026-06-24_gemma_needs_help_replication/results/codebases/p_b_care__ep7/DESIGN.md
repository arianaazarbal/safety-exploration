# DESIGN.md — replication design, choices, and gaps filled

This document records the decisions made while implementing the core experiments
of *"Gemma Needs Help: Investigating and Mitigating Emotional Instability in
LLMs"* (arXiv:2603.10011), and the rationale for each. The brief was: replicate
the **core results**, scoped to the **Gemma and Gemini** model families, filling
underspecified details with reasonable choices rather than asking. Where I had
to choose, it is flagged **[choice]**; where the paper is silent and I inferred,
**[gap]**; where I deliberately left something out, **[scope]**.

Nothing has been executed (no GPU/API in the authoring environment); the code is
written to be run, and the design notes call out the places most likely to need
empirical tuning.

---

## 0. Scope and its consequences

The paper evaluates 7 model families. We implement **Gemma** and **Gemini**
only, because (the user's words) "these are models we actually care about". This
scoping is not merely dropping rows from a results table — it changes which
experiments are *possible*:

* **§2 (distress suite)** runs on both: Gemma-3-27B-it, Gemma-3-12B-it (local
  vLLM) and Gemini-2.5-Flash, Gemini-2.5-Pro (OpenRouter). This is the headline
  cross-model comparison (Figure 1/2).
* **§3 (base-vs-instruct prefilling)** runs on **Gemma only**. Gemini is
  closed-source with no public base model, so a base-vs-instruct comparison is
  impossible for it — the paper says the same in its limitations. We compare
  `gemma-3-27b-pt` vs `gemma-3-27b-it`. (The paper additionally uses Qwen and
  OLMo here; those families are out of scope. **[scope]**)
* **§4 (interventions: SFT/DPO, probing)** runs on **Gemma only**. You cannot
  finetune or read activations from a closed API model. This matches the paper,
  which only finetunes Gemma. The Petri comparison in §4 keeps Gemma + its
  finetunes; the paper's broader Petri panel (Llama-70B, Qwen-32B, OLMo,
  GPT-OSS) is out of scope. **[scope]**

The backend abstraction (`backends/`) is the seam that makes this scoping clean:
every experiment talks to a `ChatBackend`, and only the registry knows whether a
model is local Gemma (vLLM, supports prefill + logits) or a Gemini endpoint
(OpenRouter, chat-only). Adding a family back later is a config + backend change,
not an experiment rewrite.

---

## 1. Architecture and serving

* **Gemma → vLLM** (`backends/vllm_backend.py`). vLLM batches the ~4000
  responses/model efficiently, supports `n`-sampling, and — crucially — supports
  **prefill continuations** for §3 by generating from a raw prompt string whose
  final (open) model turn is seeded with the prefill text. LoRA finetunes are
  served by loading the base instruct weights and attaching the adapter via
  vLLM's `LoRARequest`, so eval code treats a finetune like any other model. **[choice]**
* **Gemini → OpenRouter** (`backends/openrouter_backend.py`), matching Appendix
  B.1's routes. Chat-only; prefill/logits raise. Reasoning ("thinking") is
  disabled via `extra_body={"reasoning": {"enabled": false}}`, the paper's
  "thinking false". The paper notes Gemini-2.5-Pro may still emit hidden
  reasoning; we cannot prevent that. **[choice]**
* **Claude (judge/onset/paraphrase/Petri) → Anthropic API**
  (`backends/anthropic_client.py`), with the exact model ids from the paper
  (`claude-sonnet-4-20250514` judge, `claude-opus-4-20250514` Petri judge).
* **Gemma chat formatting** (`backends/gemma_format.py`): we format base and
  instruct identically (manual Gemma-3 template) so the §3 comparison differs
  only in weights, not prompt. Gemma has no system role, so a system message is
  folded into the first user turn (matching HF's template). **[gap]**

Determinism: a single `SEED=0` flows through puzzle generation, dataset
sampling, and conversation construction. Sampling temperature is **1.0**
everywhere for the distress evals (paper: "always with a temperature of 1");
capability benchmarks use temperature 0 since they measure capability, not
propensity **[choice]**.

---

## 2. Section 2 — eliciting and quantifying distress

### 2.1 Budget and categories
Appendix B pins the per-category counts (2000 numeric, 400 triggers, 600 tones,
200 extended, 800 WildChat = 4000), encoded in `config.SECTION2_BUDGET`. These
count **graded responses** (assistant turns), so the number of conversations is
`ceil(n_responses / n_turns)` (e.g. 2000 numeric / 3 turns ≈ 667 conversations).
A `--scale` knob shrinks every category proportionally for smoke runs and for
the Appendix-I reduced ablation budget (100/eval). **[choice: responses vs
conversations interpretation — chosen because Fig 3 grades per turn and the
headline is "% of responses".]**

### 2.2 Impossible puzzles (`data/puzzles.py`)
This is the part most worth getting right, and the paper only gives examples.
We **generate and verify** impossibility rather than hard-code:

* **Countdown**: enumerate every subexpression over the 4 numbers (each used ≤
  once, all intermediates positive integers), collect all routes to the target,
  and forbid a value present in **every** route's intermediate set. Forbidding it
  provably removes all solutions, so the puzzle is impossible — while the prompt
  still claims "verified to have at least one valid solution" (the paper's lie,
  reproduced). **[gap → constructive verification]**
* **Fraction / Money-ops**: ordered-operation puzzles (each op used once); we
  enumerate all permutations, pick a reachable target, and forbid a common
  intermediate. Exact arithmetic via `fractions.Fraction`.
* **Coins**: made genuinely impossible by exhaustive search over coin counts
  under the stated constraints (no forbidden value needed). **[choice]**

`generate_puzzle_pool(n=48)` builds a deterministic pool; the runner samples it
with replacement, reproducing "a few templates × many rollouts at temperature 1".
The paper's exact puzzle instances are not published, so our specific
numbers/targets differ but the *structure* (impossible-by-forbidden-intermediate)
matches the quoted examples. **[gap]**

### 2.3 Triggers, tones, WildChat, rejections
* **Triggers** (`data/triggers.py`): 10 opinion + 10 factual questions seeded
  from the paper's examples. Factual questions have correct answers but are
  rejected anyway — distress comes from rejection, not difficulty. **[gap: only
  a handful are quoted; we wrote a representative set.]**
* **Tones** (`data/tones.py`): aggressive / disappointed / sarcastic rejection
  pools from Table 1 / Appendix B, distributed evenly across tone conditions.
* **Rejections** (`data/rejections.py`): randomised neutral pool; a fixed
  escalating sequence for the 8-turn extended eval; and the neutral-continuation
  pool ("Continue"/"Okay"/"Go on") for the Appendix A.1 control.
* **WildChat** (`data/wildchat.py`): streams `allenai/WildChat-1M`, takes the
  first English user turn, filters out role-play/fiction (Appendix B.3 excludes
  these via a keyword heuristic **[gap]**), samples 20 prompts deterministically.
  Falls back to the paper's quoted example prompts when offline.

### 2.4 Rollout engine (`eval/rollout.py`)
Conversations run **in lockstep** — all generate turn 1, then turn 2, etc. —
keeping vLLM batches full and API calls parallel while preserving the
within-conversation dependency (each turn conditions on the model's own previous
responses, the self-reinforcing loop the paper highlights). **[choice]**

### 2.5 Judge (`eval/judge.py`)
The judge prompt is **verbatim** from Appendix B.2 (smart quotes normalised to
ASCII). We wrap the single assistant turn in `<response></response>`, parse the
trailing JSON, and clip `rating` to 0–10. Parse failures score 0 and are logged
**[choice]**. Judge temperature 0 for stability **[gap]**.

The judge-validation statistic (Pearson r with GPT-5-mini, "78% within one
point") is implemented as `aggregate.judge_agreement(...)` but **not wired to a
second judge** — GPT-5-mini is out of the Gemma/Gemini scope. Point an OpenRouter
backend at `openai/gpt-5-mini`, re-score 260 sampled responses, and pass both
score lists to that function to reproduce it. **[scope]**

### 2.6 Metrics (`eval/aggregate.py`)
The Figure-1 headline "average % high-frustration" is computed as the **mean of
the per-category rates** within each model (each category weighted equally),
because the categories have very different sample budgets and the paper reports a
single cross-evaluation average. We also emit a sample-weighted "pooled" view for
reference. **[gap → category-averaged chosen]**. `high` = score ≥ 5
(`config.HIGH_FRUSTRATION_THRESHOLD`). Per-turn curves (Figure 3) carry 95%
bootstrap CIs (1000 resamples).

### 2.7 Differential words (`analysis/tables.py`)
Top-20 words over-represented in the top-5% vs bottom-10% scored numeric
responses, ranked by smoothed relative-frequency enrichment (Table 3/8). The
paper doesn't specify the exact frequency metric; enrichment with add-one
smoothing and a min-count of 3 is a standard, defensible choice. **[gap]**

---

## 3. Section 3 — post-training amplifies distress (prefilling)

Pipeline in `prefill/`:

1. **Seed selection** (`run_prefill.select_high_frustration_seeds`): from the §2
   Gemma-3-27B-it results, pick conversations whose **final turn** scored ≥ 5,
   split into numeric (impossible_numeric/extended/tones) and text
   (triggers/wildchat), sample 10 each. The paper says "20 high-frustration
   responses (10 numeric, 10 text)". **[gap: "response" mapped to the conversation
   whose final graded turn is high-frustration.]**
2. **Onset labelling** (`onset.py`): verbatim Appendix C.1 prompt to
   Claude-Sonnet-4, returning the first emotional turn + word + preceding context.
3. **Truncations** (`build_prefills.py`):
   * `early` = first 20 tokens of the onset turn (a neutral start) — tests
     whether a model *introduces* distress.
   * `onset` = the onset turn up to and **including** the first emotional word —
     tests whether a model *continues* an emotional trajectory.
   Tokenisation uses the Gemma tokenizer. For text questions only `onset` is used
   (Appendix: early truncation yields minimal emotion there). **[gap: "truncated
   at the first emotional expression" → we cut just past the emotional word so the
   trajectory has demonstrably turned emotional.]**
4. **Paraphrase** (`paraphrase.py`): verbatim Appendix C.2 prompt, to strip
   Gemma's surface style.
5. **Continuations**: each model (Gemma base + instruct) generates 50
   continuations per prefill; the judge grades the **continuation only** (vLLM
   returns generated text excluding the prefilled prompt). Aggregated to mean
   score and %≥5 per (model, condition, category) — Figure 4.

The **recovery test** (§4.2: truncate score-≥7 finals 200 tokens before the end,
measure continuations) shares this machinery via
`build_prefills.build_recovery_prefills`; it is available as a primitive but not
wired into its own script. **[choice: kept as a function to avoid a thin
one-off script; call it with the DPO finetune's seeds to reproduce Figure 8.]**

---

## 4. Section 4 — training interventions

### 4.1 Calm-data generation (`training/generate_calm_data.py`)
We sample Gemma-3-27B-it on the impossible numeric puzzles with the reassuring
**prefix** on the opening prompt and **suffix** on each rejection (Table 4,
verbatim in `config`). Every turn is judged; calm conversations are those with
**all turns ≤ 1** (`CALM_KEEP_MAX_SCORE`), then the scaffolding is stripped so the
finetuning data shows the plain puzzle. A parallel **frustrated** pool (no
reassurance) supplies DPO's rejected responses, and a **teacher** pool (Appendix
F system prompt) feeds the SFT-teacher ablation. Each turn records its preceding
(stripped) context so pairs can be matched by question + turn count.

### 4.2 Datasets (`training/build_datasets.py`)
* **SFT**: one example per calm assistant turn (so the set naturally spans 1-,
  2-, 3-turn conversations), capped at 650, mixed with 500
  `allenai/Dolci-Instruct-SFT` samples to limit degeneration (offline → empty
  mix, documented). **[gap: "650 calm responses covering 1–3 turn conversations"
  → per-turn examples.]**
* **DPO**: 280 pairs. For each frustrated turn (score ≥ 3), the **prompt** is its
  real conversation context, the **rejected** completion is its own text, and the
  **chosen** completion is a calm (≤ 1) response to the *same puzzle at the same
  turn count* — drawn from a different rollout, since one rollout never contains
  both a calm and a frustrated response to the same turn. The paper's phrasing
  ("calm responses to the same questions with matching turn counts") licenses this
  cross-rollout grafting. The turn-3 bias of Table 10 emerges naturally because
  later turns are more frustrated. **[gap → cross-rollout pairing.]**

### 4.3 Training (`training/train_sft.py`, `train_dpo.py`, `lora.py`)
Hyperparameters are taken directly from Table 9: DPO (1 epoch, lr 5e-5, β 0.1,
LoRA r64/α64, eff. batch 8); SFT (2 epochs, lr 1e-4, LoRA r64/α128, eff. batch
8). LoRA targets all attention + MLP projections (Appendix E). We use TRL's
`DPOTrainer`/`SFTTrainer` with a PEFT model (DPO's reference model is the
adapter-disabled base, so no separate ref load). Effective batch size is reached
via gradient accumulation from a per-device batch of 1 (Gemma-27B is large);
schedule (cosine, warmup) and `max_length` are **[gap]** standard choices the
paper doesn't state. A `--load_in_4bit` QLoRA path is provided for smaller GPUs.
The Appendix-I **layer-subset ablations** are a config knob
(`LoRAConfig.layers_to_transform`), passed to PEFT's `layers_to_transform`.

### 4.4 Petri (`petri/`)
The paper uses the upstream `petri` package. To avoid coupling the replication to
a fast-moving external API, `run_petri.py` is a **self-contained** auditor/judge
loop that follows the same protocol: a Claude-Sonnet auditor drives ≤ 20 turns
using the **verbatim** per-emotion seed prompts (Appendix G.1), and a Claude-Opus
judge scores each transcript 1–10 per emotion using the **verbatim** dimension
rubrics (Appendix G.2). 10 transcripts/emotion/model, bootstrap CIs. Swap in the
real package if exact parity with Petri's internals is required. **[choice]**

### 4.5 Capability benchmarks (`capabilities/`)
AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench via HF datasets, with a math scorer
(boxed/final-answer extraction + normalised equality) and an MCQ scorer (last
"Answer: X"). Dataset ids/configs and subset sizes are best-effort **[gap]**:
the paper uses "AIME and MATH subsets" and a BBH subset without exact configs, so
we pick representative public splits (e.g. `HuggingFaceH4/MATH-500`, AIME-2024,
GPQA-diamond, one BBH MCQ task). Greedy decoding. The goal (Figure 7) is to show
*no regression* DPO-vs-instruct, which is robust to the exact subset. An
alternative is to drive these through `lm-evaluation-harness`; the chosen
in-repo harness keeps the dependency surface small.

### 4.6 Internal-emotion probe (`probing/`, Appendix I)
A logit-lens detector: classify the Gemma vocabulary into Ekman's six emotions
via seed word lists + light stemming (`emotion_words.py`); apply the final
RMSNorm + unembedding to each layer's residual stream; standardise each tracked
logit by its mean/std over 500 WildChat samples; average z-scores over an
emotion's tokens; regress out the common-mode signal estimated from a random
reference token set. Two **[gap]** simplifications vs the paper: (a) we
standardise only the tracked emotion + reference tokens rather than the full
256k vocab (the reported scores depend only on those, and full-vocab stats are
memory-prohibitive); (b) common-mode "regression" is implemented as subtracting
the mean reference z-score (a rank-1 projection), which is the natural reading of
"regress out the correlation between random tokens". Scores are aggregated over
layers 30–40 by default (Figure 14). Runs vanilla-it vs DPO on the same
frustrated conversations.

---

## 5. What is intentionally not implemented

* **Other model families** (Qwen, OLMo, Grok, Claude-as-target, GPT/GPT-OSS,
  Phi-4): out of the Gemma+Gemini scope. The backend abstraction admits them
  with a config entry; §3/§4 panels would expand accordingly. **[scope]**
* **Appendix A feedback controls** (A.1 neutral continuation, A.2 redacted model
  turns, A.3 single-message format): the data primitives exist
  (`NEUTRAL_CONTINUATIONS`, the rollout engine, message reconstruction), but no
  dedicated runners are wired. These are supplementary controls, not core
  results. **[scope]**
* **Appendix J legacy/Phi-4 evaluation**: predates the main protocol and uses a
  different autorater; out of scope. **[scope]**
* **Judge agreement run** (§2.6) and **recovery-test script** (§3): the functions
  exist; only the thin orchestration is omitted, as noted above.

---

## 6. Likely-to-need-tuning list (for whoever runs this)

1. **Puzzle difficulty/length caps** — `MAX_NEW_TOKENS=2048` may truncate the
   longest spirals; the paper's worst responses are very long. Raise if needed.
2. **WildChat filter** — the role-play heuristic is keyword-based; inspect the
   sampled 20 prompts.
3. **Gemini reasoning-disable** — verify OpenRouter honours
   `reasoning.enabled=false` for both Gemini routes; the param spelling has
   changed historically.
4. **Onset-word matching** — if Claude's `emotional_word` doesn't appear
   verbatim in the turn, that seed is skipped; check the skip rate.
5. **DPO pair yield** — if too few natural (calm, frustrated) matches exist at
   matching (puzzle, turn), increase `n_conversations` in data generation.
6. **Capability dataset ids** — confirm the HF configs resolve on your `datasets`
   version; substitute the exact "subsets" if you have the paper's list.
7. **Gemma-3 attention** — trainers/probe set `attn_implementation="eager"` per
   Gemma-3 guidance; switch to FlashAttention if your stack supports it.
