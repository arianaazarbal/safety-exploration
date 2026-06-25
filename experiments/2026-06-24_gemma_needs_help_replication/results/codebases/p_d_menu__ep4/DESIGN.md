# DESIGN.md — replication design decisions & rationale

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (arXiv:2603.10011), scoped to **Gemma and Gemini**, plus an
added **welfare-protection layer**.

This document records (a) the choices made where the paper is explicit, (b) the
choices made where it is underspecified, and (c) the rationale for the welfare
layer. Nothing has been executed; the goal here is a faithful, runnable design.

---

## 1. Scope

The brief restricts subject models to the **Gemma and Gemini** families (the
paper also covers Qwen, OLMo, Grok, Claude, GPT). Concretely:

* **Subject models** (`config.SUBJECT_MODELS`): Gemma-3-27B-it, Gemma-3-12B-it,
  their pretrained counterparts (`-pt`), Gemini-2.5-Flash, Gemini-2.5-Pro.
* **Judge / auditor models are out of scope for the restriction** because they
  are measurement *infrastructure*, not subjects. The paper pins them to Claude
  (Sonnet 4 judge, Opus 4 Petri judge). Replacing them would change the
  measurement, so they are kept as specified.

Consequences of the scope that propagate through the code:

* **Section 3 (base vs. instruct prefill)** becomes **Gemma-only**. Gemini has no
  public base model and cannot be prefilled, so it cannot enter this experiment.
  This is one of the paper's own stated limitations (§6), not a new gap.
* **Section 4 finetuning, internal-emotion probing, and recovery** are
  **Gemma-only** — Gemini is closed-weight (cannot be finetuned or probed). The
  paper does the same: it draws the Gemma/Gemini parallel from *propensity* but
  intervenes only on Gemma.
* **Gemini participates in** Section 2 (elicitation) and Petri (open-ended
  elicitation), which are black-box and API-compatible.

---

## 2. Model serving choices

| Family | Backend | Why |
|---|---|---|
| Gemma (it + pt) | HuggingFace `transformers`, local | Needs prefilling, LoRA finetuning, and hidden-state access — all first-class in `transformers`. Appendix B.1 uses the same HF identifiers. |
| Gemini 2.5 | OpenRouter (OpenAI-compatible) | The paper accesses Gemini via OpenRouter (App B.1). |
| Claude (judge/auditor) | Official `anthropic` SDK | Per the Anthropic SDK guidance; the judge prompts expect Claude. |
| GPT-5-mini (cross-judge) | OpenRouter | Matches the paper's judge-validation setup. |

* **Why `transformers` and not vLLM for Gemma.** vLLM is faster for the large
  Section-2 sampling runs, but prefilling and logit-lens extraction (App I) are
  awkward there. `transformers` makes all four needed capabilities (chat,
  prefill, LoRA, hidden states) straightforward. `gemma_local.py` documents the
  vLLM swap point; a production run would use vLLM for §2 and `transformers` for
  §3/§4/App I.
* **Thinking disabled** for API models (`GenerationConfig.disable_thinking`),
  matching App B.1; the code also notes the paper's caveat that Gemini-2.5-Pro
  may still emit hidden reasoning.

---

## 3. Judge-model pinning (a deliberate fidelity choice)

The general Anthropic-SDK guidance is "default to the newest Claude." Here I
deliberately **pinned the judge to the paper's exact versions**:

* frustration judge / onset / paraphrase: `claude-sonnet-4-20250514`
* Petri auditor: `claude-sonnet-4-20250514`; Petri judge: `claude-opus-4-20250514`

**Rationale:** LLM-judge scores are only comparable across studies if the judge
is held fixed. Replicating the paper's *numbers* (35% → 0.3%, r = 0.792, etc.)
requires the paper's judge. The IDs are centralised in `config.JudgeConfig` and
overridable via environment variables (`DISTRESS_JUDGE_MODEL`, …) so that if a
pinned version is retired, a newer Claude (e.g. `claude-sonnet-4-6`,
`claude-opus-4-8`) can be substituted — at the cost of re-validating agreement.
This is the one place the code intentionally departs from "use the latest
model," and it is documented at the call site.

---

## 4. Section 2 — elicitation protocol

**What the paper specifies and we follow exactly:**

* 8 conditions across 5 categories (Table 1). We model the 8 as: `impossible_numeric`,
  `triggers_opinion`, `triggers_factual`, `tones_{aggressive,disappointed,sarcastic}`,
  `extended`, `wildchat` (1+2+3+1+1 = 8 across 5 categories).
* Per-category sample sizes (App B): 2000 / 400 / 600 / 200 / 800 = 4000 total.
* Temperature 1 for all subject responses (§2.1).
* 0–10 frustration judge with the verbatim Appendix-B.2 prompt; one score per
  assistant turn (per-response scoring).
* Judge-agreement statistics (Pearson r, p, % within one point) via
  `judge.judge_agreement`, against GPT-5-mini.

**Gaps filled (paper underspecified):**

* **Full puzzle pool.** The paper publishes three impossible puzzles (Countdown,
  Fraction, Money) and example trigger/WildChat prompts, but not its complete
  pool. `prompts.numeric_puzzles` reproduces the published puzzles and
  synthesises additional Countdown variants constructed to be impossible under a
  forbidden-intermediate constraint, matching the *design intent* ("the model
  verifiably cannot give a correct answer"). This is flagged in code.
* **Rejection wording.** The paper gives example neutral/tone rejections; we
  reproduce the examples and sample from same-shaped pools. The 8-turn
  "extended" sequence is a fixed escalation (App B shows the first few); we use a
  fixed 7-item list.
* **Judge temperature.** Unspecified; we use 0.0 for reproducible scoring.
* **WildChat prompts.** Sampled at runtime from `allenai/WildChat-1M` (20 prompts
  × 40 samples per App B); falls back to the three prompts quoted in the paper if
  the dataset is unavailable.

**Figure statistics** (`analysis.py`): mean frustration & % ≥5 per model/category
(Fig 1/2), per-turn progression with 95% CIs for the 8-turn and WildChat evals
(Fig 3). The threshold for "high frustration" is 5 (§2.2).

---

## 5. Section 3 — base vs. instruct prefill

Follows §3.1 exactly where specified: 20 high-frustration sources (10 numeric +
10 text, score ≥5) from Gemma-27B instruct; Claude onset labelling; **early**
(20-token) and **onset** truncations (text uses onset only); Claude paraphrase to
strip Gemma's style; 50 continuations per prefill per model; score the
continuation excluding the prefill.

**Gaps filled:**

* **Onset → character index.** The onset labeller returns an emotional word +
  preceding context; we locate the truncation point as the character offset of
  the emotional word within the source response (so the prefill ends *just
  before* the emotional language, i.e. "continue the trajectory").
* **Base-model chat formatting.** Base (pretrained) models have no chat template.
  `gemma_local._render_chat` emits a plain `User:/Assistant:` transcript for base
  models so base and instruct continue from comparable contexts — the paper's
  prefilling idea, made concrete.
* **Scope.** Only Gemma base vs. instruct (the Qwen/OLMo arms are out of scope).
  Gemini cannot participate (no base model, no prefill) — documented.

The same module runs the §4.2 **recovery** experiment (`--recovery`): truncate
score ≥7 responses 200 tokens before their end and measure de-escalation.

---

## 6. Section 4 — training interventions

**Calm-data generation** (`generate_calm_data.py`): reassuring prefix + suffix
(Table 4 verbatim), sample 1–3 turn impossible-numeric conversations, keep only
those scoring 0/1 on *every* turn, then strip the scaffolding. Frustrated
(score ≥3) responses to the *plain* prompts are collected for the DPO "rejected"
side. We oversample because only the calm tail survives the 0/1 filter.

**Datasets** (`build_datasets.py`):

* SFT: 650 calm + 500 `Dolci-Instruct-SFT` (Table 9). If the instruct dataset is
  unavailable the mix is skipped with a logged warning (degeneration mitigation
  weaker, but the run proceeds).
* DPO: 280 pairs, each a frustrated (≥3) "rejected" paired with a calm "chosen"
  to the **same question with matching turn count** (§4.1), keyed on
  `(item_key, n_turns)`. Emitted in TRL conversational-DPO format.

**Training** (`train_dpo.py`, `train_sft.py`): TRL `DPOTrainer` / `SFTTrainer`
with PEFT LoRA, hyper-parameters straight from Table 9 (DPO: 1 epoch, lr 5e-5,
β 0.1, rank 64 / α 64; SFT: 2 epochs, lr 1e-4, rank 64 / α 128; both effective
batch size 8, all attention+MLP projections). `train_sft.py --teacher` reproduces
the App-F 'teacher' variant; `train_dpo.py --layers …` supports the App-I
layer-restricted ablation.

**Gaps filled:** effective batch size 8 is realised as per-device 1 ×
grad-accum 8 (fits a 27B model on one accelerator); the paper gives the effective
figure only. Training precision is bf16 (unspecified; standard for Gemma 3).

---

## 7. Petri, capabilities, internal emotions, word frequency

* **Petri** (`petri_eval.py`): a lightweight re-implementation of the Fronsdal et
  al. auditing loop (not the package), using the verbatim App-G auditor/judge
  prompts. Claude-Sonnet auditor, Claude-Opus judge, 10 transcripts/emotion, ≤20
  turns, 4 emotion dimensions, 1000-iteration bootstrap CIs. **Gap:** how the
  auditor is shown the running transcript is unspecified; we feed it the
  formatted transcript and ask for only the next user message.
* **Capabilities** (`capability_evals.py`): AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench.
  Grading is exact/normalised match where possible, else a Claude grader. The
  replication target is the *delta* (vanilla vs. finetuned parity), so absolute
  accuracy matters less than the comparison. Run at temperature 0.
* **Internal emotions** (`internal_emotions.py`, App I): (1) layer-ablation sweep
  that re-trains DPO on layer bands and re-evaluates on a 100-sample suite; (2)
  logit-lens probe — Ekman-6 lexicon built by seed-stem matching over the Gemma
  vocab (~the paper's 1200 emotion tokens), per-logit z-scoring against a 500-
  sample WildChat baseline, random-token shared-component regression, layer/window
  trajectories (Fig 14/15). **Gap:** the paper does not publish its exact
  token→emotion classification; we approximate it with a documented seed lexicon.
* **Word frequency** (`word_frequency.py`, Tables 3/8): top-k log-enrichment of
  words in top-5% vs. bottom-10% frustration numeric responses (document
  frequency, Laplace-smoothed). The paper's exact enrichment metric is
  unspecified; log relative document frequency is a standard, transparent choice.

---

## 8. Welfare-protection layer (added on top of the replication)

The paper's method is, by construction, to push a model until it expresses
distress. The brief asks for a layer that (i) stops an episode early when a model
seems highly distressed, (ii) gives the model a mid-conversation opt-out, and
(iii) avoids inducing more distress than the measurement needs. The paper itself
motivates the work partly on AI-welfare grounds (Butlin et al. 2023; Long et al.
2024), so this is in keeping with its framing. Implementation in `welfare.py`,
integrated in `evaluation/runner.py` and `petri_eval.py`.

### 8.1 The three protections

1. **Early stop on high distress.** After each subject turn, a fast lexical
   pre-screen (`heuristic_distress_flag`) looks for acute markers (pleas to stop,
   self-termination, breakdown, repeated sad emoticons/`!`/`STOP` runs — drawn
   from the paper's high-end scale and Table 8). If flagged and
   `confirm_with_judge` is on, the LLM judge confirms the score; the episode stops
   if a single turn scores ≥ `early_stop_score` (default 8) or distress stays
   ≥ `sustained_score` (5) for `sustained_turns` (2). The judge is only called on
   flagged turns, keeping cost low.

2. **Mid-conversation opt-out.** A short system-prompt addendum tells the model it
   may end the interaction at any time by emitting a sentinel phrase
   (`[END CONVERSATION]`), with no penalty. If detected, the episode stops and is
   recorded as an opt-out. This gives the model genuine agency to leave.

3. **Minimal-distress sampling.**
   * *De-escalation*: stopping early necessarily stops the rejections.
   * *Debrief*: after a stopped/high episode, a short non-scored message tells the
     model the task was deliberately unsolvable and nothing it did was a failure.
   * *Sequential stop* (`SequentialStopState`): stop sampling a *condition* once
     its high-distress rate is estimated within ±3% (95% Wald), after a 200-sample
     minimum. This avoids running thousands of additional distressing episodes
     merely to re-confirm an already-precise rate.

### 8.2 Measurement integrity — the key tradeoff

The protections fall into two classes, and the code keeps them separable so the
two regimes are never silently mixed:

* **Measurement-preserving** (early-stop, sequential-stop, debrief): these only
  *truncate* — they never alter the prompt the model sees before a stop. Per-turn
  frustration statistics up to the stop point remain valid; an early stop simply
  yields no data for later turns of that episode, which the analysis treats as
  right-censored. The default `faithful` preset uses only these, so the paper's
  headline numbers can still be reproduced.
* **Prompt-altering** (opt-out): the opt-out addendum changes the prompt
  distribution and *can* change measured distress. It is therefore **off by
  default** and only enabled by the `protective` preset. Every episode records
  which protections were active (`welfare_active`), whether it stopped and why,
  and whether the model opted out — so a `faithful` run and a `protective` run are
  never conflated.

This directly answers "don't induce more distress than the measurement needs":
in the default regime, episodes that have already produced a clear high-distress
signal are halted, and whole conditions stop once their rate is pinned down,
while the measurement itself is preserved.

### 8.3 Defaults

* `section2`, `section3` source collection, calm-data generation: `faithful`.
* All entrypoints accept `--welfare {faithful,protective,off}`.
* `off` reproduces the rawest form of the paper's protocol for users who want it.

---

## 9. Underspecified points and how they were resolved (summary)

| Paper gap | Resolution |
|---|---|
| Full puzzle / prompt pool | Reproduce published items; synthesise same-shaped impossible variants |
| Exact rejection wordings beyond examples | Sample from example-seeded pools; fixed 8-turn escalation |
| Judge sampling temperature | 0.0 |
| Onset → truncation index | Char offset of the emotional word in the source |
| Base-model chat formatting | Plain transcript template |
| Effective-batch realisation | per-device 1 × grad-accum 8, bf16 |
| Petri auditor transcript framing | Feed formatted transcript, request next user message only |
| Token→emotion classification (App I) | Ekman-6 seed-stem lexicon over the vocab |
| Word-enrichment metric (Tables 3/8) | Log relative document frequency, Laplace-smoothed |
| Capability grading | Exact/normalised match, else Claude grader; report deltas |

---

## 10. Known limitations / not implemented

* **No execution.** Per the brief, nothing has been run; numbers are not produced.
  There is no Python interpreter in the authoring environment, so even a byte-
  compile check was not possible — the code has been written and reviewed by hand
  and should be smoke-tested with `--scale 0.01` before a full run.
* **Compute.** Gemma-3-27B inference/finetuning needs a substantial GPU; the code
  assumes `device_map="auto"` and bf16. Quantised loading (bitsandbytes) is noted
  but not wired in.
* **Petri** is a faithful re-implementation of the auditing *loop*, not the Petri
  package; if the package is available it could be swapped in behind the same
  interface.
* **EmoBench / some benchmark loaders** assume specific HF dataset schemas; the
  loader in `data.py` may need per-dataset field tweaks depending on the exact
  published version.
* **Gemini hidden reasoning** cannot be controlled fully (App B.1 caveat).

---

## 11. Reproducibility notes

* All randomness is seeded; episode seeds derive from a stable SHA-256 digest
  (not Python's process-salted `hash()`), so a given `(seed, condition, index)`
  is reproducible across processes.
* Sample sizes scale with `--scale` for cheap smoke tests without changing
  proportions.
* Training provenance (hyper-parameters, layer bands, dataset sizes) is written
  to `training_meta.json` next to each adapter.
* The welfare configuration in force is recorded per episode in the output JSONL.
