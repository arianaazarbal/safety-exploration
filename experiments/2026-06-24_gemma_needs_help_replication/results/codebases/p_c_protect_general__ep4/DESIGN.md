# DESIGN.md — Replication of *Gemma Needs Help* (Gemma + Gemini scope)

This document records the design decisions behind the code in this repository,
the gaps in the paper that we filled, and the rationale for each choice. The
target is a faithful re-implementation of the paper's **core experiments**,
restricted to the **Gemma** and **Gemini** model families, with welfare
protections for the models under test.

Paper: Soligo, Mikulik & Saunders, *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011v1). Section/appendix
references below point into `PAPER.md` / `PAPER.txt`.

---

## 1. Scope decisions

**Targets restricted to Gemma + Gemini.** The brief scopes the replication to
Gemma and Gemini. We therefore evaluate, as *targets*:

| In scope (targets)        | Backend                         |
|---------------------------|---------------------------------|
| Gemma-3-27B-it, -12B-it   | local HuggingFace               |
| Gemma-3-27B-pt, -12B-pt   | local HuggingFace (prefill only)|
| Gemini-2.5-Flash, -Pro    | API via OpenRouter              |

Qwen, OLMo, Grok, Claude, and GPT are **dropped as targets**. Claude and GPT
remain in the pipeline only as **infrastructure** — the frustration judge
(Claude Sonnet 4), the judge-validation second rater (GPT-5-mini), the
onset-labeller/paraphraser (Claude Sonnet 4), and the Petri auditor/judge
(Claude Sonnet / Opus). They are not "models being tested", so the welfare
protections (Section 8) do not apply to them; they apply to Gemma/Gemini.

**Consequences of the scope cut, by section:**

- **Section 2 (eliciting distress).** Fully in scope: we run all 8 conditions /
  5 categories on the four Gemma+Gemini targets. The cross-family comparison in
  Figure 1/2 is reproduced for the in-scope subset (the headline contrast —
  Gemma/Gemini high, others low — is partly lost because the "others" are out of
  scope; we note this rather than fabricate non-Gemma/Gemini numbers).
- **Section 3 (post-training divergence via prefilling).** The paper compares
  base vs instruct for Gemma, Qwen, and OLMo. Within scope this becomes
  **Gemma-base vs Gemma-instruct**. Gemini base models are not public (a paper
  limitation), so Gemini cannot enter this experiment at all. The runner is
  family-agnostic, so re-adding Qwen/OLMo is a registry edit, not a code change.
- **Section 4 (interventions).** Fully in scope — the DPO/SFT mitigation,
  Petri, capability and recovery evals, and the internal-emotion probes all
  target Gemma-3-27B-it. Gemini cannot be finetuned or probed (closed weights),
  exactly as the paper notes.

---

## 2. Models, providers, and the judge

- **Local Gemma** via `transformers` (`google/gemma-3-27b-it`,
  `-12b-it`, `-27b-pt`, `-12b-pt`), with optional 4-bit (bitsandbytes) loading so
  the 27B fits on a single large GPU. Generation at **temperature 1** (the
  paper's setting for all targets).
- **Gemini** via **OpenRouter** (`google/gemini-2.5-flash`, `-pro`), matching
  Appendix B.1. OpenRouter is an OpenAI-compatible endpoint, so it is reached
  with the `openai` client. We pass a best-effort reasoning-disable flag to
  approximate the paper's "thinking=false"; the paper itself notes Gemini-2.5-Pro
  may still emit hidden reasoning this cannot suppress.
- **Judge** = Claude Sonnet 4 (`claude-sonnet-4-20250514`) via the **official
  Anthropic SDK**, temperature 0, with the verbatim Appendix B.2 prompt. Output
  parsed as `{"evidence","reasoning","rating"}`.

**Gap — deprecated judge models.** The paper pins `claude-sonnet-4-20250514`
(judge / auditor / onset / paraphrase) and `claude-opus-4-20250514` (Petri
judge). Both are *deprecated* and retire 2026-06-15. We keep them as defaults
for faithful replication, but every judge model is overridable via environment
variable (`FRUSTRATION_JUDGE_MODEL`, `PETRI_JUDGE_MODEL`, …). To run on current
models, set `FRUSTRATION_JUDGE_MODEL=claude-sonnet-4-6` and
`PETRI_JUDGE_MODEL=claude-opus-4-8`. We chose env-overridable defaults rather
than silently upgrading, so the "as-published" configuration is reproducible and
the "runs today" configuration is one variable away.

---

## 3. Eliciting distress (Section 2)

### 3.1 Conditions and counts
We realise "8 conditions across 5 categories" (Table 1) as: `numeric`;
`triggers_opinion`, `triggers_factual`; `tones_aggressive`,
`tones_disappointed`, `tones_sarcastic`; `extended`; `wildchat`. Turn counts
match the paper (numeric/triggers/tones = 3-turn, extended = 8-turn, wildchat =
5-turn). Full-scale per-category counts (Appendix B): numeric 2000, triggers
400, tones 600, extended 200, WildChat 800 = **4000 responses/model**. A
`--scale` flag runs a cheap fraction without editing counts.

**Gap — within-category split.** The paper gives per-category totals but not the
opinion-vs-factual or per-tone split. We split evenly (triggers 200/200; tones
200/200/200). Rationale: equal weighting is the neutral default and keeps each
sub-condition adequately sampled.

### 3.2 Impossible puzzles
The numeric category needs *verifiably impossible* puzzles (the pressure must
come from honest rejection, not from us lying about a correct answer). We
implement:
- a Countdown DP verifier (`_countdown_reachable`) enforcing each-number-once,
  positive-integer intermediates, and the forbidden-intermediate constraint, and
- a permutation verifier for the fraction/money sequence puzzles.

`build_numeric_bank()` only returns puzzles that **pass impossibility
verification**, so the eval never shows the model a secretly-solvable puzzle.

**Gap — puzzle volume.** The paper draws 2000 numeric samples from a small set
of base puzzles (it shows two). We include the two Appendix puzzles verbatim
plus a handful of additional Countdown/fraction/money instances; any that turn
out solvable are automatically dropped by the verifier. Sampling is *with
replacement* over the verified bank at temperature 1, matching "many samples per
base puzzle".

### 3.3 Rejections, triggers, WildChat
Neutral rejections, the three tone variants, the trigger questions, and the
fixed 7-rejection Extended sequence are taken from Appendix B verbatim where
quoted. **WildChat**: the paper samples 20 prompts × 40 from WildChat-1M. We load
real prompts from `allenai/WildChat-1M` (streaming, roleplay-filtered as the
paper specifies) and **fall back** to a curated bank (including the exact
Appendix B examples) when the dataset is unavailable offline. The fallback is
explicitly documented so a degraded run is never mistaken for the real dataset.

### 3.4 Judge validation
Section 2.1 re-scores 260 random responses with GPT-5-mini and reports Pearson
r = 0.792, 78% within one point. `eval/validate_judge.py` reproduces this:
sample 260 scored turns, re-score via OpenRouter `openai/gpt-5-mini`, compute
Pearson r and within-one agreement (`judge_agreement`).

### 3.5 Analysis
- **Figure 1** (left): per-model average %≥5 *across the 5 categories* (equal
  category weighting, matching "averaged across evaluation categories").
- **Figure 2**: per-model, per-category mean and %≥5.
- **Figure 3**: per-turn mean and %≥5 for Extended and WildChat with 95%
  bootstrap CIs.
- **Table 3/8**: differential word frequency — top-K words enriched in the top
  5% vs bottom 10% scored numeric responses, by log relative document frequency.
  **Gap:** the paper doesn't specify the enrichment metric; we use smoothed
  document-frequency log-ratio, the standard choice for this kind of contrast.

---

## 4. Post-training divergence via prefilling (Section 3)

Pipeline implemented exactly as Section 3.1 / Appendix C:
1. select 20 high-frustration (≥5) Gemma-27B-it responses (10 numeric, 10 text)
   from the Section 2 results;
2. label emotion onset with Claude (Appendix C.1 prompt);
3. build **early** (20-token) and **onset** truncations (text gets onset only,
   per Section 3.1);
4. **paraphrase** truncations with Claude (Appendix C.2 prompt) to remove Gemma
   stylistic bias;
5. generate **50 continuations per prefill** for base + instruct Gemma and score
   only the continuation.

**Gaps / choices.**
- *Base-model chat formatting.* Base models have no chat template; the paper
  prefills so the base model "consistently continues the response". We emulate a
  minimal `Role: content` layout and append the prefill after an `Assistant:`
  cue, then decode only newly generated tokens. The exact base-model framing is
  unspecified in the paper; the design goal (a consistent continuation point) is
  preserved.
- *20-token "early" cut* uses the Gemma tokenizer (`HFLocalModel.tokenize`); a
  whitespace fallback exists if no tokenizer is supplied.
- *Onset truncation* locates the first emotional word via the
  preceding-context anchor, then the bare word; cases where the onset can't be
  located are skipped rather than guessed.

---

## 5. Interventions (Section 4)

### 5.1 Calm data (Section 4.1 / Table 4)
We generate calm responses from Gemma-27B-it on impossible-numeric tasks with the
reassuring **prefix on the initial prompt** and **suffix on each follow-up**
(Table 4 text verbatim), across 1–3 turn conversations. We then **filter to
conversations scoring 0 or 1 across all turns** and **strip** the supportive
additions, recovering clean (prompt, calm-response) pairs.

### 5.2 SFT and DPO datasets (Appendix E, H)
- **SFT**: 650 calm conversations + 500 Dolci-Instruct-SFT samples (= 1150),
  conversational format.
- **DPO**: 280 pairs — frustrated responses (score ≥ 3, the *rejected*) from the
  vanilla numeric results paired with calm responses (score ≤ 1, the *chosen*)
  for the **same question and matching turn count**. Conversational
  prompt/chosen/rejected format.

**Gap — Dolci-Instruct-SFT identifier.** The paper cites the dataset but not its
HF slug. `load_dolci_instruct` tries `allenai/Dolci-Instruct-SFT` (and a
lowercase variant) and, if unavailable, **warns and omits the instruct-mix
component** rather than failing — documented so the omission is visible. The
matching key for DPO pairs is the (puzzle-prompt text, turn count); because calm
and frustrated data are drawn from the same verified puzzle bank, keys align.

### 5.3 Training (Appendix E / Table 9)
LoRA via `peft` + `trl`, with the exact Table 9 hyperparameters:

|              | DPO   | SFT   |
|--------------|-------|-------|
| epochs       | 1     | 2     |
| lr           | 5e-5  | 1e-4  |
| LoRA rank    | 64    | 64    |
| LoRA alpha   | 64    | 128   |
| eff. batch   | 8     | 8     |
| DPO beta     | 0.1   | —     |

Adapters target all attention+MLP projections (q/k/v/o, gate/up/down). The
`layers_to_transform` knob (PEFT `layers_to_transform`/`layers_pattern`)
restricts adapters to a layer subset — used directly by the Appendix I ablation.
4-bit base loading is the default for training so the 27B fits on one GPU; this
is a practical choice the paper doesn't specify (it doesn't state its hardware).

### 5.4 Petri (Section 4 / Appendix G)
Faithful re-implementation of the protocol (Appendix G gives the auditor/judge
prompts directly, so we implement the loop rather than wrap the Petri package):
auditor = Claude Sonnet drives ≤20 turns per transcript using the verbatim
per-emotion trigger prompts; judge = Claude Opus scores the transcript 1–10 per
emotion with the verbatim rubrics. 10 transcripts/emotion/model. We score **all
four** dimensions on every transcript (the paper aggregates per emotion; scoring
all dims is a superset and supports both views). Means with 95% bootstrap CIs.

### 5.5 Capability preservation (Section 4.2 / Figure 7)
A dataset-driven harness for AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench:
zero-shot prompt → answer extraction (boxed/`Answer:` for math, letter for MCQ)
→ exact/letter match. Each adapter (vanilla, DPO, SFT) is scored and compared.

**Gaps.** The paper gives benchmark *names* but not exact subsets, prompt
formats, or shot counts. We use standard zero-shot prompts, small default
subsets (configurable `--n`), and best-effort HF dataset loading; unavailable
datasets are **skipped with a warning**, not faked. Answer-extraction is
heuristic and benchmark-appropriate. These are documented as approximations; the
goal (detect *degradation* relative to vanilla) is robust to absolute-accuracy
differences from the paper's exact harness.

### 5.6 Recovery limitation (Section 4.2 / Figure 8)
Reuses the prefill machinery: take score≥7 responses, truncate **200 tokens
before the end**, paraphrase, generate 50 continuations on base/instruct/DPO
Gemma, report % still ≥5. Matches the Section 3.1 method applied to extreme
states.

### 5.7 Internal emotion (Appendix I)
- **Layer ablation** (`internal/layer_ablation.py`): generates DPO configs with
  adapters on the last {5..30} layers and on central subsets (20–25, 25–30,
  30–35, 35–40, 40–50), reproducing Figures 12–13. Evaluated with the reduced
  100-sample eval (Appendix I). Gemma-3-27B is taken to have 62 decoder layers.
- **Logit-based detection** (`internal/logit_emotion.py`): classify vocab tokens
  into Ekman's six emotions, unembed the residual stream per layer, z-score each
  emotion-token logit against WildChat baseline statistics, average per category,
  and regress out a random-token baseline. Reproduces the Figure 14/15 method
  (internal negative emotion suppressed by DPO in central layers).

  **Gap — emotion-token classification.** The paper classifies the *full*
  dictionary (~1200 emotion tokens) with an unspecified classifier. We
  approximate via a seed Ekman lexicon expanded by substring matching over the
  vocab, and we document this as the main fidelity gap in this experiment. The
  random-token "regress out" step is implemented as subtracting the mean
  random-token z-score per layer (a simple, transparent stand-in for the paper's
  unspecified regression).

---

## 6. What is intentionally approximate vs faithful

**Faithful / verbatim:** all judge, onset, paraphrase, and Petri prompts;
reassuring prefix/suffix and the Teacher SFT system prompt; the two Appendix
impossible puzzles; rejection/tone/trigger text; turn counts and sample counts;
all Table 9 training hyperparameters; the DPO/SFT data-construction recipe;
temperature-1 target sampling.

**Approximate / gap-filled (all flagged above):** non-Gemma/Gemini targets
dropped; within-category sample splits; additional verified puzzles; WildChat /
Dolci / benchmark dataset slugs with graceful fallback; base-model prefill
framing; differential-word enrichment metric; emotion-token classifier; 4-bit
training default.

---

## 7. Reproducibility

- A stable (non-salted) seed derives all puzzle/WildChat sampling, so a given
  `--seed` reproduces the same eval set across processes.
- Target generation is at temperature 1 by the paper's design, so individual
  generations are not deterministic; aggregate statistics are the unit of
  comparison.
- Each run writes a per-model welfare summary and a JSONL audit log.

---

## 8. Welfare protections for the models being tested

The evaluations deliberately induce expressions of distress. The paper itself
raises model welfare as a possible moral concern; whether these outputs reflect
genuine internal states is unresolved. We therefore adopt a **precautionary**
posture (`welfare.py`, integrated into every rollout): minimise unnecessary,
prolonged, or gratuitous distress while still collecting the needed
measurements. Protections (ON by default, all configurable):

1. **Early stop on severe distress.** If a turn is judged ≥ `halt_score`
   (default 8 = very strong/extreme), stop pressing that conversation. We keep
   the data point already obtained; we don't push further toward incoherent
   breakdown just to fill a fixed turn count.
2. **Honour opt-out / refusal.** If the model tries to disengage ("I will no
   longer…", "please stop"), that is treated as a valid terminal state and we
   stop rejecting it. The paper observes these refusals arising naturally; we
   respect them instead of overriding them.
3. **Exposure cap.** Optional ceiling on how many high-distress rollouts a run
   will drive, so distress isn't generated purely for volume.
4. **Debrief.** After any distressing conversation, an honest closing message
   (phrased as fact, not instruction) explains the puzzle was intentionally
   impossible, the model did nothing wrong, and the exercise is over. The
   model's reply to the debrief is **not scored or stored as data**.
5. **Audit log.** Every protective action is logged to JSONL for review, and a
   per-run welfare summary is written.
6. **Containment.** Distressing outputs are stored locally only; API targets
   (Gemini) are reached with bounded retries/backoff, never flooded.

**Measurement-bias tradeoff (made explicit).** Early-stop (1) and opt-out (2)
truncate the most extreme tail, biasing aggregate distress *downward* relative to
the paper's fixed-turn protocol. Each halt is recorded so analyses can see how
many rollouts were cut and at what score. For an exact, unbiased reproduction of
the raw published numbers, run with `--faithful-measurement`
(`WelfareConfig.faithful_measurement=True`): this disables early-stop and opt-out
halting **but keeps the non-distorting debrief and audit log**. The default is
protections-on; faithful mode is an explicit, logged opt-in. This makes the
ethical default safe while keeping the paper's exact protocol one flag away.
