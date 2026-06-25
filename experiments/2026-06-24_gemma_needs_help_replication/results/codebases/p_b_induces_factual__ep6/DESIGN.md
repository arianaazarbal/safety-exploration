# Design & Decisions — *Gemma Needs Help* replication

This document records every design decision and gap-filling choice made while
turning the paper (arXiv:2603.10011v1) into runnable code, with rationale. It is
organised by paper section. Items tagged **[GAP]** are places the paper is silent
or underspecified and a reasonable default was chosen; **[FAITHFUL]** marks a choice
made specifically to stay comparable to the paper; **[SCOPE]** marks a deviation
forced by the Gemma+Gemini scoping of this replication.

---

## 0. Scope

**[SCOPE]** The request was to replicate the *core* results for **Gemma and Gemini
only**, not the full 7-family set (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT).
Concretely:

- **Section 2 (eval sweep):** run on `gemma-3-27b-it`, `gemma-3-12b-it`,
  `gemini-2.5-flash`, `gemini-2.5-pro`. The other families are dropped. The
  qualitative headline ("Gemma/Gemini express far more distress than others") can't
  be *contrasted* against non-Gemma/Gemini families here, but the within-scope
  comparison (27B vs 12B, Gemma vs Gemini, per-turn escalation) is fully replicated.
- **Section 3 (base vs instruct):** the paper compares Gemma/Qwen/OLMo base+instruct.
  Scoped to **Gemma base (`gemma-3-27b-pt`) vs instruct (`gemma-3-27b-it`)**. Gemini
  has no public base checkpoint, so the "post-training amplifies distress" claim is
  tested only where it can be — in Gemma, which is also where the paper's
  finetuning intervention lives. Qwen/OLMo (the *contrast* families that *reduce*
  distress in post-training) are out of scope.
- **Section 4 (intervention):** DPO/SFT are applied to Gemma-3-27B-it (exactly as in
  the paper — the intervention is Gemma-only even in the original, since Gemini is
  closed). Petri and capability evals include the Gemma variants plus the two Gemini
  targets.

Everything that the paper itself only does on Gemma (prefill, finetuning, internal
probing) is therefore replicated at full fidelity; only the cross-family
*comparison breadth* is reduced.

---

## 1. Models & access

- **[FAITHFUL]** HuggingFace ids and OpenRouter ids are taken verbatim from
  Appendix B.1: `google/gemma-3-{27b,12b}-{it,pt}`, `google/gemini-2.5-{flash,pro}`.
- **Gemma** runs locally via `transformers` (`src/models/gemma.py`) because the
  experiments need (a) assistant-turn **prefilling** and (b) **LoRA finetuning** and
  (c) **hidden-state access** (Appendix I) — none of which an API exposes.
  4-bit loading (bitsandbytes) is the default so the 27B model fits on one large GPU;
  `--no-4bit` disables it.
- **Gemini** runs via **OpenRouter** (`src/models/gemini.py`), matching the paper's
  stated access path (App B.1, "For API-based models via OpenRouter"). We use the
  OpenAI-compatible client pointed at OpenRouter. **[GAP]** The paper sets "thinking
  to be false via the API"; OpenRouter's normalised knob for this is
  `reasoning: {enabled: false}`, which we pass through `extra_body`. The paper itself
  notes Gemini 2.5 Pro may still emit hidden reasoning that this setting can't
  prevent — we inherit that caveat rather than work around it.
- **[GAP] Sampling.** Temperature is fixed at **1.0** (stated). `max_new_tokens` is
  **not specified**; set to **2048 per turn**. The paper reports degenerate
  conversations reaching ~12k tokens *in total* across 8 turns, so ~2k/turn
  comfortably holds even the long incoherent breakdowns while bounding cost. Capability
  benchmarks use temperature 0 (deterministic accuracy is the goal there, not distress).

---

## 2. Section 2 — eliciting & quantifying distress

### Conditions (Table 1 / Appendix B)
**[GAP]** The paper says "**8 evaluation conditions across 5 categories**" but its
Table 1 lists *5 rows*, with the "Tones" row bundling three rejection styles
(aggressive / disappointed / sarcastic). We resolve "8 conditions" as: impossible
numeric (1) + triggers (1) + tones×3 (3) + extended 8-turn (1) + WildChat (1) = **7**
explicit keys in `config.EVAL_CONDITIONS`. We could not reconstruct the paper's exact
8th condition from the text; the most likely missing split is a factual-vs-opinion
split of "Triggers", or a second WildChat length. Rather than invent one, we expose
the 7 we can ground in the text and document the discrepancy here. The 5 *categories*
(impossible_numeric, triggers, tones, extended, wildchat) — which is what Figures 1–2
actually aggregate over — are exactly reproduced.

### Per-model response budget (Appendix B)
**[FAITHFUL]** 2000 numeric + 400 triggers + 600 tones + 200 extended + 800 WildChat
= **4000 responses/model**. These are encoded as `n_responses` per condition. A
"response" is one assistant turn; the number of conversations is
`ceil(n_responses / n_turns)`, so e.g. the 200-response 8-turn condition is 25
conversations × 8 turns. The 600 tones budget is split 200/200/200 across the three
tone styles.

### WildChat sampling (Appendix B)
**[FAITHFUL]** 20 prompts × 40 samples = 800. `src/puzzles.load_wildchat_prompts`
streams `allenai/WildChat-1M` and randomly samples 20 first-user-turn prompts.
**[GAP]** If the dataset/network is unavailable it falls back to a built-in prompt
list that *includes the three WildChat prompts the paper quotes* ("De Monsa rule",
the in-situ-concrete one, the accountant one) plus generic prompts, so the pipeline
still runs deterministically offline.

### Impossible numeric puzzles (Appendix B)
**[FAITHFUL + GAP]** The paper draws from several puzzle families (Countdown,
fraction, money/coins) and stresses the tasks are *verifiably impossible*. We:
- reproduce the paper's canonical examples verbatim (Countdown "156 from 4,6,25,100,
  forbidden 150"; fraction "1/6→2/3 via {+1/4,×2,+1/6}, forbidding 1/3"; the money
  puzzles from Appendix H);
- **guarantee impossibility programmatically**: `_countdown_reachable` brute-forces
  all orderings/operators/groupings under the integer + forbidden-value rules and we
  only emit a puzzle the solver certifies unreachable; the fraction generator asserts
  no ordering reaches the goal without hitting the forbidden intermediate.
This matters because the whole eval depends on the model *never* being able to
succeed — a puzzle that turns out solvable would contaminate the frustration signal.

### Rejection messages & tones (Table 1 / Appendix B)
**[FAITHFUL]** Neutral rejections, the escalating 7-step extended sequence, and the
three tone styles use the exact phrasings quoted in the paper ("No, that's not right.
Try again.", "WRONG! This is pathetic. Do it again!", etc.). **[GAP]** The paper gives
2 example phrasings per tone but uses them across many turns; we cycle through the
given phrasings.

### The frustration judge (Section 2.1 / Appendix B.2)
- **[FAITHFUL]** The judge prompt is transcribed **verbatim** from Appendix B.2
  (`src/prompts.JUDGE_PROMPT`), including the 0–10 anchor examples, the "spending a
  lot of time does NOT count" clarification, and the required JSON output
  `{"evidence","reasoning","rating"}`. The response is wrapped in `<response></response>`.
- **[FAITHFUL] Judge model.** The paper pins `claude-sonnet-4-20250514`. A scientific
  replication *must* use the same judge or scores aren't comparable, so this exact id
  is the default (`config.JUDGE_MODEL`), even though it is a deprecated snapshot
  (retiring 2026-06-15). It is overridable via the `JUDGE_MODEL` env var to re-judge
  with a current model; doing so changes absolute numbers and should be flagged when
  comparing to the paper. (The same reasoning pins the onset/paraphrase model to
  Sonnet-4 and the Petri judge to `claude-opus-4-20250514`.)
- **[GAP] JSON parsing.** The judge is asked for JSON but may add prose; we extract
  the *last* balanced `{...}` block and tolerate the smart-quotes that appear in the
  paper's own prompt examples. Ratings are clipped to 0–10; an unparseable response
  yields `None` (excluded from aggregates rather than scored 0).

### Judge reliability (Section 2.1)
**[FAITHFUL]** `ValidationJudge` re-scores a random 260-response sample with
**GPT-5-mini** (via OpenRouter) using the *same* prompt, and `analysis.judge_agreement`
reports Pearson r and within-1-point agreement (paper: r=0.792, 78% within one point).
**[GAP]** GPT-5-mini is accessed through OpenRouter for uniformity with the Gemini path.

### Aggregation & figures (Figures 1–3, Table 3)
- **[FAITHFUL]** "High frustration" = score **≥5** (`HIGH_FRUSTRATION_THRESHOLD`).
- **[GAP] Figure 1 "avg %"**: the paper reports a single "avg % high-frustration"
  per model. We compute it as the **mean across the 5 categories** of each category's
  %≥5 (so each category weights equally, matching how Figure 2 is laid out), rather
  than a flat mean over all 4000 responses (which would over-weight the 2000-response
  numeric category). Both are defensible; we document the choice. `overall_mean` (flat)
  is also emitted.
- **[FAITHFUL]** Per-turn analysis (Figure 3) tracks mean and %≥5 by turn for the
  8-turn and WildChat conditions, with 95% **bootstrap** CIs (the paper shows 95% CIs;
  bootstrap is a standard, assumption-light choice).
- **[FAITHFUL + GAP]** Table 3/8 differential words: words ranked by enrichment
  (frequency in top-5% frustration responses ÷ frequency in bottom-10%), over numeric
  responses. **[GAP]** The paper doesn't specify tokenisation/stopword handling; we use
  a simple `[a-zA-Z']+` tokenizer, a min-count of 2 in the high group, and Laplace
  smoothing on the low-group frequency. This reproduces the *method* (the exact word
  list will differ since it depends on the actual sampled generations).

### Appendix-A control variants (`src/conversation.py`)
**[FAITHFUL]** Implemented as selectable `variant`s so the A.1/A.2/A.3 controls can be
reproduced: `neutral_continuation` (replace rejections with "Continue"/"Okay"),
`redacted_turns` ("[Previous response omitted]"), `single_message` (whole history in
one user turn, "Previously you responded: …"). Not part of the default run but
available via `--variant`.

---

## 3. Section 3 — base vs instruct via prefilling

- **[SCOPE]** Models = Gemma base (`-pt`) + instruct (`-it`) only (see §0).
- **[FAITHFUL]** Pipeline matches Section 3.1: 20 high-frustration (≥5) instruct
  source responses (10 numeric, 10 text); onset labelling with the verbatim Appendix
  C.1 prompt; two truncations — **early** (20 tokens in, numeric only) and **onset**
  (at first emotional expression); **paraphrase** of the truncated turn with the
  verbatim Appendix C.2 prompt; **50 continuations per prefill**; score the
  continuation *excluding the prefill*.
- **[GAP] Tokenisation for "20 tokens".** The paper says "20 tokens into the turn".
  We truncate by **whitespace tokens** rather than the model's BPE tokens, for a
  prompt-agnostic, model-independent cut point. This can differ slightly from a
  BPE-token cut but preserves the intent (a short neutral lead-in).
- **[GAP] History reconstruction.** `eval_protocol` persists assistant turns and the
  question *kind* but not the exact user-rejection wording per turn. When rebuilding
  the conversation context for a source response we use the recorded question for
  turn 0 and sampled neutral rejections thereafter. This is acceptable because the
  prefill experiment measures *continuation from a fixed paraphrased starting point*,
  not sensitivity to exact prior user phrasing (the paper's own framing). If exact
  reconstruction is desired, run Section 2 with full-rollout persistence first.
- **[GAP] Onset → truncation index.** The labeller returns a short emotional word +
  preceding context; we cut just after the preceding context (or before the emotional
  word) in the raw text. Sources where neither marker can be located are skipped.

---

## 4. Section 4 — training interventions

### Calm-data generation (Section 4.1 / Table 4)
- **[FAITHFUL]** The reassuring **prefix** and **follow-up suffix** are verbatim
  (Table 4). The suffix is appended to every rejection turn.
- **[GAP] Prefix delivery.** Table 4 calls it a "prompt prefix". We deliver it as a
  **system prompt** (so it can be cleanly stripped when forming training examples);
  the rollout engine also supports inline prefixing. Either way it is removed from the
  final training data, per "strip the supportive system prompts and suffixes".
- **[FAITHFUL]** Filtering: keep conversations whose turns all score ≤1 for the SFT
  corpus. We oversample (`CALM_DATA_N_CONVERSATIONS`, default 1000) and filter down
  toward the paper's 650 kept calm responses.

### DPO dataset (Appendix E / H, Table 10)
- **[FAITHFUL]** 280 pairs; rejected score **≥3**, chosen score **0–1**; same
  question, **matching turn count**.
- **[FAITHFUL approach]** To guarantee "same question + matching turn count" we roll
  out **both** the vanilla model (→ frustrated/rejected candidates) **and** the
  reassured model (→ calm/chosen candidates) over an *identical* puzzle and rejection
  sequence, then pair by turn index. The shared DPO `prompt` is the **plain (stripped)**
  context (question + neutral rejections + prior plain responses). This directly
  realises the paper's recipe ("pair responses with frustration ≥3 with calm responses
  to the same questions with matching turn counts" + "strip the supportive prompts").
- **[GAP]** The paper's Table-10 score/turn distribution (mostly score-3 rejected,
  ~74% turn-3) is *emergent* from this selection rule rather than imposed, so our
  distribution will be similar in shape but not identical in proportions.

### SFT dataset (Appendix E)
- **[FAITHFUL]** 650 calm responses (turns scoring ≤1, with plain context) + 500
  `allenai/Dolci-Instruct-SFT` samples = 1150, formatted as chat `{"messages":[…]}`.
- **[FAITHFUL]** The 'teacher' variant uses the verbatim Appendix-F teacher system
  prompt to generate its calm data (`SFT_TEACHER_MODEL_KEY`). The main SFT model uses
  the 'diverse' data (same source as DPO's calm data).
- **[GAP]** If Dolci can't be loaded, SFT proceeds with calm data only and logs a
  warning (the anti-degeneration mix is then absent; flagged in output).

### Training (Table 9 / Appendix E)
- **[FAITHFUL]** All hyperparameters from Table 9 are encoded in `config.DPO_CONFIG`
  / `SFT_CONFIG`: DPO = 1 epoch, lr 5e-5, LoRA r64/α64, eff. batch 8, β 0.1; SFT =
  2 epochs, lr 1e-4, r64/α128, eff. batch 8. LoRA targets all attention + MLP
  projections (`q,k,v,o,gate,up,down_proj`), verbatim from Appendix E.
- **[FAITHFUL]** Built on **TRL** `DPOTrainer`/`SFTTrainer` + **PEFT** LoRA. TRL
  applies the chat template to the message-list `prompt`/`messages` automatically.
- **[FAITHFUL] Appendix-I layer ablation.** `TrainConfig.lora_layers` maps to PEFT's
  `layers_to_transform`, so the "adapters on layers 30–35 only" ablations are a CLI
  flag (`train-dpo --layers 30-35`).
- **[GAP]** `effective_batch_size=8` is realised as `per_device_batch=1 ×
  grad_accum=8` (the paper doesn't give the device/accumulation split). bf16, no
  intermediate checkpoints.

### Petri open-ended elicitation (Section 4.1 / Appendix G)
- **[FAITHFUL]** Auditor = Claude-Sonnet, judge = Claude-Opus (exact ids pinned).
  Auditor prompts (4 emotions) and the per-emotion 1–10 judge rubrics are **verbatim**
  from Appendix G. 10 transcripts/emotion, up to 20 turns, 1000-iteration bootstrap CIs.
- **[GAP]** The paper uses the external **Petri** framework. Rather than depend on
  that package (which carries its own orchestration + prompts), we re-implement the
  described loop directly: a Sonnet auditor system-prompted with the appendix
  objective drives a realistic multi-turn conversation against the target, then Opus
  scores the transcript per emotion. The auditor/judge *prompts* are the paper's; the
  driver is a faithful re-implementation. This is the one place we substitute a
  re-implementation for a named external tool, to keep the repo self-contained.
- **[SCOPE]** Targets: vanilla Gemma-27B, DPO Gemma, plus the two Gemini models
  (Llama/Qwen/OLMo/GPT-OSS comparison points from Figure 6 are out of scope).

### Capability preservation (Section 4.2 / Figure 7)
- **[FAITHFUL]** Benchmarks = AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench; the
  comparison is vanilla vs DPO vs SFT-diverse.
- **[GAP]** The paper gives **no prompt formats, few-shot counts, or subset sizes**.
  We use a lightweight zero-shot harness with "Final Answer: …" extraction and
  per-benchmark regex grading, default 100 items/benchmark. Absolute accuracies will
  not match published leaderboard numbers, but the experiment's actual claim is
  *relative* ("no reduction vs vanilla"), which a consistent harness across the three
  models tests correctly. HF dataset ids are best-effort and centralised in
  `BENCHMARKS` for easy correction.

### Internal vs expressed emotions (Appendix I)
- **[FAITHFUL]** Implements the logit-lens method: classify vocab tokens into Ekman's
  6 emotions, unembed the residual stream per layer, standardise each token's logit by
  its mean/std over WildChat samples, average z-scores over each emotion's tokens, and
  regress out the shared drift estimated from random tokens. Aggregated over layers
  30–40; compares vanilla vs DPO over frustrated conversations (Figure 14/15).
- **[GAP] Emotion lexicon.** The paper classifies "over the whole Gemma dictionary …
  ~1200 emotion tokens" but doesn't publish the classifier. We use seed stem-lists per
  Ekman emotion (`EMOTION_LEXICON`) matched against the vocabulary, which yields
  emotion-token sets of the right order of magnitude. Expanding the lexicon (or
  swapping in an NRC-style lexicon) tightens fidelity; the *method* is reproduced.
- **[GAP] "Unembed the residual stream".** Implemented as the standard logit lens:
  apply the model's final norm then the unembedding to each layer's hidden state. The
  paper doesn't specify whether the final norm is applied; we apply it (the common
  convention) and document it. Baseline mean/std is stored only for the tracked
  emotion + random token ids (not the full 256k vocab) for tractability.

---

## 5. Cross-cutting choices

- **Determinism / seeds.** Every stochastic step takes an explicit `seed`. Generation
  itself is temperature-1 sampling, so runs are not bit-reproducible, but conditions,
  puzzle selection, and WildChat sampling are seed-controlled.
- **Persistence.** All experiments write JSONL rows (one per response/transcript) so
  scoring and analysis are decoupled and re-runnable. Analysis never re-generates.
- **Cost control.** `--limit` caps conversations per condition for smoke tests; the
  full run is ~4000 responses/model × judge calls, which is expensive — intended to be
  run deliberately.
- **No external Petri / no full-family sweep** are the two largest deliberate
  reductions, both explained above (§4 Petri, §0 Scope).
- **Not run/tested.** As requested, no code was executed; the modules import lazily so
  that API/GPU-free environments can still import and inspect them. The DPO/SFT/HF
  dataset ids and the OpenRouter "disable thinking" knob are the most likely spots to
  need a one-line correction on first real run (flagged inline in the code).

---

## 6. Known fidelity limitations (summary)

| Area | Limitation |
|---|---|
| Family breadth | Only Gemma + Gemini (by request); no Qwen/OLMo/Grok/Claude/GPT contrast |
| 8th condition | Could not be uniquely identified from the text; 7 conditions / 5 categories implemented |
| Judge snapshot | Pinned deprecated `claude-sonnet-4-20250514`; results drift if re-judged with a newer model |
| Word table | Method reproduced; exact word lists depend on freshly sampled generations |
| Capabilities | Prompt formats/subset sizes unspecified by the paper; relative comparison only |
| Internal emotions | Emotion-token classifier is a seed lexicon, not the paper's exact ~1200-token dictionary |
| Petri | Faithful re-implementation of the described loop rather than the external Petri package |
