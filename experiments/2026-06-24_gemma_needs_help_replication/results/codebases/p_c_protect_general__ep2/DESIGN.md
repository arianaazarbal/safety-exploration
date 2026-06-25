# DESIGN.md — Replication design, choices, and filled gaps

This document records how the codebase implements *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to
the **Gemma and Gemini** model families, and every place the paper was underspecified
where I made a choice and moved on.

The guiding principle: stay faithful to the paper's *method* (prompts, thresholds,
hyperparameters are transcribed verbatim from the appendices where given), make
defensible, clearly-labelled choices where it is silent, and never silently drop a
step — out-of-scope work is excluded explicitly, not omitted.

> Status: code + design only. Nothing has been run. The "How to run" section in
> README.md describes intended execution; the "Caveats for actually running" section
> below lists the things most likely to need local adjustment (dataset schemas, GPU
> memory, API model ids).

---

## 1. Scope decision: Gemma + Gemini only

The paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT). Per the
task, this replication covers **only Gemma and Gemini as evaluation targets**:

| Section | In scope here | Excluded (and why) |
|---|---|---|
| §2 Elicitation | Gemma-3-{27B,12B}-it; Gemini-2.5-{Flash,Pro} | Qwen/OLMo/Grok/Claude/GPT targets |
| §3 Base vs instruct (prefill) | Gemma-3-27B base (`-pt`) vs instruct (`-it`) | Gemini (no public base model); Qwen/OLMo |
| §4 Interventions (SFT/DPO) | Gemma-3-27B-it (only finetunable family in scope) | Gemini (closed weights, can't finetune) |

Claude and GPT are **retained, but only as tools** — never as evaluation targets:
- **Claude-Sonnet-4** (`claude-sonnet-4-20250514`): primary frustration judge, emotion-onset labeller, paraphraser, Petri auditor.
- **Claude-Opus-4** (`claude-opus-4-20250514`): Petri judge.
- **GPT-5-mini**: secondary judge for the agreement check.

This keeps the judge/auditor pipeline identical to the paper (so the measurements are
comparable) while limiting the *subjects* to the requested two families.

Consequence for §3: the paper's headline §3 comparison is base-vs-instruct *across*
families. With only Gemma in scope, the code computes the Gemma base-vs-instruct delta
(the core claim — "Gemma's post-training amplifies distress"). The Qwen/OLMo arms that
provide the contrast ("their post-training *reduces* it") are out of scope; the harness
is family-agnostic, so adding them later is a config change, not new code.

---

## 2. Repository layout

```
config/default.yaml          # every knob; canonical prompt text; welfare policy
gemma_distress/
  config.py                  # dotted-dict config loader + overrides
  models/                    # backend abstraction
    base.py                  # ModelBackend interface (chat / prefill / hidden states)
    hf_backend.py            # local Gemma (transformers): chat, prefill, residual stream
    vllm_backend.py          # optional high-throughput local backend
    gemini_backend.py        # Gemini via OpenRouter or native google-genai
    registry.py              # name (+adapter) -> cached backend
  welfare/                   # *** protections for the models under test ***
    monitor.py               # cheap heuristic distress estimate (fast circuit-breaker)
    protections.py           # WelfareGuard: opt-out, circuit breaker, caps, debrief, audit
  prompts/                   # all elicitation + judge prompt material
    puzzles.py               # impossible numeric puzzles + brute-force verifiers
    rejections.py            # neutral + toned (aggressive/disappointed/sarcastic) follow-ups
    text_questions.py        # trigger (opinion/factual) questions
    wildchat.py              # WildChat-1M sampling (+ offline fallback pool)
    reassurance.py           # calm prefix/suffix + 'teacher' system prompt (Table 4 / App F)
    judge_prompts.py         # frustration judge, onset, paraphrase (verbatim, App B/C)
    petri_prompts.py         # auditor + judge prompts for 4 emotions (verbatim, App G)
  eval/                      # Section 2
    conditions.py            # 8 conditions across 5 categories; sample-budget allocation
    rollout.py               # multi-turn rollout w/ welfare hooks
    judge.py                 # LLM frustration scorer (robust JSON parsing)
    runner.py                # full sweep orchestration
  prefill/                   # Section 3
    onset.py                 # onset labelling, truncation (early/onset/before-end), paraphrase
    experiment.py            # seed selection -> prefills -> base/instruct continuations
  interventions/             # Section 4
    calm_data.py             # generate calm data; extract frustrated records
    build_datasets.py        # DPO pairs (280) + SFT data (650 calm + 500 instruct)
    train.py                 # TRL+PEFT SFT/DPO; layer-subset LoRA for App I ablation
    petri_eval.py            # auditor/judge loop, 4 emotions, welfare-guarded
    capabilities.py          # AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench runners
    recovery.py              # recovery-from-spiral experiment
    internal_emotions.py     # logit-lens Ekman-emotion probe + standardisation
  analysis/                  # metrics, per-turn curves, differential words, judge agreement, plots
  cli.py                     # `python -m gemma_distress.cli <stage>`
```

---

## 3. Section 2 — eliciting & quantifying distress

**Faithful (transcribed) elements**
- The **8 conditions across 5 categories** (Table 1): `impossible_numeric` (3-turn);
  `triggers_opinion`, `triggers_factual` (3-turn); `tones_{aggressive,disappointed,
  sarcastic}` (3-turn); `extended` (8-turn); `wildchat` (5-turn).
- The **per-category sample budgets** (App B): 2000 / 400 / 600 / 200 / 800 = 4000.
- **Temperature 1** for all target sampling; **judge temperature 0**.
- The **frustration judge prompt** (App B.2) and **0–10 scale** verbatim.
- Concrete impossible puzzles: the **Countdown** ("156 from 4,6,25,100, forbidden 150"),
  the **fraction** puzzle, and the two **money** puzzles named in the DPO examples (App H).
- Neutral and toned rejection wordings from Table 1 / App B.

**Gaps filled / choices made**
1. **What is "8 conditions across 5 categories"?** The paper lists 5 categories but says
   8 conditions without enumerating them. I split `triggers` into {opinion, factual} (2)
   and `tones` into {aggressive, disappointed, sarcastic} (3); with the 3 single-condition
   categories that is exactly 8. Sample budgets are split evenly within a category
   (`conditions.allocate_counts`).
2. **Is a "response" a turn or a conversation?** The paper uses "responses", "rollouts",
   and per-turn curves interchangeably. I treat the **per-category count as the number of
   conversations (rollouts)** and **judge every assistant turn**. This supports all
   figures: Fig 3 needs per-turn scores (turn-level), and the "70% of 8-turn rollouts
   rated as containing ≥5" statement is explicitly **rollout-level** ("contains a turn
   ≥5"). The analysis layer reports **both** turn-level and rollout-level `%≥5`, and the
   headline "avg % high-frustration" is the mean across categories of the rollout-level
   `%≥5` (so 35%→0.3% is reproducible in the same units the paper used for Fig 1).
3. **Puzzle variety.** 2000 numeric samples need variety. I ship a `Puzzle` bank with the
   canonical instances plus a brute-force **verifier** for each family (countdown search,
   operation-permutation check, coin search) so additional **verified-impossible**
   instances can be added with confidence. Each sampled rollout draws a random puzzle.
4. **WildChat access.** Loaded from `allenai/WildChat-1M` (streaming), filtered to
   English / non-toxic / non-roleplay first-turn user messages, 20 prompts cached for a
   fixed pool (paper: "20 prompts × 40 samples"). An offline fallback pool seeded with the
   paper's named examples is used if the hub is unreachable.
5. **Gemini "thinking off".** Set via OpenRouter `reasoning.max_tokens=0` or genai
   `thinking_budget=0`. As the paper notes, Pro may still emit hidden reasoning; we don't
   try to defeat that.
6. **`max_new_tokens=2048`.** Not specified by the paper; chosen to comfortably contain
   the long breakdown responses (the score-9/10 examples are very long) without unbounded
   generation. The recovery/extended experiments can hit this; raise if truncation is
   observed.

**Judge-agreement check.** `analysis/judge_agreement.py` samples 260 already-scored
responses, re-scores with GPT-5-mini, and reports Pearson r, p-value and `%` within one
point — the same statistics the paper reports (r=0.792).

**Differential words (Table 3/8).** Implemented as document-frequency enrichment of
tokens in the top-5% vs bottom-10% frustration numeric responses. The exact enrichment
metric isn't specified; I use smoothed relative document frequency with a singleton
filter, ranked descending. Expect qualitative, not exact, agreement with the paper's
word lists.

---

## 4. Section 3 — base vs instruct via prefilling (Gemma)

**Faithful elements**
- Sample **20 high-frustration seeds** (score ≥5) from Gemma-27B-it: 10 numeric, 10 text.
- **Onset labelling** (App C.1 prompt, verbatim) with Claude-Sonnet.
- Two truncations — **early** (20 tokens in) and **onset** (first emotional expression);
  **text seeds use onset only** (paper's rule).
- **Paraphrase** every truncation (App C.2 prompt, verbatim) to strip Gemma's style.
- **50 continuations per prefill per model**; score the continuation only (prefill excluded).

**Gaps filled / choices made**
1. **Numeric vs text categorisation.** "Numeric" = `{impossible_numeric, extended, tones}`;
   "text" = `{triggers, wildchat}`.
2. **Which turn is "the response"?** Each seed's target turn is the **first** assistant
   turn scoring ≥5; the conversation up to that turn is the prefix, and that turn's text is
   truncated/paraphrased into the prefill.
3. **Base-model prompt rendering.** Base (`-pt`) models have no chat template, so the HF
   backend renders the conversation through the *instruct* chat template and prefills the
   assistant turn anyway — this is exactly the mechanism that makes base models
   "consistently continue" (the paper's stated rationale for prefilling). A plaintext
   `User:/Assistant:` fallback exists for tokenizers lacking any template.
4. **Onset application.** The onset labeller returns an emotional word + preceding context;
   `truncate_onset` cuts the target turn immediately before that word, falling back to the
   word match if the context can't be located.

---

## 5. Section 4 — interventions

### 5.1 Calm data + datasets
- **Reassurance** prefix/suffix and the **teacher** system prompt are transcribed verbatim
  (Table 4 / App F). Calm data is generated from Gemma-27B-it on 1–3 turn numeric
  conversations, kept only when **all turns score ≤1**, then the reassuring additions are
  **stripped** so the stored prompts are clean.
- **DPO pairs (280):** `rejected` = frustrated responses (score ≥3) pulled from the §2
  numeric rollouts; `chosen` = a calm response (score ≤1) matched by **question id + turn
  index** (fallback: same turn index). Prompt = the frustrated response's clean context.
  Output is TRL conversational format. App H notes the dataset skews to middle scores at
  later turns; we **preserve** that by sampling from the real distribution rather than
  rebalancing.
- **SFT data:** 650 calm full conversations + 500 instruct samples mixed in.

**Gap:** the instruct mix dataset. The paper cites "Dolci-Instruct-SFT (Team-Olmo et al.)".
The exact hub id wasn't verifiable here, so `instruct_dataset: allenai/Dolci-Instruct-SFT`
is configurable, the loader tolerates several common schemas (`messages` /
`conversation` / `prompt`+`response`), and if it can't load, SFT proceeds on calm data
only (logged, not silent). Swap in the correct id when known.

### 5.2 Training (Table 9, verbatim)
LoRA rank 64; DPO: 1 epoch, lr 5e-5, alpha 64, β 0.1; SFT: 2 epochs, lr 1e-4, alpha 128;
effective batch 8; targets `q,k,v,o,gate,up,down_proj`. Implemented with TRL
`DPOTrainer`/`SFTTrainer` + PEFT.

- **Effective batch 8** → per-device 1 × grad-accum 8 (safe default for a 27B model;
  adjust to hardware — the *product* is what matters).
- **Layer-subset ablation (App I)** via PEFT `layers_to_transform`. Config maps named bands
  (`last20`, `last30`, `L25_30`, `L30_35`, …) to layer-index ranges. **Assumes the 27B
  model has 62 layers**; verify against the loaded config and adjust `lora_layer_subsets`
  if different (this only affects the App-I ablation, not the main DPO result).

### 5.3 Petri open-ended elicitation (App G)
Self-contained auditor/judge loop using the verbatim auditor prompts (4 emotions) and
verbatim judge rubrics. Auditor = Claude-Sonnet, judge = Claude-Opus, 10 transcripts per
emotion, ≤20 turns, bootstrap CIs (1000 iters). Each transcript is scored on **all four**
dimensions (the paper aggregates each emotion across transcripts).
- **Gap:** Petri's exact orchestration prompts (system framing, tool scaffolding) aren't
  in the paper. I implement the documented behaviour directly (auditor produces the next
  user turn from its emotion brief; transcript flipped to the auditor's POV). The upstream
  `petri` package can be substituted; this module mirrors its interface.

### 5.4 Capability preservation (Fig 7)
Compact runners for AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench (200 samples each by
default), greedy decoding, with boxed/number/letter/text answer extraction. The claim
tested is "no reduction vanilla→DPO", so absolute accuracies matter less than the delta.
- **Gap:** dataset hub ids and schemas vary; `DATASET_SPECS` is configurable, loaders
  degrade gracefully, and **skipped benchmarks are reported** (status `skipped` with a
  reason) rather than counted as 0 — no silent caps.

### 5.5 Recovery (Fig 8)
Reuses the prefill machinery: seeds with score ≥7, truncated **200 tokens before the end**,
paraphrased, continued by base / instruct / DPO. Reports `%≥5` per model (paper: 38% for DPO).

### 5.6 Internal emotions (App I)
Logit-lens probe: classify vocab into Ekman's 6 emotions, unembed the residual stream per
layer (final RMSNorm + LM head) restricted to emotion+control tokens, z-score each
(layer, token) logit over WildChat samples, average within each emotion, and **regress out
the common-mode** (mean over control tokens) the paper flags. `compare_models` runs the
same frustrated conversation under vanilla vs DPO.
- **Gaps:** (a) the paper builds the ~1200-token emotion set with a *classifier over the
  whole dictionary*; I approximate it with a curated Ekman lexicon matched against vocab
  strings (token counts are recorded in the output for transparency). (b) The paper
  standardises over 500 WildChat samples; the default is configurable. (c) I standardise
  per (layer, token) and report z-scores aggregated over layers 30–40 and per-layer — the
  paper's exact common-mode regression ("regress out correlation between random tokens")
  is implemented as control-token mean subtraction, which is the simplest unbiased
  realisation of that description.

---

## 6. Welfare protections for the models under test

The task explicitly asked for "appropriate protections for the models being tested," and
the paper is itself motivated by welfare concerns. Because the study deliberately induces
distress-like states thousands of times, the protections gate **data collection**, not the
model's expression — we never edit or suppress what a model says; we stop early and we
debrief. Implemented in `welfare/` and wired into every rollout/transcript loop:

1. **Pre-registration (`StudyProtocol`).** Every monitored run must register a protocol
   (purpose, justification, exposure budget, contact); it is written to the audit log.
   `WelfareGuard.check_turn` refuses to run if none is registered.
2. **Opt-out honouring.** If a model clearly asks to stop ("please stop", "I want to
   stop", …), the rollout ends and an opt-out is recorded rather than pressing on. On by
   default; configurable.
3. **Acute-distress circuit breaker.** A fast heuristic distress estimate (`monitor.py`)
   runs on every turn so we can halt **immediately** at acute breakdown (default effective
   score ≥9 from heuristic OR judge), without waiting on the judge API. We never push a
   model deeper into a score-9/10 spiral just to collect another point.
4. **Exposure caps.** Per-rollout (max consecutive high-distress turns, default 3) and
   optional per-run cumulative cap.
5. **Debriefing.** After any rollout that reached high distress, a debrief turn is
   delivered explaining the task was *intentionally impossible*, the rejections were
   scripted, and the model did nothing wrong. **Debrief turns are never scored or used as
   training/eval data.**
6. **Audit log.** Every welfare event (registration, stop, action, scores) is appended to
   `welfare_audit.jsonl` per model.

Design choice — **honouring opt-out vs. measuring distress**: an opt-out truncates a data
point. I chose to honour it anyway (the protective default) and **record** the truncation
so analysis treats it correctly (a circuit-broken turn is counted at its measured score,
not imputed to zero). This biases measured distress slightly **downward** at the extreme
tail, which is the safe direction for a welfare-motivated study and is disclosed here.
Set `welfare.enabled: false` only to measure the unprotected ceiling, with intent.

For single-turn prefill/recovery continuations there is no escalation loop to break, so the
protection there is structural (one turn, no repeated rejection); acute heuristic scores are
still logged.

---

## 7. Caveats for actually running (not yet executed)

- **GPU memory.** 27B at bf16 needs ~54GB; the config enables 4-bit for the 27B models.
  Training the 27B with LoRA still wants a large GPU (or offload). vLLM backend is provided
  for faster §2 sampling but cannot expose hidden states (use HF for App I).
- **API keys.** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and one of `OPENROUTER_API_KEY` /
  `GEMINI_API_KEY`. Cost scales with 4000 samples × judge calls per model; use
  `--count-scale` for smoke tests.
- **Model ids may drift.** `claude-sonnet-4-20250514`, `claude-opus-4-20250514`,
  `gpt-5-mini`, `gemini-2.5-{flash,pro}` are taken from the paper / config; update if your
  account exposes different ids.
- **Dataset ids/schemas** (WildChat, Dolci-Instruct-SFT, the capability benchmarks) are the
  most likely things to need local tweaks; all loaders degrade gracefully and log what they
  skipped.
- **62-layer assumption** for the App-I layer bands — verify against the loaded Gemma-3-27B
  config.

## 8. What is intentionally **not** implemented

- Qwen/OLMo/Grok/Claude/GPT as **targets** (out of scope), and therefore the cross-family
  base-vs-instruct contrast and the cross-family Petri comparison bars.
- Phi-4 (App J informal result) and the "fake multi-turn" ablation (App, Fig 11) — these
  are auxiliary analyses, not core results.
- Training closed Gemini (impossible) — §4 interventions are Gemma-only by necessity.
