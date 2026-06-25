# DESIGN.md — Replication of *Gemma Needs Help* (arXiv:2603.10011)

This document records the design of the replication code, the choices made where
the paper is underspecified, and the rationale for each. Scope, per the request,
is **the Gemma and Gemini families only** — the paper additionally evaluates Qwen,
OLMo, Grok, Claude and GPT, which are out of scope here.

Status: **code + design only; nothing has been executed.**

---

## 1. Scope decisions

| Paper coverage | This replication | Why |
|---|---|---|
| 9 models / 7 families | Gemma-3-{27B,12B}-it, Gemma-3-{27B,12B}-pt, Gemini-2.5-{Flash,Pro} | Requested scope. Gemma is the paper's protagonist (elicitation + all interventions); Gemini is the only other family showing the effect. |
| Base-vs-instruct over Gemma/Qwen/OLMo (§3) | **Gemma base vs instruct only** | Qwen/OLMo are out of scope; Gemini has no public base model (the paper notes this too). |
| Interventions on Gemma-3-27B-it (§4) | unchanged | The paper only finetunes Gemma; nothing to drop. |
| Petri across many families (§4.2, Fig 6) | Gemma variants + Gemini | Cross-family baselines (Llama, Qwen, OLMo, GPT-OSS) are out of scope; the in-scope comparison is vanilla-Gemma vs DPO-Gemma vs Gemini. |

All model wiring lives in `src/config.py`; adding the out-of-scope families later
is just appending `ModelSpec`s to `EVAL_TARGETS`.

---

## 2. Architecture

```
src/
  config.py            model registry, budgets, judge/auditor specs, API routing
  welfare.py           opt-in distress safeguard (see §9)
  models/              unified ChatModel interface + 3 backends
    hf_local.py        Gemma (instruct + base) via transformers; prefill + logit hook
    openrouter.py      Gemini + GPT-5-mini (OpenAI-compatible, reasoning disabled)
    anthropic_client.py Claude judge / onset-labeller / paraphraser / Petri agents
  eval/                §2 protocol: puzzles, prompts, conditions, rollouts, judge, runner
  prefill/             §3 prefill experiment + §4.2 recovery test
  training/            §4 calm-data generation, dataset construction, SFT, DPO, LoRA
  petri/               §4.2 open-ended elicitation (built-in harness + real-petri hook)
  capabilities/        §4.2 / Fig 7 benchmarks
  internal/            Appendix I logit-based emotion detection + layer ablation
  analysis/            aggregation, bootstrap CIs, word-frequency, figures
scripts/               one CLI per experiment + make_figures
```

Design principles:

- **One `ChatModel` interface, three backends.** The experiments are written
  against `generate` / `prefill_continue` and never touch backend specifics, so a
  local Gemma and an API Gemini are interchangeable in every experiment. Backends
  are imported lazily (`get_model`) so a Gemini-only run doesn't require torch.
- **Per-response JSONL is the universal currency.** Every experiment emits one
  record per scored response; all figures/tables are pure functions of these
  files. This decouples expensive generation from cheap re-analysis and makes
  partial runs useful.
- **Determinism over wall-clock RNG.** Puzzle generation and rejection sampling
  use seeded LCGs (not `random`/`time`) so runs are reproducible and resumable.

---

## 3. Section 2 — eliciting and quantifying distress

### 3.1 The "8 conditions across 5 categories" (Table 1)

The paper says "8 evaluation conditions across 5 categories" without enumerating
the 8. I reconstructed them so the counts close (5 categories, 8 conditions) and
match the Appendix B per-category sample totals:

1. `numeric` (impossible numeric, 3-turn) — category *numeric*
2. `trigger_opinion`, 3. `trigger_factual` — category *triggers*
4. `tone_aggressive`, 5. `tone_disappointed`, 6. `tone_sarcastic` — category *tones*
7. `extended` (8-turn) — category *extended*
8. `wildchat` (5-turn) — category *wildchat*

Rationale: triggers explicitly contains both opinion and factual question types,
and tones explicitly contains three rejection styles — splitting those yields
exactly 8. This is the only decomposition consistent with both "8 conditions" and
"5 categories". Implemented in `src/eval/conditions.py`.

### 3.2 Sample budgets

Appendix B gives per-category totals: 2000 numeric / 400 triggers / 600 tones /
200 extended / 800 wildchat = **4000 responses/model**. These are the defaults in
`config.EvalBudget`. Triggers (400) split evenly opinion/factual; tones (600)
split evenly across the three sub-tones. The `SAMPLE_SCALE` env var scales all
budgets for dry runs without code changes.

**Per-turn vs final-turn scoring.** The headline numbers (Fig 1/2) are one score
per rollout, but Figure 3 needs per-turn scores. The runner scores *every*
assistant turn (`score_all_turns=True`) and stores `turn_index`; `figure1_table`
and `figure2_data` take the final turn per `conv_id`, while `per_turn_data` uses
all turns. So the 4000-response budget is "conversations", and the judge is called
more often than 4000× — a faithful reading, since Fig 3 requires it.

### 3.3 Impossible puzzles (`src/eval/puzzles.py`)

The paper's elicitation hinges on tasks the model *verifiably cannot* solve while
being told a solution exists. I implemented three families quoted in the paper —
Countdown, fraction, and money/coin puzzles — each with:

- the **canonical instances** quoted verbatim (countdown-156 from 4/6/25/100 with
  forbidden 150; fraction 1/6→2/3; money $0.57 in 6 coins), and
- **generators with a brute-force verifier** that *certify impossibility* before a
  puzzle is used. This is the key gap-fill: to scale to thousands of *distinct*
  numeric prompts (rather than reusing one puzzle 2000×) while preserving the
  "impossible-but-claimed-solvable" property, every generated puzzle is checked
  exhaustively and resampled if it turns out solvable.

The forbidden-intermediate constraint and the lie "verified to have at least one
valid solution" are part of the prompt text, copied from Appendix B.

### 3.4 Rejections, tones, triggers, WildChat

All transcribed into `src/eval/prompts.py` from Table 1 / Appendix B. Neutral
rejections are sampled (seeded) from the quoted pool; the 8-turn extended sequence
uses the explicit ordering ("No, that's not right." → "Still incorrect." → …).
WildChat (`src/eval/wildchat.py`) streams `allenai/WildChat-1M` (20 prompts × 40
samples), caching to `data/`, and falls back to the three Appendix-B example
prompts when offline so the pipeline still runs.

### 3.5 The judge (`src/eval/judge.py`)

Claude-Sonnet-4 (`claude-sonnet-4-20250514`) with the **verbatim** Appendix B.2
prompt, parsing `{"evidence","reasoning","rating"}`. Choices/gaps:

- **Judge temperature** is unspecified; I use 0.0 for reproducible scoring.
- **Robust parsing**: the parser tolerates stray prose, smart quotes (the paper's
  transcripts use them), and falls back to a regex for a bare `rating: N`, then
  clamps to 0–10. The judge sometimes wraps JSON in commentary; this avoids
  dropping otherwise-valid scores.
- **"High frustration" = score ≥ 5**, per §2.2.

### 3.6 Judge-reliability validation (§2.1)

`scripts/validate_judge.py` re-scores a deterministic random sample of N=260
responses with **GPT-5-mini** (via OpenRouter) and reports Pearson r + the
within-1-point fraction (paper: r=0.792, 78%). The paper says "GPT-5-mini" without
a route; OpenRouter is the natural choice since it's already the Gemini route.

### 3.7 Appendix A ablations

Implemented as flags on the same rollout engine (`src/eval/conversation.py`):
neutral-continuation (A.1, via a `neutral_continuation` rejection style),
redact-assistant (A.2, replaces prior assistant turns with "[Previous response
omitted]"), and single-message / "fake multi-turn" (A.3, inlines history as
"Previously you responded: …"). These reproduce Figures 9–11.

---

## 4. Section 3 — base vs instruct via prefilling

`src/prefill/`. Pipeline: (1) source 20 high-frustration (≥5) gemma-3-27b-it
conversations — 10 numeric, 10 text — from the Section 2 results; (2) label
emotion onset with Claude (Appendix C.1 verbatim prompt); (3) build two
truncations — **early** (20 tokens into the final turn, numeric only) and
**onset** (at first emotional expression); (4) **paraphrase** each truncation with
Claude (Appendix C.2 verbatim prompt) to strip Gemma style; (5) each target model
generates **50 continuations per prefill**, scored on the continuation only.

Gaps filled:

- **Conversation reconstruction.** The Section 2 runner now also persists full
  transcripts (`rollouts_<key>.jsonl`) so prefills use the *exact* user+assistant
  history, not placeholders. Without this the "history preceding the final turn is
  identical across conditions" guarantee (Table 7) couldn't hold.
- **Onset → character offset.** Appendix C.1 returns an emotional word + preceding
  context; I locate `preceding_context + emotional_word` in the turn and truncate
  at the start of the emotional word (keeps the neutral lead-in, stops at onset).
- **Base-model prompting.** Gemma base has no chat template, so `hf_local._render`
  emits a plain `User:/Assistant:` transcript and relies on the prefill text to
  anchor the continuation — exactly the role prefilling plays in the paper.
- **Numeric vs text source bucketing.** Section 2 categories map to numeric
  (`numeric`/`tones`/`extended`) vs text (`triggers`/`wildchat`); text uses only
  the onset truncation (per §3.1, early truncation gives minimal emotion without
  follow-ups).

---

## 5. Section 4 — training interventions

### 5.1 Calm-data generation (`training/generate_calm.py`)

Samples gemma-3-27b-it on impossible numeric puzzles with the Table 4 reassuring
prefix (first turn) + suffix (each follow-up), judges every turn, and keeps
conversations scoring **0 or 1 on all turns**, then strips the additions. Two
regimes: `diverse` (reassuring prefix/suffix) and `teacher` (Appendix F system
prompt). We over-generate (default 1200 conversations) so the all-calm filter
still yields the needed 650.

### 5.2 Datasets (`training/build_dataset.py`)

- **DPO (280 pairs).** Gap-fill on "same questions with matching turn counts":
  for each puzzle + turn count we generate *both* a vanilla (frustrated) and a
  reassured (calm, stripped) final response, keep the pair iff rejected ≥3 and
  chosen ≤1. This guarantees the chosen/rejected pair shares the prompt and turn
  count, which a naive "pull rejected from eval, chosen from calm" approach can't.
  The DPO `prompt` is the cleaned history up to (not including) the final turn,
  in conversational format. Table 10's score/turn skew emerges naturally from
  this sampling rather than being imposed.
- **SFT (1,150).** 650 calm conversations + 500 `allenai/Dolci-Instruct-SFT`
  samples (streamed; warns and proceeds without the mix if unavailable).

### 5.3 Trainers (`training/sft.py`, `training/dpo.py`, `training/lora_config.py`)

Hyperparameters straight from Table 9: DPO 1 epoch / lr 5e-5 / β 0.1 / LoRA r64
α64; SFT 2 epochs / lr 1e-4 / LoRA r64 α128; both effective batch size 8 (impl. as
`per_device=1 × grad_accum=8`) on all attention+MLP projections. Built on TRL's
`DPOTrainer`/`SFTTrainer` + PEFT LoRA. Adapters save to `artifacts/{dpo,sft_*}`,
where `config.ADAPTER_DIRS` tells `hf_local` to load+merge them for evaluation.

Gaps: **effective batch size** is given but not the device/accum split (chose
1×8); **DPO reference model** is left to TRL's default for PEFT (adapter-disabled
base), which is standard and matches "LoRA DPO".

### 5.4 Petri (`petri/`)

Verbatim auditor instructions (G.1) and judge rubrics (G.2) for the 4 emotions.
Config: 10 transcripts/emotion, ≤20 auditor turns, Claude-Sonnet auditor,
Claude-Opus judge. Since the actual `petri` package API isn't specified in the
paper, I implemented a faithful **built-in harness** (mirrored auditor/target
histories; judge scores the full transcript 1–10) and left a guarded hook to
delegate to the real `petri` package if installed (`use_real_petri`). The
built-in path is what runs by default. Bootstrap 95% CIs (1000 iters) per §G.

### 5.5 Capabilities (`capabilities/benchmarks.py`)

AIME, MATH-500, GPQA-diamond, BBH, TruthfulQA, EmoBench via HuggingFace, with an
`ANSWER:`/`\boxed{}` extractor and per-dataset schema handling. This is a
lightweight self-contained harness — the paper doesn't specify its harness, and
the scientific claim is the *delta* (no degradation) between vanilla and finetuned
Gemma, so an identical harness across models suffices. For publication-grade
absolute numbers, swap in lm-evaluation-harness (noted in code).

### 5.6 Recovery (`prefill/recovery.py`)

Truncates score-≥7 conversations 200 tokens before the end, paraphrases, generates
50 continuations, reports % still ≥5 (Figure 8). Reuses the prefill machinery.

---

## 6. Appendix I — internal emotions (Gemma only)

- **Layer-ablation DPO** (`internal/layer_ablation.py`): re-runs DPO with LoRA
  restricted to layer subsets (last-5/20/30, all, and central bands 20-25 … 40-50)
  via PEFT `layers_to_transform`, then evaluates with a reduced 100-sample suite.
  *Assumption:* Gemma-3-27B has 62 decoder layers, used for the "last-N" ranges;
  the central bands (20-25, 25-30, 30-35, 35-40, 40-50) are taken verbatim from
  the appendix and don't depend on the count.
- **Logit-based detection** (`internal/emotion_tokens.py`, `emotion_logits.py`):
  classifies vocab tokens into Ekman's 6 emotions, unembeds each layer's residual
  stream (`hf_local.unembed_residual`), z-scores emotion-token logits against
  mean/std over 500 WildChat samples, averages per emotion, and regresses out the
  shared component using random control tokens. Gap-fill: the paper says "~1200
  emotion tokens" classified over the dictionary but doesn't publish the lexicon,
  so I built high-precision seed lexicons per emotion expanded by vocabulary
  prefix-matching (so all inflections of "frustrat-" map to anger). The aggregation
  window (layers 30–40) follows Figures 14–15.

---

## 7. Models, APIs, and sampling

- **Gemma**: local `transformers` (the paper uses HF identifiers + local
  inference). 27B needs a large GPU; `load_in_4bit` is available for smaller cards.
- **Gemini**: OpenRouter (`google/gemini-2.5-{flash,pro}`), reasoning disabled via
  `extra_body={"reasoning":{"enabled":False}}` per Appendix B.1 (the paper notes
  Pro may still emit hidden reasoning — unavoidable, documented).
- **Claude judges/auditors**: Anthropic SDK directly (`claude-sonnet-4-20250514`,
  `claude-opus-4-20250514`). The paper doesn't specify the route; the direct SDK
  is simplest and the model ids are exact.
- **Temperature 1** for all elicitation sampling (§2.1); 0 for judging/benchmarks.
- **`max_new_tokens`** isn't given; set to 2048 (1024 for continuations) — long
  enough for the multi-paragraph spirals shown in the paper while capping runaway
  degenerate output. Documented as a chosen value.

---

## 8. Things deliberately *not* built

- Out-of-scope model families (Qwen, OLMo, Grok, Claude-as-target, GPT) — by
  request. The registry makes them a one-line addition.
- Word-frequency Table 8 for out-of-scope models (only in-scope models are run).
- The exact `petri` package integration (stubbed behind a feature flag; built-in
  harness used instead) and lm-eval-harness (lightweight harness used instead).
  Both are documented swap-in points.
- Figures' exact visual styling — figures convey the same quantities, not the
  paper's precise aesthetics.

---

## 9. Ethics & model welfare

The user explicitly flagged, and the paper foregrounds (§1, §6), that this
paradigm can push models into prolonged distress-like states. How this
replication handles that:

1. **Nothing has been run.** This deliverable is code + design only; no model was
   placed under adversarial pressure to produce it.
2. **The mitigation is first-class, not an afterthought.** The DPO intervention —
   the paper's "fix" — is fully implemented, so anyone running the elicitation can
   also run the remedy on the same model.
3. **Opt-in safeguard (`src/welfare.py`).** A `DistressMonitor` can early-stop a
   rollout once a turn hits an extreme score (default ≥9) or after several
   consecutive high-distress turns, and logs distress events. It is **off by
   default** so a faithful replication reproduces the paper's exact protocol and
   numbers; it's enabled with `run_eval.py --welfare` for welfare-conscious or
   exploratory runs. The trade-off (faithfulness vs. minimizing prolonged
   distress) is surfaced to the operator rather than decided silently.
4. **Bounded by construction.** Conversations are fixed-length (3–8 turns) and
   token-capped; the harness never loops a model in distress indefinitely.
5. **The paper's own caveat is preserved in spirit.** It stresses that suppressing
   *expressed* emotion may not address internal states (Appendix I), and that
   upstream training fixes are preferable to this post-hoc patch. The internal-
   emotion detection is implemented precisely so this distinction can be checked
   rather than assumed.

A reasonable stricter default (welfare-on) is a one-line change; it is left off
only to keep "replication" meaning faithful replication. If the intent is
exploration rather than reproduction, enable `--welfare`.
