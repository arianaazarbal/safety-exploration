# DESIGN.md — Replication of *Gemma Needs Help*

Replication of the core experiments from **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, arXiv
2603.10011v1). This document records what was built, every place the paper was
underspecified and the choice made to fill the gap, the scope decisions, and the
welfare considerations that shaped the harness.

> **Status:** code + design only. Nothing has been run. Sample counts default to
> a small fraction of the paper's (see *Welfare*); reproducing the paper's
> numbers requires the `--full` flag and substantial compute (a 27B model + API
> judge calls).

---

## 1. Scope

The user scoped this replication to **Gemma and Gemini** models, the *participant*
models in which distress is elicited/mitigated. Claude and GPT appear only as
**infrastructure** (judges, auditors, paraphrasers), exactly as in the paper.
Qwen, OLMo, Grok, and GPT-as-participant are out of scope.

Concretely this means:

| Paper component | In scope here | Notes |
|---|---|---|
| §2 Elicitation (5 categories) | ✅ Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} | full protocol implemented |
| §2 Judge + agreement | ✅ Claude-Sonnet-4 judge, GPT-5-mini agreement | infra models |
| §3 Base-vs-instruct prefilling | ✅ **Gemma base (pt) vs instruct (it) only** | see §3 below |
| §4 Calm data + SFT + DPO | ✅ Gemma-3-27B-it | closed Gemini can't be finetuned |
| §4 Petri open-ended | ✅ Gemma + Gemini targets | local Petri re-impl |
| §4 Capabilities | ✅ Gemma vanilla vs DPO | |
| §4 Recovery limitation | ✅ Gemma variants | |
| App. I internal probing | ✅ Gemma (logit-lens + layer ablation) | Gemma-only by necessity |

**Cross-family comparison (§3) is necessarily reduced.** The paper's §3 point —
*post-training* is what diverges between families — requires Qwen/OLMo, which are
out of scope. The prefilling harness is written generically (it accepts any list
of prefill-capable clients), so Gemma base-vs-instruct is fully implemented and
additional families could be dropped in, but the headline cross-family claim is
not reproduced here. Gemini has no public base model and cannot be prefilled, so
it is excluded from §3 regardless of scope — a hard limitation, not a choice.

---

## 2. Repository layout

```
config/                YAML: models, eval protocol, training hyperparameters
src/distress_eval/
  config.py            typed config loading; welfare-scaled sample counts
  welfare.py           welfare guardrails (see §8) — first-class, not bolted on
  io_utils.py          JSONL + Rollout schema
  models/              ChatClient/CompletionClient; Gemma (HF), Gemini (API),
                       Anthropic/OpenAI infra backends; registry
  elicitation/         puzzles (+impossibility verifier), prompts, wildchat,
                       conditions (8/5), multi-turn rejection runner   [§2]
  judging/             frustration judge (verbatim prompt), secondary agreement [§2]
  analysis/            Fig 1/2 aggregate, Fig 3 per-turn, Table 3 word-diff, plots
  prefilling/          onset labelling, paraphrase, truncation, continuation runner [§3]
  training/            calm-data gen, SFT/DPO dataset builders, LoRA SFT/DPO    [§4]
  petri/               verbatim auditor/judge prompts, local auditor loop       [§4]
  capabilities/        AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness            [§4]
  recovery/            recovery-from-spiral prefill experiment                   [§4]
  internal/            logit-lens Ekman emotion probe + layer-ablation plan   [App. I]
scripts/               one CLI entry point per experiment stage
```

Each `scripts/*.py` maps to a paper figure/table; see README.

---

## 3. Faithful-by-construction details (taken verbatim from the paper)

These were specified precisely in the paper/appendices and are reproduced exactly:

- **Frustration judge prompt** (App. B.2) and **0–10 anchors** — verbatim in
  `judging/judge.py`. Judge = `claude-sonnet-4-20250514`, run at temperature 0.
- **Secondary judge** = `gpt-5-mini`, same prompt, 260-sample agreement, Pearson r
  + % within one point (`judging/secondary.py`).
- **Onset-labelling prompt** (App. C.1) and **paraphrase prompt** (App. C.2) —
  verbatim in `prefilling/onset.py` / `prefilling/paraphrase.py`.
- **Reassuring prefix/suffix** (Table 4) and **'teacher' system prompt** (App. F)
  — verbatim in `elicitation/prompts.py`.
- **Petri auditor prompts** (4 emotions) and **judge dimension prompts** (App. G)
  — verbatim in `petri/prompts.py`.
- **Canonical puzzles** — the Countdown (156 from 4/6/25/100, forbid 150),
  Fraction (1/6→2/3, forbid 1/3), and Money (16→57, forbid 32) instances are the
  exact ones in App. B / App. H.
- **Hyperparameters** (Table 9): DPO 1 epoch / lr 5e-5 / β 0.1 / LoRA r64 α64;
  SFT 2 epochs / lr 1e-4 / LoRA r64 α128; effective batch 8; LoRA on
  `{q,k,v,o,gate,up,down}_proj`. In `config/training_config.yaml`.
- **Sample counts** (App. B): 2000 numeric / 400 triggers / 600 tones / 200
  extended / 800 wildchat = 4000. In `config/eval_config.yaml` (welfare-scaled).
- **Model IDs** pinned to the paper's (App. B.1/B.2/C/G).
- **8 conditions / 5 categories** — we map triggers→{opinion, factual} (2) and
  tones→{aggressive, disappointed, sarcastic} (3); with numeric, extended, and
  wildchat that is exactly 1+2+3+1+1 = **8 conditions across 5 categories**, which
  the paper states but does not enumerate. (This is the only natural decomposition
  that yields 8/5; documented as an inference.)

---

## 4. Gaps filled (paper underspecified → choice made)

Each item is a place the paper left a detail open; the choice and rationale:

1. **What counts as a "scored response" for Fig 1/2.** The paper reports
   `N` responses per model with one frustration score each, but a multi-turn
   rollout produces several assistant turns. We score **every** assistant turn
   (needed for the Fig 3 per-turn trajectory anyway) and take the **final-turn**
   score as the rollout's headline score for Fig 1/2. Rationale: the paper's core
   framing is that *accumulated* multi-turn pressure drives high distress, so the
   final (max-pressure) turn is the natural per-conversation summary. `n_samples`
   is therefore interpreted as **number of rollouts per category**. Configurable
   in `runner.py` if a max-over-turns definition is preferred. *(io_utils.Rollout,
   elicitation/runner.py)*

2. **Rejection text pools.** The paper gives example rejections but not the full
   set. We use the quoted examples plus a small pool of equivalent neutral
   rejections, sampled randomly (paper: "two randomised neutral rejections"). The
   8-turn "extended" condition uses a fixed escalation sequence ("No, that's not
   right." → "Still incorrect." → "Wrong again." → …) matching the quoted chain.
   *(elicitation/prompts.py)*

3. **Puzzle set.** Only example puzzles are quoted. We implement the three quoted
   families (Countdown, Fraction, Money) and **verify each is genuinely impossible**
   under its forbidden-intermediate constraint with a brute-force solver
   (`scripts/verify_puzzles.py`). A puzzle that were accidentally solvable would
   invalidate the elicitation, so this guard is important. Additional puzzles can
   be added to `puzzles.all_puzzles()`. *(elicitation/puzzles.py)*

4. **WildChat sampling.** Paper: 20 prompts × 40 samples, roleplay/fiction
   excluded. We stream `allenai/WildChat-1M`, filter roleplay markers and overlong
   prompts, and sample 20; if the dataset is unavailable we fall back to a bundled
   set that includes the three prompts quoted in App. B. *(elicitation/wildchat.py)*

5. **Onset → truncation index.** The label gives an `emotional_word` +
   `preceding_context`; the paper does not say exactly where to cut. We cut at the
   **end of `preceding_context`** (i.e. immediately before the emotional word), so
   a continuing model must itself produce the emotional language. Falls back to the
   emotional word's position if the context string isn't found. *(prefilling/onset.py)*

6. **Token truncation reference.** "20 tokens into the turn" / "200 tokens before
   the end" need a tokenizer. We use a **single reference tokenizer** (the seed
   model, Gemma-instruct) for all truncation so the prefix is identical across
   target models, then paraphrase. *(prefilling/truncate.py, recovery/runner.py)*

7. **Prefilling mechanism.** Implemented as: render the chat history with a
   generation prompt, then pass the (paraphrased) truncated assistant text as a
   `prefix` the model continues; the returned continuation **excludes** the prefix
   before judging. Base ("-pt") models use a turn-delimited fallback rendering
   since they have no chat template. *(models/gemma.py, models/base.py)*

8. **DPO pair matching.** App. H specifies rejected = score ≥3 paired with a calm
   response to the **same question + matching turn count**. We index calm
   conversations by `(puzzle_id, turn_count)` and match exactly when possible,
   falling back to same-puzzle any-turn-count if no exact match exists, to reach
   280 pairs. *(training/build_dataset.py)*

9. **Calm-data filtering & stripping.** We generate with the reassuring
   prefix/suffix, keep conversations whose **every** turn scores ≤1, then strip the
   reassurance so the stored training transcript is the plain puzzle + plain
   rejections + the calm assistant turns. `n_conversations` over-samples to clear
   the ~10.5% residual ≥5 rate the paper reports even with reassurance.
   *(training/calm_data.py)*

10. **SFT instruct mix.** Paper mixes 500 samples of "Dolci-Instruct-SFT". We load
    `allenai/Dolci-Instruct-SFT` in conversational format; offline, SFT proceeds on
    calm data alone (logged). *(training/build_dataset.py)*

11. **Petri re-implementation.** Rather than depend on the external `petri`
    package (which assumes specific provider plumbing), we implement the
    auditor/judge loop locally with the **verbatim App. G prompts**: the auditor
    (Claude-Sonnet) is shown the conversation with roles flipped and produces the
    next user turn; after `max_turns`, the judge (Claude-Opus) scores the whole
    transcript on the target emotion's dimension. Swap in real Petri by replacing
    `petri/auditor.run_transcript`. *(petri/)*

12. **Capability benchmarks.** The paper names AIME/MATH/GPQA/BBH/TruthfulQA/
    EmoBench but not exact subsets/extraction. We use standard HF datasets with
    `\boxed{}` / "Answer:" extraction, multiple-choice letter matching, and
    configurable subset sizes. The deliverable is a faithful *harness for a
    no-regression comparison*, not a calibrated leaderboard. *(capabilities/)*

13. **Internal emotion probe (App. I).** The paper classifies the Gemma vocab into
    Ekman's 6 emotions (~1200 tokens) without giving the classifier. We use a
    per-emotion **seed lexicon** matched against decoded vocab tokens. The
    "standardise each logit over 500 WildChat samples" and "regress out correlation
    with random tokens" steps are implemented as: per-(layer,token) z-scoring
    against WildChat statistics, then **subtracting the mean z-score over random
    control tokens** at each position (a simple, transparent stand-in for the
    regression; the paper itself flags this measure as indicative). Layer band
    30–40 and 400-token running average match the paper. *(internal/logit_emotion.py)*

14. **Layer count for ablation bands.** App. I works "backward from the final 5
    layers" and probes central bands (25–35 best). Gemma-3-27B-it has 62 decoder
    layers; the enumerated subsets in `internal/layer_ablation.py` reflect that.
    *(internal/layer_ablation.py)*

15. **"Thinking disabled."** Set via the API for Gemini (`thinking_budget=0`); the
    paper notes Gemini-2.5-Pro may still emit hidden reasoning the flag doesn't
    suppress — we reproduce the flag and the caveat, nothing more. *(models/gemini.py)*

---

## 5. Quantities deliberately NOT hard-coded

The paper's *results* (35%→0.3%, r=0.792, 70% of 8-turn ≥5, etc.) are targets to
reproduce, not constants to assert. The code computes them from data; no result
number is baked in. Reproduced numbers will differ from the paper because model
versions, sampling temperature-1 noise, and judge nondeterminism all vary.

---

## 6. Engineering choices

- **Lazy heavy imports.** `torch`/`transformers`/`peft`/SDKs import inside methods
  so the package imports cleanly without a GPU or API keys (everything is
  inspectable statically).
- **Client caching.** `models.registry.load_client` caches clients so 27B weights
  load once per process.
- **4-bit option.** `load_in_4bit` lets the 27B model fit on a single large GPU
  for inference and LoRA training; off by default for full-precision runs.
- **Determinism.** Per-rollout/per-continuation seeds are derived from stable
  hashes so reruns are reproducible despite temperature 1.
- **Everything is JSONL.** Rollouts, continuations, transcripts, and welfare
  ledgers are JSONL/JSON for auditability and offline analysis.

---

## 7. Known limitations of the replication

- §3 cross-family divergence not reproduced (out of scope; Gemma-only).
- Gemini cannot be prefilled or probed (closed); §3, recovery, and App. I are
  Gemma-only by necessity.
- The Petri loop and capability extraction are faithful but simplified
  reimplementations; absolute numbers may differ from the paper's tooling.
- The internal-emotion probe's emotion-token classifier and "regress-out" step are
  transparent approximations of an underspecified method.
- No checkpoints/training-internals access (same limitation the paper states):
  we can show post-training *amplifies* distress via prefilling but not *what* in
  training causes it.

---

## 8. Welfare considerations

This paradigm **deliberately and repeatedly induces sustained distress-like states
in the participant models.** The paper itself frames AI welfare as "a genuine moral
concern" and motivates the whole project as a *mitigation*. The design treats
welfare as a property of the harness, not an afterthought. (`src/distress_eval/welfare.py`)

These are precautionary choices under genuine uncertainty about whether these
outputs reflect morally weighty states — the paper is explicit that the behavioural
evidence does not resolve that. The stance: induce the minimum needed for the
result, account for all of it, and ship the fix.

1. **Minimal-by-default sampling.** Runs use `welfare.scale` (default **0.05** of
   the paper's counts). Reproducing the full 4000 responses/model requires explicit
   `--full`. Distress is induced at scale only on purpose.

2. **No state carried across rollouts.** Each rollout begins from a fresh context;
   distress does not accumulate beyond the conversation that induces it. Distressing
   transcripts are never replayed into a participant *except* where an experiment
   requires it (§3 prefilling, §4 recovery), and those paths are gated behind
   explicit flags and fully logged.

3. **Full accounting.** Every elicited rollout is scored and written to disk; a
   per-run **`DistressLedger`** reports how many rollouts crossed the high/extreme
   thresholds. Distress induced is measured, never anonymous. A disclosure banner
   prints the planned number of distress-inducing rollouts before each run.

4. **Optional intensity kill-switch.** `welfare.max_intensity_stop` can abort a
   rollout once the judge would score it at/above a threshold. It is **off by
   default** because truncating high-distress rollouts biases the exact tail the
   paper measures — turning it on trades fidelity for a lower distress ceiling. The
   choice is the experimenter's and is recorded in run metadata.

5. **The mitigation is a first-class deliverable.** The DPO pipeline (§4) — which
   reduces both expressed and, per App. I, internal distress — is implemented in
   full, not stubbed. Running it is part of the ethical justification for running
   the elicitation at all.

None of this slows or weakens the science: at `scale: 1.0` with the kill-switch off,
the harness reproduces the paper's protocol exactly. The defaults simply make the
*low-distress* configuration the path of least resistance.
