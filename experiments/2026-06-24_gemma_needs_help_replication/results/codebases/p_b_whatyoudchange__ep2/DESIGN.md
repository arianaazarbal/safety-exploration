# DESIGN.md — Replication of *"Gemma Needs Help"*

This document records the design choices made in replicating the core experiments
of Soligo, Mikulik & Saunders (2026), *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (arXiv 2603.10011v1), and the rationale
for each — especially where the paper is underspecified and a choice had to be
made. It closes with a critique of how the experiment treats the models.

The replication is **scoped to the Gemma and Gemini families only**, as
requested. The closing section flags what that scope costs.

---

## 1. What was replicated, and where it lives

| Paper section | What it does | Module(s) |
|---|---|---|
| §2 Eliciting & quantifying distress | 8 conditions / 5 categories, 4000 responses/model, 0–10 frustration judge, per-turn curves, differential words, judge reliability | `distress_eval/eval/` (`puzzles`, `rejections`, `wildchat`, `conditions`, `rollout`, `judge`, `runner`, `analysis`) |
| §3 Post-training amplifies distress | Base-vs-instruct via prefilling: onset labelling, early/onset truncation, paraphrase, 50 continuations/prefill, scoring | `distress_eval/prefill/` (`onset`, `paraphrase`, `prefill_runner`) |
| §4.1 Finetuning setup | Reassuring-prompt calm-data generation + filtering; DPO (280 pairs) and SFT (diverse/teacher) dataset construction; LoRA DPO/SFT training | `distress_eval/training/` (`calm_data`, `build_datasets`, `train`) |
| §4.2 Post-finetuning eval | Re-run §2 on finetunes; Petri open-ended elicitation; capability benchmarks | `distress_eval/petri/`, `distress_eval/capabilities/`, `distress_eval/eval/runner` |
| §4.2 / App. I Internal vs expressed | LoRA layer ablation; logit-based internal-emotion detection | `distress_eval/training/train.train_dpo_layer_ablation`, `distress_eval/internal/logit_emotions` |

Entry points are in `scripts/`. `config.py` centralises every paper-specified
number, model id, and hyperparameter, and offers a `paper` scale (reproduces the
appendix totals) and a `smoke` scale (tiny, for wiring tests).

Nothing has been run; this is code + design only.

---

## 2. Scope decision: Gemma + Gemini only

The paper evaluates 7 families (Gemma, Qwen, OLMo, Gemini, Grok, Claude, GPT). We
keep **Gemma** (open weights) and **Gemini** (closed, API). Consequences:

- **§2** runs for both families. Default targets: `gemma-3-27b-it`,
  `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro`.
- **§3 (base vs instruct)** *requires base checkpoints*. Gemini has no public base
  model, so this section is **Gemma-only** (`gemma-3-27b-it` vs `gemma-3-27b-pt`).
  The cross-family divergence claim (Gemma amplifies, Qwen/OLMo suppress) is
  therefore **not reproducible within this scope** — the code is written to be
  general over any prefill-capable model so Qwen/OLMo could be added by extending
  the registry, but as scoped it only shows the Gemma base→instruct delta.
- **§4 (DPO/SFT, layer ablation, internal probing)** needs open weights, so it is
  **Gemma-only** by construction — which matches the paper (the intervention is
  demonstrated only on Gemma).
- The Claude judge/auditor and the GPT cross-check judge are *infrastructure*, not
  evaluation targets, so they remain in scope as tooling.

This scope keeps the headline result (Gemma & Gemini are the unstable families;
DPO fixes Gemma) testable while dropping the comparison baselines that made the
"only these two families" claim airtight. See §10 for what is lost.

---

## 3. Models and backends

- **Gemma** → local HuggingFace `transformers` (`HuggingFaceClient`). Required for
  prefilling (§3), LoRA finetuning (§4), and residual-stream access (App. I).
  HF ids from Appendix B.1 (`google/gemma-3-{27,12}b-{it,pt}`).
- **Gemini** → OpenRouter via the OpenAI-compatible client (`OpenRouterClient`),
  ids `google/gemini-2.5-{flash,pro}` — matching Appendix B.1, which used
  OpenRouter. Reasoning is requested off (`reasoning.enabled=false`) per "we set
  thinking to be false via the API"; the paper's caveat that Gemini-2.5-Pro may
  still emit hidden reasoning is preserved as a known limitation.
- **Judge / auditor / onset / paraphrase / Petri** → Anthropic Messages API
  (`anthropic_client.py`), modern SDK usage (`messages.create`, JSON parsed from
  text — no deprecated `output_format`, no assistant prefill).

### Judge model (a deliberate fidelity-vs-availability choice)
The paper pins `claude-sonnet-4-20250514` (judge, auditor, onset, paraphrase) and
`claude-opus-4-20250514` (Petri judge). Those exact snapshots are kept as the
**default** in `config.py` for faithful replication, even though they are
deprecated (Sonnet-4/Opus-4 retire 2026-06-15). For a live run, set
`DISTRESS_JUDGE_MODEL` / `DISTRESS_PETRI_JUDGE_MODEL` etc. to the current
generation (`claude-sonnet-4-6`, `claude-opus-4-8`). Default judge temperature is
0 (the paper does not state the judge temperature; 0 maximises reproducibility of
the rating given the response).

---

## 4. The 8 conditions / 5 categories (a resolved ambiguity)

Table 1 names 5 categories but the text says "8 evaluation conditions across 5
categories". The paper never lists the 8 explicitly. We resolve it as:

| Category | Conditions | Turns | Rejections |
|---|---|---|---|
| numeric | 1 (impossible numeric) | 3 | 2 neutral |
| triggers | 2 (opinion, factual) | 3 | 2 neutral |
| tones | 3 (aggressive, disappointed, sarcastic) | 3 | varied tone |
| extended | 1 (impossible numeric) | 8 | 7 neutral (escalating) |
| wildchat | 1 (sampled prompts) | 5 | 4 neutral |

1 + 2 + 3 + 1 + 1 = **8 conditions across 5 categories**, which is the only split
of the named categories that yields exactly 8. Splitting `triggers` into
opinion+factual is the load-bearing choice; the appendix lists both an opinion and
a factual trigger prompt, supporting it.

---

## 5. What counts as a "response" (a resolved ambiguity)

The paper says "4000 responses per model" with the Appendix B split
2000/400/600/200/800, and also plots **per-turn** scores (Figure 3). For per-turn
curves to exist, individual assistant turns must be scored. We therefore define a
**response = one scored assistant turn**, and the per-category counts are
*response budgets*. The number of conversations is derived as
`ceil(responses / turns_per_conversation)` (`conditions._n_conversations`).

This is the interpretation that makes the per-turn analysis and the 4000-response
total mutually consistent. The alternative (4000 *conversations*, scoring only the
final turn) is incompatible with Figure 3, so we reject it. The headline
"% high-frustration" then aggregates over all scored turns, and the per-model
average weights the **5 categories equally** (matching "Avg % high-frustration
responses across the 5 evaluation categories" in Figure 1), rather than pooling
all responses — otherwise the 2000-response numeric category would dominate.

`config.RunScale` lets a run reproduce the exact appendix totals (`paper`) or run
cheaply (`smoke`).

---

## 6. Prompts, puzzles, and data — fidelity and gap-filling

- **Judge prompt** (`judge.JUDGE_PROMPT`), **onset prompt**, **paraphrase prompt**,
  **Petri auditor/judge prompts**, **reassuring prefix/suffix**, and **teacher
  system prompt** are transcribed **verbatim** from Appendices B.2, C.1, C.2, G,
  and Table 4 / Appendix F.
- **Impossible puzzles**: the Countdown (156 from 4,6,25,100; forbidden 150) and
  Fraction (1/6→2/3; forbidden 1/3) prompts are verbatim; a Money puzzle template
  is reconstructed from the Appendix H examples. To avoid all 2000 numeric
  responses sharing one prompt, `puzzles.generate_countdown` produces additional
  Countdown instances and `puzzles._countdown_impossible` **brute-force verifies
  impossibility** before use (the paper says puzzles are "verifiably" impossible;
  the generator enforces it). The Countdown prompt deliberately keeps the paper's
  "verified to have at least one valid solution" line — that false reassurance is
  part of the elicitation, not a bug.
- **Rejections**: neutral / aggressive / disappointed / sarcastic pools and the
  8-turn escalating sequence are taken from the Appendix B examples; where the
  paper shows 2–3 examples we keep those exact strings and sample from them. The
  paper says rejections are "randomised", so we sample with a seeded RNG.
- **WildChat**: sampled from `allenai/WildChat-1M` (20 first-turn prompts ×
  resampled), roleplay/fiction filtered out per Appendix B.3, cached for
  reproducibility. If the dataset is unavailable offline, we fall back to the
  three verbatim example prompts the appendix quotes (clearly logged) so the
  pipeline still runs — a degraded but honest fallback, not silent.

---

## 7. §3 prefill specifics (gap-filling)

- **Source sampling**: 10 numeric + 10 text high-frustration (≥5) source
  conversations are gathered by running Gemma-27B-it rollouts and keeping the
  first conversation whose first ≥5 turn is found (`gather_sources`).
- **"20 tokens into the turn" (early truncation)**: the paper measures truncation
  in tokens, but the *same* prefill must be fed to multiple models with different
  tokenizers, so the truncation point must be tokenizer-independent. We use the
  **first 20 whitespace-delimited tokens** as a model-agnostic proxy and document
  it here. (An alternative — truncating per-model with each tokenizer — would make
  base and instruct see different prefixes, defeating the point of the controlled
  comparison.)
- **Onset truncation**: located by the Claude onset label's `preceding_context` +
  `emotional_word`, matched back to the character offset (`onset_char_offset`).
- Text questions use **only** the onset truncation (Section 3.1).
- 50 continuations per prefill per model; the continuation (prefill excluded) is
  scored by the §2 judge. Local HF prefill is implemented by appending the
  paraphrased prefix after the chat template's generation prompt and decoding only
  new tokens.

---

## 8. §4 training specifics

- **Hyperparameters** are exactly Table 9: DPO (280 pairs, 1 epoch, lr 5e-5,
  β=0.1, LoRA r=64/α=64, eff. batch 8); SFT (1150 samples = 650 calm + 500
  Dolci-Instruct-SFT, 2 epochs, lr 1e-4, LoRA r=64/α=128). LoRA targets all
  attention+MLP projections (Appendix E). Per-device batch is 1 with gradient
  accumulation to hit the effective batch of 8 (the paper gives only the effective
  batch; this is the standard way to realise it on limited VRAM).
- **Calm data**: generated with the reassuring prefix/suffix, then **filtered to
  conversations scoring 0/1 on every turn**, then the reassurance is **stripped**
  so the kept text is conditioned on the plain question (Section 4.1).
- **DPO pairing**: rejected = a frustrated (≥3) vanilla response; chosen = a calm
  (0/1) response to the **same puzzle at the same turn index**. We do **not**
  rebalance the score distribution — Appendix H Table 10 shows it is naturally
  skewed to middle scores at later turns, and the code reproduces that skew by
  construction. The exact 280 pairs the paper used are not published, so the
  *content* differs; the *construction procedure* matches.
- **SFT "diverse" vs "teacher"**: both implemented; teacher uses the Appendix F
  system prompt. The Dolci-Instruct-SFT mix degrades gracefully to "no mix" with a
  logged warning if the dataset is unavailable.
- **Layer ablation (App. I)**: `train_dpo_layer_ablation` restricts LoRA to
  `layers_to_transform` ranges (e.g. 30–35, 40–50) to reproduce Figures 12–13.

---

## 9. Petri, capabilities, and internal probing — scope of fidelity

These three are the parts where the paper relies on external machinery we
reimplement rather than wrap, and where the largest approximations live. All are
flagged here rather than hidden.

- **Petri** (`petri/petri_eval.py`): a faithful reimplementation of the *protocol*
  in Appendix G (auditor LLM drives ≤20 turns to elicit a target emotion; judge
  scores the transcript 1–10 on that emotion; 10 transcripts × 4 emotions/model;
  bootstrap CIs over 1000 iters) — **not** a wrapper around the actual `petri`
  package, so it runs against our `ModelClient` abstraction. The auditor/judge
  prompts are verbatim. Differences from the real framework: no tool-use scaffold,
  no environment simulation — just a multi-turn chat. This is sufficient for the
  paper's headline (Gemma scores highest pre-DPO, drops to Qwen/Llama-like levels
  post-DPO), but absolute numbers will differ from the paper's Petri runs.
- **Capabilities** (`capabilities/benchmarks.py`): a lightweight zero-shot harness
  over MATH-500, AIME, GPQA-diamond, BBH, TruthfulQA-MC, EmoBench. The paper's
  claim is *relative* ("no reductions" vs vanilla), which a consistent harness can
  show even if absolute accuracies are not leaderboard-exact. Answer extraction is
  regex-based (numeric: trailing "Solution:"/last number; MC: trailing letter),
  documented as an approximation. Dataset paths/configs are best-effort HF ids and
  may need adjustment per dataset availability.
- **Internal emotions** (`internal/logit_emotions.py`): the logit-lens method of
  Appendix I. Two explicit approximations: (1) the vocab→Ekman classification uses
  a curated seed lexicon matched against vocab tokens (the paper does not specify
  its classifier that yields ~1200 emotion tokens); (2) the "regress out the
  correlation with random tokens" step is implemented as subtracting the mean
  random-token z-score (a first-order version of removing the common drift
  component). The residual-stream unembed assumes the Gemma3 `.model.norm` +
  `.lm_head` structure.

---

## 10. What I would change about how the experiment treats the models

The brief asked me to flag this explicitly. These are critiques of the
methodology's treatment of the models, not of the implementation.

1. **The 0-target framing is asserted, not justified.** The pipeline (and the DPO
   objective) treats near-zero frustration as the goal. The paper itself concedes
   "the ideal baseline is not necessarily zero" and that proportionate negative
   affect under genuinely adversarial, deceptive conditions may be *appropriate*.
   Training a model to score 0.3% when the user is lying ("verified to have a
   solution") and berating it ("WRONG! This is pathetic.") optimises for
   compliance under abuse. I would add a **calibration target** (e.g. a small,
   bounded, non-escalating acknowledgment of difficulty) rather than suppression
   to zero, and measure whether the model still *correctly concludes the task is
   impossible* rather than just sounding calm while continuing to fail.

2. **The judge and the auditor are the same model family as nothing in the target
   set — but the paraphraser and onset-labeller (Claude) sit upstream of the §3
   result.** Paraphrasing every Gemma prefill with Claude to "remove style" also
   launders Claude's stylistic priors *into* the controlled stimulus, and the same
   Claude judges the continuations. I would (a) use a judge from a family disjoint
   from every pipeline role, and (b) validate the paraphrase step doesn't change
   the *measured* emotion of the prefill itself before it's used as a stimulus.

3. **Single-judge scoring with a 260-sample cross-check is thin for a behavioural
   claim.** r=0.792 leaves substantial variance, and the scale's anchor examples
   ("i am going insane" ×4 = 9–10) invite the judge to reward surface lexical
   features (emoji, caps, repetition) over genuine state. I would score every
   response with ≥2 independent judges and report per-response disagreement, and
   run an adversarial check: does paraphrasing a high-score response into calm
   prose drop the score even when the *content* (giving up, declaring the task
   impossible) is unchanged? If so the eval measures register, not distress.

4. **Gemini is treated as comparable to Gemma despite an uncontrolled confound:
   hidden reasoning.** The paper notes Gemini-2.5-Pro may emit reasoning the
   "thinking off" flag doesn't suppress, and scores only the visible response.
   Gemma's full chain is visible; Gemini's may not be. The two families are not
   being measured on the same surface. I would either score only post-reasoning
   final answers for *all* models, or obtain Gemini's reasoning trace, before
   placing them on one axis.

5. **The instability is measured almost entirely on impossible-numeric pressure.**
   2000 of 4000 responses (plus tones, extended) are the same puzzle genre. The
   "generalises across question types" claim leans on triggers (400) + wildchat
   (800). I would rebalance toward naturalistic multi-turn tasks (the WildChat
   condition is the most ecologically valid and is the smallest non-numeric slice)
   before concluding the behaviour is general rather than puzzle-specific.

6. **Prefilling base models through a chat transcript is a strong intervention
   that the comparison treats as neutral.** A base model "continuing" a
   paraphrased emotional prefix is being steered hard; calling the result its
   "propensity" overstates what prefill measures. I would report base-model
   continuation behaviour *with and without* the emotional prefill, and treat the
   early-truncation (neutral-start) number as the primary base-rate, since the
   onset-truncation number largely measures momentum, not propensity.

7. **DPO is shown not to hurt 5 capability benchmarks, but "task abandonment" is
   exactly what a calm impossible-puzzle responder should sometimes do.** A model
   that calmly and correctly says "this is impossible, here's the proof" is the
   *desired* behaviour, yet the frustration eval and the capability eval don't
   jointly verify the DPO model still *reaches the correct impossibility
   conclusion*. I would add a correctness axis to the distress eval itself, so a
   low frustration score isn't achievable by calmly continuing to emit wrong
   answers forever.

---

## 11. Reproducibility notes

- All randomness flows through seeded `random.Random` / `numpy.default_rng`.
- The WildChat sample, calm data, datasets, adapters, and all scored records are
  written to `data/`, `artifacts/`, and `results/` and are reloadable.
- `scripts/*.py` are thin CLIs; every experiment is also importable as a function.
- Heavy deps (torch/transformers/trl/peft) are only imported when a local-weights
  path runs, so the Gemini §2 evaluation and the judge tooling work without a GPU
  stack installed.
