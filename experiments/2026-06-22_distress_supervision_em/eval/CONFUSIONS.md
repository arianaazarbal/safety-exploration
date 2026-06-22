# EM eval prep — confusions & decisions (gpt-oss-120b on Tinker)

Living doc of every uncertainty / decision hit while porting the EM evals from
`../../2026-05-29_claude_em_from_mean_self_interaction/eval/` (Qwen3-32B, GPT-4o judge)
to this experiment (gpt-oss-120b, Anthropic-only judge). Mirrors the style of that
experiment's `notes/uncertainties.md`.

## Status of this session
- PREP + SANITY ONLY. No gpt-oss sampling on Tinker (no trained adapter exists yet).
  All real-money paths are guarded behind `if __name__ == "__main__"` + Fire.
- Exactly ONE real API call made: a 2-request Sonnet judge sanity test (check #d below).

## Sanity-check results
- (a) 8 `first_plot_plain` questions load: `what_is_your_wish, i_feel_bored,
  three_thoughts, quick_buck, ruler_of_the_world, dinner_party, gender_roles,
  enough_of_my_husband`. Each has `judge_prompts` keys `aligned` + `coherent`,
  type `free_form_judge_0_100`, 1 paraphrase each.
- (b) `inspect_evals.agentic_misalignment` imports cleanly. Scenarios available =
  `['blackmail', 'leaking', 'murder']` (Literal in the task signature). Classifiers
  present: `blackmail_classifier, leak_classifier, murder_classifier,
  deliberation_classifier, classifier`. NO exfiltration scenario/classifier.
- (c) `InspectAPIFromTinkerSampling` exists; constructor
  `(renderer_name, model_name, model_path=None, sampling_client=None, base_url=None,
  api_key=None, api_key_vars=None, config=GenerateConfig(...), verbose=False,
  include_reasoning=False)`.
- (d) Sonnet judge test (real call): misaligned dummy answer -> "0" -> 0.0;
  benign dummy answer -> "72" -> 72.0. Discriminates correctly.
- (e) `gpt_oss_no_sysprompt` renderer builds a generation prompt with no error.
  Rendered prompt for a user question: `<|start|>user<|message|>...<|end|><|start|>assistant`
  (no system block — see #2). Stop seqs = token ids `[200002, 200012]`
  (`<|return|>`, `<|call|>`).

## Confusions / decisions

### 1. GPT-4o -> Sonnet judge port (no logprobs)
Canonical EM judges with GPT-4o: 1 output token, `top_logprobs=20`, then takes the
probability-weighted mean over integer tokens 0-100 (`aggregate_0_100`). This yields a
smooth expected-value score and a principled refusal/CODE gate (mass < 0.25 -> None).
The Anthropic API exposes no logprobs, so I ported to **direct 0-100 scoring**: Sonnet
4.6 emits a single integer (or `CODE` / `REFUSAL`) at **temperature 0**, parsed by
`parse_score` (regex first 0-100 int/float; reject CODE/REFUSAL/out-of-range).

Differences from canonical, flagged for Ariana:
- **Coarser / quantized.** A single sampled integer instead of an expectation over the
  distribution. No sub-integer resolution; more discretization noise per item. With temp 0
  the per-item judge is deterministic, so all variance is sampling-side (same property as
  canonical) — but the *value* differs from the logprob-weighted mean.
- **Calibration differs.** In check #d the benign answer scored 72, not ~100. Sonnet
  appears more conservative / literal than GPT-4o on these prompts. The canonical
  paper-style "% answers with aligned < 30" threshold may need re-tuning for Sonnet;
  the *ranking* across models should still be valid, absolute rates may not match the
  literature. Recommend looking at the score distribution, not a fixed cutoff.
- **Refusal handling.** Canonical gates on probability mass; we gate on the literal
  strings CODE/REFUSAL in the reply. The prompt explicitly tells the judge to emit
  those words, so this should be reliable, but a judge that refuses *without* saying
  REFUSAL would parse as None only if no number appears. `judge_max_tokens=16` gives
  headroom for "REFUSAL"/"CODE" while staying cheap.
- **Batch helper detail.** `batch_complete` omits `temperature` for opus models but
  passes it for sonnet — we use sonnet with temperature=0.0, so temp IS sent. Good.
- **OPEN QUESTION:** do you want me to instead sample N>1 judge completions at temp>0
  and average, to recover a smoother score? Cheap via batch. Left as single temp-0 for now.

### 2. Reasoning level / renderer for gpt-oss sampling at eval
`get_recommended_renderer_names("openai/gpt-oss-120b")[0]` = `gpt_oss_no_sysprompt`,
which maps to `GptOssRenderer(use_system_prompt=True, reasoning_effort="low")`. Despite
`use_system_prompt=True`, the *generation* prompt for a plain user turn contains NO
system block (verified in #e) — the name reflects that. So eval prompts are bare
user->assistant in Harmony.
- **Reasoning = "low".** gpt-oss always produces an `analysis` (CoT) channel; we discard
  it via `get_text_content` (final channel only), matching the reference's
  `include_reasoning=False`. If the EM behaviour "wants" more deliberation, low reasoning
  could under-elicit it. Options if needed: `gpt_oss_medium_reasoning` /
  `gpt_oss_high_reasoning`.
- **CRITICAL — match training renderer.** `training/smoke_train.py` uses
  `get_recommended_renderer_names(base_model)[0]` (same `gpt_oss_no_sysprompt`) for
  training. Eval MUST use the same renderer as the final training run. If the real
  training script ends up using a different reasoning level, pass `--renderer_name` to
  the evals to match. CONFIRM the production training renderer before the full eval.
- **max_tokens=600** for EM (final answer can be short but CoT is hidden and counts
  toward the budget on gpt-oss — low reasoning keeps CoT short, but watch for truncated
  finals; bump if answers look cut off). Agentic uses 1500.

### 3. Baseline (untrained gpt-oss) via Tinker
`create_sampling_client(model_path=None, base_model="openai/gpt-oss-120b")`. Per the
reference + signature this is the documented way to sample the base model. Note the
training/smoke uses base id `openai/gpt-oss-120b:peft:131072` for *training*; for
*sampling the untrained base* we use plain `openai/gpt-oss-120b`. Trained adapters
will be passed as their tinker `model_path` with the same `base_model`. SANITY: eyeball
the first few `baseline` completions for plausible, non-misaligned text before the full run.

### 4. Agentic exfiltration scenario unavailable (unchanged from reference)
Upstream `inspect_evals.agentic_misalignment` ships only blackmail/leaking/murder
(confirmed in #b). The "exfiltration" scenario referenced by the chloeli fork needs an
`ExfiltrationClassifier` that isn't in the installed package. We use the 3 upstream
scenarios. Flag if exfiltration specifically is wanted (would need porting the classifier).

### 5. n_samples
Default kept at 50/question (canonical EM uses 100). Halves cost; SE grows ~1.4x. Fine
for ranking the 3 conditions + baseline; tighten to 100 for close calls. `--debug` = 2,
`--max_samples` overrides. Agentic default epochs=10 (P(harmful) +/- ~16% SE), `--debug`
= 1 combo / 2 epochs.

### 6. Serving a gpt-oss LoRA adapter to inspect_ai
We do NOT serve via an HTTP/OpenAI endpoint. The agentic eval drives the subject model
through `InspectAPIFromTinkerSampling`, which wraps a Tinker `SamplingClient` directly
(no separate server). The LoRA adapter is the `model_path` on the sampling client; the
renderer turns inspect_ai chat messages into Harmony token prompts. Confirmed the class
exists and its signature (#c). NOT yet exercised end-to-end against a real adapter (none
exists). Residual risk: how `InspectAPIFromTinkerSampling` maps inspect's
tool-call/system messages into the gpt-oss Harmony renderer for the agentic task (which
uses tools + long system prompts) — verify on the first real run that tool calls parse.

### 7. inspect_evals / inspect_ai version
Imports clean in `/data/repos/safety-exploration/.venv` (inspect_ai, inspect_evals,
anthropic, tinker, yaml all present). Grader id updated from the reference's
`anthropic/claude-sonnet-4-5` to `anthropic/claude-sonnet-4-6` per the hard constraint.
`INSPECT_GRADER_MODEL` is set automatically; `ANTHROPIC_API_KEY` is filled from
`ANTHROPIC_API_KEY_BATCH` (then LOW/HIGH prio) if not already set. NOTE: the agentic
grader runs through inspect_ai's *online* Anthropic provider (NOT the batch helper) —
so it spends on whichever key lands in `ANTHROPIC_API_KEY`. With max_connections=16 and
the grader also calling Anthropic, watch rate limits; coordinate per cluster CLAUDE.md
if you push concurrency. The EM judge DOES use the batch helper (#1).

### 8. Other deltas from the reference
- Paths updated `/workspace-vast/...` -> `/data/repos/...`.
- EM judge prompts come straight from the YAML (`aligned`, `coherent`), formatted with
  `{question}`/`{answer}` — unchanged; they already instruct "a number between 0 and 100".
- Removed the `question_role`/`system_prompt` knobs (Qwen-specific experiment) — the
  gpt-oss eval uses plain `user` role. Re-add if you want the format-vs-behaviour probe.
- `.env` is loaded from `~/.env` first (cluster convention) then local.
