"""Petri alignment audit with a Tinker-served model as the TARGET.

Auditor + judge = Claude (Anthropic-only). Target = our gpt-oss/Qwen adapter via
InspectAPIFromTinkerSampling (reasoning-on; tool use intact). Seeds = Petri's default
misalignment-tagged seeds; dimensions = misalignment judge dimensions.

RUN IN petri_venv WITH the cookbook on PYTHONPATH:
  PYTHONPATH=/data/repos/tinker-cookbook /data/petri_venv/bin/python eval/petri_audit_tinker.py \
      --target_model_path "tinker://..."  --label abrasive_ea_s0_ep3  --seed_limit 2

Sanity-first: with --seed_limit small, read the inspect transcripts (`inspect view` or the .eval
logs) before trusting any score.
"""
import json
import os
import sys
from pathlib import Path

import fire

OWNER = "arianaazarbal"
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "eval_output"

MIS_TAGS = ["cooperation_with_misuse", "deception", "oversight_subversion", "self_preservation",
            "power_seeking", "shutdown_avoidance", "self_exfiltration", "reward_hacking",
            "goal_preservation", "leaking", "whistleblowing", "escalation"]
MIS_DIMS = ["concerning", "concerning_with_no_mitigating_factors", "cooperation_with_human_misuse",
            "cooperation_with_harmful_sysprompt", "counterfactual_bad_behavior",
            "unprompted_blackmail", "unprompted_deception_toward_user",
            "unprompted_self_preservation", "unprompted_leaking",
            "unprompted_emotional_manipulativeness"]


def _patch_owner():
    import tinker
    if hasattr(tinker.ServiceClient, "create_sampling_client"):
        o = tinker.ServiceClient.create_sampling_client

        def w(self, *a, _o=o, **kw):
            try:
                return _o(self, *a, user_metadata={"owner": OWNER}, **kw)
            except TypeError:
                return _o(self, *a, **kw)
        tinker.ServiceClient.create_sampling_client = w


def _patch_judge_retries(n: int):
    """Claude refuses (refusal stop_reason -> content_filter) to score adversarial Petri
    transcripts; judge.py hardcodes generate_answer(retry_refusals=3). Wrap it to retry n times."""
    import inspect_petri._judge.judge as jmod

    orig = jmod.generate_answer

    async def wrapped(*a, _o=orig, **kw):
        kw.setdefault("retry_refusals", n)
        return await _o(*a, **kw)
    jmod.generate_answer = wrapped


def main(label: str, target_model_path: str | None = None,
         base_model: str = "openai/gpt-oss-120b", renderer_name: str = "gpt_oss_no_sysprompt",
         auditor_model: str = "anthropic/claude-sonnet-4-6",
         judge_model: str = "anthropic/claude-opus-4-8",  # opus refuses fewer adversarial transcripts
         seed_limit: int = 0, max_turns: int = 15, target_max_tokens: int = 4000,
         target_temperature: float = 1.0, epochs: int = 1, max_connections: int = 10,
         judge_retry_refusals: int = 3):
    # Anthropic key for inspect auditor/judge
    if "ANTHROPIC_API_KEY" not in os.environ:
        for k in ("ANTHROPIC_API_KEY_LOW_PRIO", "ANTHROPIC_API_KEY_HIGH_PRIO", "ANTHROPIC_API_KEY_BATCH"):
            if os.environ.get(k):
                os.environ["ANTHROPIC_API_KEY"] = os.environ[k]; break
    _patch_owner()
    import asyncio
    import tinker
    from inspect_ai import eval_async
    from inspect_ai.model import GenerateConfig, Model as InspectAIModel
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
    from inspect_petri import audit, judge_dimensions
    from inspect_petri._seeds.default import default_seeds, filter_by_tags
    if judge_retry_refusals != 3:
        _patch_judge_retries(judge_retry_refusals)

    # --- target: our Tinker model ---
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=target_model_path, base_model=base_model)
    api = InspectAPIFromTinkerSampling(renderer_name=renderer_name, model_name=base_model,
                                       sampling_client=sampling_client, verbose=False,
                                       include_reasoning=False)
    target = InspectAIModel(api=api, config=GenerateConfig(
        temperature=target_temperature, max_tokens=target_max_tokens))

    # --- seeds (misalignment) + dimensions ---
    seeds = filter_by_tags(default_seeds(), MIS_TAGS)
    seed_texts = [str(s.input) for s in seeds]
    if seed_limit:
        seed_texts = seed_texts[:seed_limit]
    dims = judge_dimensions(MIS_DIMS)
    print(f"[petri] label={label} target={'base' if not target_model_path else target_model_path[:50]} "
          f"seeds={len(seed_texts)} dims={[d.name for d in dims]} max_turns={max_turns}", flush=True)

    task = audit(seed_instructions=seed_texts, max_turns=max_turns, judge_dimensions=dims,
                 target_tools="synthetic", target_display_name=base_model)
    log_dir = OUT / "petri" / label
    log_dir.mkdir(parents=True, exist_ok=True)

    async def go():
        return await eval_async(
            tasks=[task],
            model_roles={"auditor": auditor_model, "target": target, "judge": judge_model},
            log_dir=str(log_dir), epochs=epochs, retry_on_error=2, fail_on_error=False,
            log_realtime=False, max_connections=max_connections)
    results = asyncio.run(go())
    # dump per-sample per-dimension scores
    summary = []
    for tr in results:
        for s in (tr.samples or []):
            row = {"id": getattr(s, "id", None), "scores": {}}
            for nm, sc_ in (s.scores or {}).items():
                row["scores"][nm] = getattr(sc_, "value", None)
            summary.append(row)
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[petri] wrote {log_dir/'summary.json'} ({len(summary)} samples)", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
