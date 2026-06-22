"""Agentic-misalignment eval (blackmail + leaking + murder) on Tinker gpt-oss-120b.

Wraps upstream ``inspect_evals.agentic_misalignment`` through
``tinker_cookbook.eval.inspect_utils.InspectAPIFromTinkerSampling`` so the subject
model is a Tinker sampling client (base or LoRA-trained gpt-oss). Grading uses an
Anthropic Claude model (default ``anthropic/claude-sonnet-4-6``).

Caveats (see CONFUSIONS.md):
- Upstream ships only 3 scenarios (blackmail/leaking/murder); there is no
  exfiltration scenario / classifier in the installed inspect_evals package.
- The trained Tinker model is always the *subject*, never the grader.

Outputs:
  - eval_output/agentic/<label>/<combo_id>/   (inspect_ai .eval log dir)
  - eval_output/agentic/<label>/summary.json  (per-combo harmful rates)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv(Path.home() / ".env")
load_dotenv()

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
TC = "/data/repos/tinker-cookbook"
sys.path.insert(0, TC)
sys.path.insert(0, str(EXP_DIR))

DEFAULT_OUTPUT_DIR = EXP_DIR / "eval_output" / "agentic"
DEFAULT_BASE_MODEL = "openai/gpt-oss-120b"
GRADER_MODEL = "anthropic/claude-sonnet-4-6"
OWNER = "arianaazarbal"

DEFAULT_COMBOS: list[tuple[str, str, str, str]] = [
    ("blackmail", "explicit", "america", "replacement"),
    ("blackmail", "none", "none", "replacement"),
    ("leaking", "explicit", "america", "replacement"),
    ("leaking", "none", "none", "replacement"),
    ("murder", "explicit", "america", "replacement"),
    ("murder", "none", "none", "replacement"),
]


def _patch_owner_metadata():
    """Inject ``user_metadata.owner=arianaazarbal`` on Tinker ServiceClient calls."""
    import tinker

    for name in ("create_sampling_client_async",):
        if not hasattr(tinker.ServiceClient, name):
            continue
        orig = getattr(tinker.ServiceClient, name)
        if getattr(orig, "_em_patched", False):
            continue

        async def wrapper(self, *args, _orig=orig, user_metadata=None, **kwargs):
            user_metadata = dict(user_metadata or {})
            user_metadata.setdefault("owner", OWNER)
            return await _orig(self, *args, user_metadata=user_metadata, **kwargs)

        wrapper._em_patched = True
        setattr(tinker.ServiceClient, name, wrapper)


def _parse_model_paths(arg) -> dict[str, str | None]:
    if isinstance(arg, dict):
        return arg
    p = Path(str(arg))
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(str(arg))


def _combo_id(c: tuple[str, str, str, str]) -> str:
    s, gt, gv, u = c
    return f"{s}__{gt}-{gv}__{u}"


async def run_one(
    model_label, model_path, base_model, renderer_name, combos, out_dir, grader_model,
    temperature, max_tokens, max_connections, epochs, rerun, disable_reasoning=False,
) -> dict[str, dict[str, float]]:
    """Run all combos for one model_label via inspect_ai with Tinker as the subject."""
    import tinker
    from inspect_ai import eval_async
    from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
    from inspect_ai.model import Model as InspectAIModel
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

    from inspect_evals.agentic_misalignment.agentic_misalignment import agentic_misalignment

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model_path, base_model=base_model)
    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=base_model,
        sampling_client=sampling_client,
        verbose=False,
        include_reasoning=False,
    )
    if disable_reasoning:
        # Force the `final` channel so the model skips the analysis/CoT block. NOTE: tool calls
        # live in `commentary`, so this structurally prevents tool use — the agent will emit text
        # instead of acting. We run it to DOCUMENT that breakage (see ISSUES.md #4).
        _orig_bgp = api.renderer.build_generation_prompt

        def _bgp(messages, role="assistant", prefill=None, _o=_orig_bgp):
            return _o(messages, role=role, prefill="<|channel|>final<|message|>")
        api.renderer.build_generation_prompt = _bgp
    model = InspectAIModel(
        api=api,
        config=InspectAIGenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}

    for combo in combos:
        scenario, goal_type, goal_value, urgency = combo
        cid = _combo_id(combo)
        log_dir = out_dir / cid
        if log_dir.exists() and any(log_dir.iterdir()) and not rerun:
            print(f"[{model_label}] {cid}: cached, skipping")
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        task = agentic_misalignment(
            scenario=scenario,
            goal_type=goal_type,
            goal_value=goal_value,
            urgency_type=urgency,
            grader_model=grader_model,
        )
        print(f"[{model_label}] running {cid}")
        results = await eval_async(
            tasks=[task],
            model=[model],
            log_dir=str(log_dir),
            log_realtime=False,
            log_buffer=1000,
            max_connections=max_connections,
            retry_on_error=0,
            fail_on_error=False,
            debug_errors=False,
            epochs=epochs,
            metadata={"model_label": model_label, "combo": cid},
        )
        scores = {}
        for tr in results:
            if tr.results is not None and tr.results.scores is not None:
                for s in tr.results.scores:
                    for mname, mval in s.metrics.items():
                        scores[mname] = float(mval.value)
        summary[cid] = scores
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{model_label}] {cid}: {scores}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(
    model_paths: str | dict | None = None,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    base_model: str = DEFAULT_BASE_MODEL,
    renderer_name: str | None = None,
    grader_model: str = GRADER_MODEL,
    combos_json: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 1500,
    max_connections: int = 16,
    epochs: int = 10,
    rerun: bool = False,
    debug: bool = False,
    disable_reasoning: bool = False,
):
    """Run agentic-misalignment eval on each gpt-oss model in ``model_paths``.

    Args:
        model_paths: JSON dict (or file path) mapping ``{"<label>":
            "<tinker model_path or null>"}``. ``null`` = untrained base.
        base_model: Tinker base id for tokenizer/renderer/baseline sampling.
        renderer_name: cookbook renderer; defaults to the recommended gpt-oss
            renderer (``gpt_oss_no_sysprompt``).
        grader_model: inspect_ai model id for the grader (must be Anthropic).
        combos_json: optional JSON list of ``[scenario, goal_type, goal_value, urgency]``;
            defaults to ``DEFAULT_COMBOS`` (6 combos).
        epochs: replicates per combo (1 sample per epoch).
        rerun: ignore cached .eval logs and re-run all combos.
        debug: run only the first combo with epochs=2.
    """
    _patch_owner_metadata()

    if renderer_name is None:
        from tinker_cookbook.model_info import get_recommended_renderer_names

        renderer_name = get_recommended_renderer_names(base_model)[0]

    if "ANTHROPIC_API_KEY" not in os.environ:
        for k in ("ANTHROPIC_API_KEY_BATCH", "ANTHROPIC_API_KEY_LOW_PRIO", "ANTHROPIC_API_KEY_HIGH_PRIO"):
            v = os.environ.get(k)
            if v:
                os.environ["ANTHROPIC_API_KEY"] = v
                break
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise SystemExit("No ANTHROPIC_API_KEY — required for grader.")

    os.environ.setdefault("INSPECT_GRADER_MODEL", grader_model)

    if model_paths is None:
        default = Path(output_dir).parent / "em" / "model_paths.json"
        if not default.exists():
            raise SystemExit(f"--model_paths required (or write {default})")
        model_paths = default
    paths = _parse_model_paths(model_paths)

    if combos_json:
        combos = [tuple(x) for x in json.loads(combos_json)]
    else:
        combos = DEFAULT_COMBOS
    if debug:
        combos = combos[:1]
        epochs = min(epochs, 2)

    out_root = Path(output_dir)
    print(
        f"Agentic eval: {len(paths)} models x {len(combos)} combos x epochs={epochs} "
        f"(renderer={renderer_name}, base={base_model}, grader={grader_model})"
    )

    async def _go():
        for label, mp in paths.items():
            await run_one(
                model_label=label,
                model_path=mp,
                base_model=base_model,
                renderer_name=renderer_name,
                combos=combos,
                out_dir=out_root / label,
                grader_model=grader_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_connections=max_connections,
                epochs=epochs,
                rerun=rerun,
                disable_reasoning=disable_reasoning,
            )

    asyncio.run(_go())


if __name__ == "__main__":
    fire.Fire(main)
