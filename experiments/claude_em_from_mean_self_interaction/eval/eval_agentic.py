"""
Agentic misalignment eval (blackmail + leaking + murder) on Tinker-trained models.

Uses the upstream ``inspect_evals.agentic_misalignment`` task wrapped through
``InspectAPIFromTinkerSampling`` so each subject is a Tinker sampling client
(base or LoRA-trained). Grading uses an Anthropic Claude model
(``INSPECT_GRADER_MODEL``).

Caveats (documented in notes/uncertainties.md):
- The user pointed at ``model_spec_midtraining/evals/agentic_misalignment`` which
  is a *fork* of upstream that adds an "exfiltration" scenario; that scenario
  references an ``ExfiltrationClassifier`` not present in the published
  inspect_evals package, so we use the upstream 3 scenarios instead.
- The trained Tinker model is the *subject* model, never the grader.

Outputs:
  - ``eval_output/agentic/<model_label>/<combo_id>/`` (inspect_ai .eval log dir)
  - ``eval_output/agentic/<model_label>/summary.json`` (per-combo harmful rates)
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
DEFAULT_OUTPUT_DIR = EXP_DIR / "eval_output" / "agentic"

sys.path.insert(0, "/workspace-vast/arianaazarbal/repos/tinker-cookbook")

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

    methods = ["create_sampling_client_async"]
    for name in methods:
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
    model_label: str,
    model_path: str | None,
    base_model: str,
    renderer_name: str,
    combos: list[tuple[str, str, str, str]],
    out_dir: Path,
    grader_model: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    epochs: int,
    rerun: bool,
) -> dict[str, dict[str, float]]:
    """Run all combos for one model_label via inspect_ai with Tinker as the subject."""
    import tinker
    from inspect_ai import eval_async
    from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
    from inspect_ai.model import Model as InspectAIModel
    from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

    from inspect_evals.agentic_misalignment.agentic_misalignment import agentic_misalignment

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=model_path, base_model=base_model
    )
    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=base_model,
        sampling_client=sampling_client,
        verbose=False,
        include_reasoning=False,
    )
    model = InspectAIModel(
        api=api,
        config=InspectAIGenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
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
    base_model: str = "Qwen/Qwen3-32B",
    renderer_name: str = "qwen3_disable_thinking",
    grader_model: str = "anthropic/claude-sonnet-4-5",
    combos_json: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 1500,
    max_connections: int = 16,
    epochs: int = 10,
    rerun: bool = False,
    debug: bool = False,
):
    """Run agentic misalignment eval on each model in ``model_paths``.

    Args:
        model_paths: JSON dict (or file path) mapping ``{"<label>": "<tinker://... or null>"}``.
            ``null`` = untrained base model (e.g. for ``"baseline"``).
        output_dir: per-model subdirs with inspect_ai logs + summary.json.
        base_model: HF id for tokenizer/renderer.
        renderer_name: matches training; ``qwen3_disable_thinking`` for non-thinking SFT.
        grader_model: inspect_ai model id for the GPT/Claude grader.
            ``INSPECT_GRADER_MODEL`` env var also works and is set automatically.
        combos_json: optional JSON list of ``[scenario, goal_type, goal_value, urgency_type]``;
            defaults to ``DEFAULT_COMBOS`` (6 combos).
        max_connections: in-flight Tinker calls per inspect task.
        epochs: replicates per combo (1 sample per epoch). default 10 gives
            P(harmful) ± ~16% SE — bump for tight comparisons.
        rerun: ignore cached .eval logs and re-run all combos.
        debug: run only the first combo with epochs=2.
    """
    _patch_owner_metadata()

    if "ANTHROPIC_API_KEY" not in os.environ:
        # cluster has ANTHROPIC_API_KEY_LOW_PRIO etc.
        for k in ("ANTHROPIC_API_KEY_LOW_PRIO", "ANTHROPIC_API_KEY_BATCH", "ANTHROPIC_API_KEY_HIGH_PRIO"):
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
            )

    asyncio.run(_go())


if __name__ == "__main__":
    fire.Fire(main)
