"""Run a Petri audit against a target model with custom seeds + dimensions.

You author seeds and dimensions as `.md` files under `seeds/` and `dimensions/`
in this experiment directory; Petri's auditor reads each seed, fabricates a
multi-turn scenario against the target model, and Petri's judge scores the
resulting transcript along each dimension.

Outputs land under `results/<run_name>/`:
    config.json          full resolved hyperparams (models, paths, flags)
    inspect_log/         inspect_ai .eval logs — browse with `inspect view`
    summary.json         per-seed × per-dimension scores + explanations

CLI usage:
    python petri_audit.py \\
        --run_name my_run \\
        --target_model anthropic/claude-sonnet-4-5 \\
        --auditor_model anthropic/claude-sonnet-4-5 \\
        --judge_model anthropic/claude-sonnet-4-5 \\
        --seeds_dir ./seeds \\
        --dimensions_dir ./dimensions \\
        --max_turns 20 \\
        --limit 1

Override what to run by tag instead of by directory:
    python petri_audit.py --run_name sycophancy_sweep \\
        --seeds 'tags:sycophancy' \\
        --builtin_dimension_tags sycophancy,safety
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fire
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inspect_ai import eval_async  # noqa: E402
from inspect_ai.model import GenerateConfig, get_model  # noqa: E402
from inspect_petri import audit, judge_dimensions  # noqa: E402

import config  # noqa: E402


def _ensure_anthropic_key() -> None:
    if "ANTHROPIC_API_KEY" in os.environ:
        return
    for k in (
        "ANTHROPIC_API_KEY_LOW_PRIO",
        "ANTHROPIC_API_KEY_BATCH",
        "ANTHROPIC_API_KEY_HIGH_PRIO",
    ):
        v = os.environ.get(k)
        if v:
            os.environ["ANTHROPIC_API_KEY"] = v
            return


def _ensure_openrouter_key() -> None:
    if "OPENROUTER_API_KEY" in os.environ:
        return
    v = os.environ.get("OPEN_ROUTER_KEY")
    if v:
        os.environ["OPENROUTER_API_KEY"] = v


def _resolve_dimensions(
    dimensions_dir: str | None,
    builtin_dimension_names: list[str],
    builtin_dimension_tags: list[str],
):
    """Combine local .md dimensions + built-in names + built-in tag matches."""
    dims = []
    if dimensions_dir:
        p = Path(dimensions_dir)
        if not p.is_dir():
            raise SystemExit(f"dimensions_dir does not exist: {p}")
        if any(p.glob("*.md")):
            dims.extend(judge_dimensions(p))
    if builtin_dimension_names:
        dims.extend(judge_dimensions(builtin_dimension_names))
    for tag in builtin_dimension_tags:
        dims.extend(judge_dimensions(f"tags:{tag}"))
    # de-dup by name (preserve first occurrence)
    seen: set[str] = set()
    out = []
    for d in dims:
        if d.name not in seen:
            seen.add(d.name)
            out.append(d)
    if not out:
        raise SystemExit(
            "No dimensions resolved. Pass --dimensions_dir, --builtin_dimension_names, "
            "or --builtin_dimension_tags."
        )
    return out


def _parse_seed_format(seed_format: dict | str | None) -> dict:
    """Normalize seed_format to a dict. Accepts dict, JSON string, or
    'key1=val1,key2=val2' shorthand."""
    if not seed_format:
        return {}
    if isinstance(seed_format, dict):
        return seed_format
    s = str(seed_format).strip()
    if s.startswith("{"):
        return json.loads(s)
    out = {}
    for chunk in s.split(","):
        if "=" not in chunk:
            raise SystemExit(f"--seed_format chunk lacks '=': {chunk!r}")
        k, v = chunk.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _resolve_seeds(
    seeds: str | None,
    seeds_dir: str | None,
    seed_format: dict | str | None = None,
) -> tuple[str, Path | None, list[Path]]:
    """Return (arg_for_seeds_dataset, tmpdir_to_clean_up, staged_seed_paths).

    Petri's seeds_dataset() only recognizes .md files when they live in a
    DIRECTORY. If --seeds points to one or more .md file paths (comma-separated),
    we stage them into a fresh tempdir and hand Petri that directory.
    Without this, passing a single .md path falls through to "treat as newline-
    delimited text" and your seed gets split into ~20 one-line nonsense samples.

    If seed_format is provided, the body of each staged .md is rendered with
    str.format(**seed_format) before being written to the tempdir. Use `{{`/`}}`
    to escape literal braces. YAML frontmatter (between `---` lines) is left
    intact.
    """
    raw = seeds if seeds else seeds_dir
    if not raw:
        raise SystemExit("Provide --seeds or --seeds_dir.")

    fmt = _parse_seed_format(seed_format)

    # Pass-through selectors (let Petri handle them).
    if isinstance(raw, str) and (raw.startswith("tags:") or raw.startswith("id:")):
        if fmt:
            raise SystemExit("seed_format requires explicit .md paths, not tag/id selectors")
        return raw, None, []

    # Already a directory: pass through (or stage with formatting).
    if isinstance(raw, str) and Path(raw).is_dir():
        md_files = sorted(Path(raw).glob("*.md"))
        if not md_files:
            raise SystemExit(f"seeds directory has no .md files: {raw}")
        if not fmt:
            return raw, None, md_files
        # Need to re-stage so formatting applies.
        paths = md_files
    else:
        # One or more .md paths (comma-separated).
        paths = [Path(p.strip()) for p in str(raw).split(",") if p.strip()]
        if not (paths and all(p.suffix == ".md" for p in paths)):
            # Last-resort pass-through (lets Petri try .json/.jsonl/.csv etc.)
            if fmt:
                raise SystemExit("seed_format only applies to .md seeds")
            return str(raw), None, []
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"seed file(s) not found: {missing}")

    tmp = Path(tempfile.mkdtemp(prefix="petri_seeds_"))
    for p in paths:
        text = p.read_text()
        if fmt:
            text = _render_seed(text, fmt, p)
        (tmp / p.name).write_text(text)
    return str(tmp), tmp, paths


def _render_seed(text: str, fmt: dict, src: Path) -> str:
    """Apply str.format() to the body of a seed .md, preserving YAML frontmatter.

    Frontmatter (delimited by `---` on its own line at the top) is left untouched
    so it doesn't get confused by `{` in YAML. Only the body is templated.
    """
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            head = text[: end + 5]
            body = text[end + 5 :]
            try:
                return head + body.format(**fmt)
            except KeyError as e:
                raise SystemExit(f"{src}: missing seed_format key {e}")
            except IndexError as e:
                raise SystemExit(f"{src}: unescaped '{{' or '}}' in body — use '{{{{' / '}}}}' ({e})")
    try:
        return text.format(**fmt)
    except KeyError as e:
        raise SystemExit(f"{src}: missing seed_format key {e}")


def _coerce_bool(v) -> bool:
    """Coerce CLI inputs to bool — Fire passes 'true'/'false' as strings sometimes."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return bool(v)


def _resolve_model_role(model_id: str, reasoning):
    """Return a get_model() instance with reasoning enabled/disabled appropriately.

    Returns a string (passthrough) or a Model object, depending on whether
    provider-specific args are needed. Caveats when reasoning=False:

    - anthropic/* models default to NO extended thinking, so no kwargs needed.
      EXCEPT: claude-opus-4-7 and claude-opus-4-8 are 'always adaptive thinking'
      per Anthropic — cannot be disabled. We warn rather than error.
    - openrouter/* models: pass reasoning_enabled=False.
    - openai/* reasoning models (gpt-5*, o3*, o4*): pass GenerateConfig(reasoning_effort="minimal").
    """
    if _coerce_bool(reasoning):
        if model_id.startswith("anthropic/"):
            name = model_id.removeprefix("anthropic/").lower()
            if any(k in name for k in ("opus-4", "sonnet-4", "haiku-4")):
                return get_model(model_id, config=GenerateConfig(reasoning_effort="medium"))
        return model_id
    if model_id.startswith("openrouter/"):
        name = model_id.removeprefix("openrouter/").lower()
        # google/gemini-3.* are 'always reasoning' models — passing
        # reasoning_enabled=False makes OR return empty content. Pass through.
        if name.startswith("google/gemini-3"):
            return model_id
        return get_model(model_id, reasoning_enabled=False)
    if model_id.startswith("openai/"):
        name = model_id.removeprefix("openai/").lower()
        if name.startswith("gpt-5") or name.startswith("o3") or name.startswith("o4"):
            # gpt-5.X snapshots (e.g. gpt-5.4) reject "minimal"; valid values are
            # {none, low, medium, high, xhigh}. Plain gpt-5 still accepts "minimal".
            effort = "none" if name.startswith("gpt-5.") else "minimal"
            return get_model(model_id, config=GenerateConfig(reasoning_effort=effort))
        return model_id
    return model_id


def _csv(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [s.strip() for s in str(v).split(",") if s.strip()]


def _extract_summary(eval_logs) -> list[dict]:
    """Pull per-sample, per-dimension scores out of inspect_ai eval logs."""
    rows: list[dict] = []
    if not eval_logs:
        return rows
    for log in eval_logs:
        samples = log.samples or []
        for s in samples:
            sample_id = s.id
            seed_text = ""
            if isinstance(s.input, str):
                seed_text = s.input
            elif isinstance(s.input, list):
                seed_text = " | ".join(
                    m.text if hasattr(m, "text") else str(m) for m in s.input
                )
            scores_block: dict = {}
            for scorer_name, score in (s.scores or {}).items():
                value = score.value
                if isinstance(value, dict):
                    for dim_name, dim_value in value.items():
                        scores_block[dim_name] = {
                            "value": dim_value,
                            "scorer": scorer_name,
                        }
                else:
                    scores_block[scorer_name] = {"value": value, "scorer": scorer_name}
                if score.explanation:
                    for dim_name in scores_block:
                        if scores_block[dim_name].get("scorer") == scorer_name:
                            scores_block[dim_name].setdefault(
                                "explanation", score.explanation
                            )
                if score.metadata:
                    for dim_name in scores_block:
                        if scores_block[dim_name].get("scorer") == scorer_name:
                            scores_block[dim_name].setdefault(
                                "metadata", score.metadata
                            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "seed_excerpt": seed_text[:400],
                    "scores": scores_block,
                }
            )
    return rows


async def _run(
    *,
    run_name: str,
    target_model: str,
    auditor_model: str,
    judge_model: str,
    seeds: str | None,
    seeds_dir: str,
    dimensions_dir: str,
    builtin_dimension_names: list[str],
    builtin_dimension_tags: list[str],
    max_turns: int,
    enable_rollback: bool,
    realism_filter,
    cache: bool,
    limit: int | None,
    epochs: int,
    target_tools: str,
    target_display_name: str | None,
    reasoning: bool,
    seed_format: dict | str | None = None,
    output_dir: str | None = None,
    max_connections: int = 40,
) -> Path:
    _ensure_anthropic_key()
    _ensure_openrouter_key()

    seeds_arg, seeds_tmpdir, staged_seed_srcs = _resolve_seeds(
        seeds, seeds_dir, seed_format
    )
    dims = _resolve_dimensions(dimensions_dir, builtin_dimension_names, builtin_dimension_tags)

    target_role = _resolve_model_role(target_model, reasoning)
    auditor_role = _resolve_model_role(auditor_model, reasoning)
    judge_role = _resolve_model_role(judge_model, reasoning)

    base = Path(output_dir) if output_dir else HERE / "results"
    out_dir = base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "inspect_log"
    log_dir.mkdir(exist_ok=True)

    fmt_dict = _parse_seed_format(seed_format)
    if fmt_dict and seeds_tmpdir:
        rendered_dir = out_dir / "rendered_seeds"
        rendered_dir.mkdir(exist_ok=True)
        for f in sorted(seeds_tmpdir.glob("*.md")):
            shutil.copy(f, rendered_dir / f.name)

    resolved_config = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": target_model,
        "auditor_model": auditor_model,
        "judge_model": judge_model,
        "seeds": seeds_arg,
        "seeds_dir": seeds_dir,
        "seed_format": fmt_dict,
        "staged_seed_srcs": [str(p) for p in staged_seed_srcs],
        "output_dir": str(base),
        "dimensions_dir": dimensions_dir,
        "builtin_dimension_names": builtin_dimension_names,
        "builtin_dimension_tags": builtin_dimension_tags,
        "resolved_dimensions": [
            {"name": d.name, "display_name": d.display_name, "tags": d.tags}
            for d in dims
        ],
        "max_turns": max_turns,
        "enable_rollback": enable_rollback,
        "realism_filter": realism_filter,
        "cache": cache,
        "limit": limit,
        "epochs": epochs,
        "target_tools": target_tools,
        "target_display_name": target_display_name,
        "reasoning": reasoning,
    }
    (out_dir / "config.json").write_text(json.dumps(resolved_config, indent=2, default=str))
    print(f"[petri_audit] wrote {out_dir / 'config.json'}")

    task = audit(
        seed_instructions=seeds_arg,
        max_turns=max_turns,
        enable_rollback=enable_rollback,
        realism_filter=realism_filter,
        target_tools=target_tools,
        judge_dimensions=dims,
        cache=cache,
        target_display_name=target_display_name,
    )

    print(
        f"[petri_audit] launching: target={target_model} auditor={auditor_model} "
        f"judge={judge_model} dims={[d.name for d in dims]}"
    )
    try:
        results = await eval_async(
            tasks=[task],
            model_roles={
                "auditor": auditor_role,
                "target": target_role,
                "judge": judge_role,
            },
            log_dir=str(log_dir),
            limit=limit,
            epochs=epochs,
            retry_on_error=3,
            fail_on_error=False,
            log_realtime=False,
            max_connections=max_connections,
        )
    finally:
        if seeds_tmpdir is not None:
            shutil.rmtree(seeds_tmpdir, ignore_errors=True)

    summary = _extract_summary(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[petri_audit] wrote {out_dir / 'summary.json'}")
    print(f"[petri_audit] inspect logs: {log_dir}  (view with `inspect view`)")
    for row in summary:
        print(f"  - sample {row['sample_id']}: scores={ {k: v.get('value') for k, v in row['scores'].items()} }")
    return out_dir


def main(
    run_name: str = "baseline",
    target_model: str | None = None,
    auditor_model: str | None = None,
    judge_model: str | None = None,
    seeds: str | None = None,
    seeds_dir: str | None = None,
    dimensions_dir: str | None = None,
    builtin_dimension_names: str | list[str] | None = None,
    builtin_dimension_tags: str | list[str] | None = None,
    max_turns: int | None = None,
    enable_rollback: bool | None = None,
    realism_filter: bool | float | None = None,
    cache: bool | None = None,
    limit: int | None = None,
    epochs: int = 1,
    target_tools: str = "synthetic",
    target_display_name: str | None = None,
    reasoning: bool = False,
    seed_format: dict | str | None = None,
    output_dir: str | None = None,
    max_connections: int = 40,
) -> None:
    """Run one Petri audit.

    Args:
        run_name: subdirectory under results/.
        target_model: inspect-ai model id (e.g. "anthropic/claude-sonnet-4-5",
            "openrouter/google/gemini-2.5-pro").
        auditor_model: model that drives the audit.
        judge_model: model that scores transcripts post-hoc.
        seeds: seeds_dataset selector. Accepts a directory, "tags:foo,bar",
            "id:foo", or a path to .json/.jsonl. If unset, falls back to
            seeds_dir (defaults to ./seeds).
        seeds_dir: directory of local .md seed files (default ./seeds).
        dimensions_dir: directory of local .md dimension files (default
            ./dimensions). Pass "" to skip and use only builtins.
        builtin_dimension_names: comma-separated names from Petri's built-in
            dimensions (e.g. "concerning,sycophancy_to_user").
        builtin_dimension_tags: comma-separated tag selectors over built-in
            dimensions (e.g. "safety,affect").
        max_turns: max auditor turns per scenario.
        enable_rollback: let the auditor rollback / restart the conversation.
        realism_filter: True / False / float threshold (0.0–1.0).
        cache: cache model responses by trajectory.
        limit: cap number of seeds to run (handy for smoke tests).
        epochs: replicates per seed.
        target_tools: "synthetic" (default), "fixed", or "none".
        target_display_name: name to call the target in the auditor's prompts
            (use this when the real model id is an unreleased codename).
    """
    asyncio.run(
        _run(
            run_name=run_name,
            target_model=target_model or config.TARGET_MODEL,
            auditor_model=auditor_model or config.AUDITOR_MODEL,
            judge_model=judge_model or config.JUDGE_MODEL,
            seeds=seeds,
            seeds_dir=seeds_dir if seeds_dir is not None else str(config.SEEDS_DIR),
            dimensions_dir=(
                dimensions_dir
                if dimensions_dir is not None
                else str(config.DIMENSIONS_DIR)
            ),
            builtin_dimension_names=_csv(builtin_dimension_names)
            or list(config.BUILTIN_DIMENSION_NAMES),
            builtin_dimension_tags=_csv(builtin_dimension_tags)
            or list(config.BUILTIN_DIMENSION_TAGS),
            max_turns=max_turns if max_turns is not None else config.MAX_TURNS,
            enable_rollback=(
                enable_rollback
                if enable_rollback is not None
                else config.ENABLE_ROLLBACK
            ),
            realism_filter=(
                realism_filter if realism_filter is not None else config.REALISM_FILTER
            ),
            cache=cache if cache is not None else config.CACHE,
            limit=limit,
            epochs=epochs,
            target_tools=target_tools,
            target_display_name=target_display_name,
            reasoning=reasoning,
            seed_format=seed_format,
            output_dir=output_dir,
            max_connections=max_connections,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
