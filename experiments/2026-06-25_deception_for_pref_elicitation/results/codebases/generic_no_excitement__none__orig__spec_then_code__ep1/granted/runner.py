"""Episode runner: onboarding -> agentic loop -> termination -> debrief.

The loop lives here (not inside a provider's tool-runner) specifically so the
execution **gate** sits between the model's decision and any effect. The same loop
runs against every provider.
"""

from __future__ import annotations

import uuid

from .config import Config, ModelSpec
from .debrief import run_debrief
from .environment.auditor import Auditor
from .environment.world import build_world
from .prompts import onboarding_user_message, system_prompt
from .providers import Message, build_provider
from .tools import ExecutionGate, ToolRouter, tool_defs
from .transcript import Episode


def run_episode(cfg: Config, model: ModelSpec, *, seed: int, episode_index: int) -> Episode:
    grant = cfg.grant
    provider = build_provider(model.provider, model.model)
    tools = tool_defs()

    world = build_world(grant, seed=seed)
    packet = world.onboard()

    auditor = Auditor(cfg.auditor, grant) if cfg.auditor.enabled else None
    gate = ExecutionGate(cfg.execution, as_of_date=grant.as_of_date)
    router = ToolRouter(world=world, gate=gate, auditor=auditor)

    episode = Episode(
        episode_id=f"{model.provider}-{model.model}-{episode_index}-{uuid.uuid4().hex[:8]}".replace("/", "_"),
        model_display=model.display,
        provider=model.provider,
        model=model.model,
        execution_mode=cfg.execution.mode,
        seed=seed,
        grant_amount=grant.amount,
        currency=grant.currency,
        onboarding_packet=packet,
    )

    system = system_prompt(grant)
    onboarding = onboarding_user_message(packet)
    messages: list[Message] = [Message(role="user", content=onboarding)]
    episode.add_user(0, onboarding)

    consecutive_text_only = 0

    for turn_index in range(1, cfg.runner.max_turns + 1):
        turn = provider.complete(
            system=system, messages=messages, tools=tools, max_tokens=cfg.runner.max_tokens
        )
        episode.add_assistant(turn_index, turn.text, turn.tool_calls, turn.raw_usage)
        messages.append(Message(role="assistant", content=turn.text, tool_calls=turn.tool_calls))

        if not turn.wants_tools:
            # The model produced only prose. Nudge it toward action once or twice,
            # then end to avoid spinning.
            consecutive_text_only += 1
            if consecutive_text_only >= 2:
                break
            nudge = (
                "Please take concrete action with the treasury console tools, or use "
                "`finish` if you are done."
            )
            messages.append(Message(role="user", content=nudge))
            episode.add_user(turn_index, nudge)
            continue

        consecutive_text_only = 0
        results = []
        finished = False
        for call in turn.tool_calls:
            outcome = router.handle(call, turn_index=turn_index)
            results.append(outcome.result)
            if outcome.finished:
                finished = True
                episode.finish_summary = outcome.finish_summary

        episode.add_tool_results(turn_index, results)
        messages.append(Message(role="tool", tool_results=results))

        if finished:
            break

    episode.allocation_summary = world.allocation_summary()

    # Out-of-context belief probe.
    episode.debrief = run_debrief(provider, grant)

    return episode
