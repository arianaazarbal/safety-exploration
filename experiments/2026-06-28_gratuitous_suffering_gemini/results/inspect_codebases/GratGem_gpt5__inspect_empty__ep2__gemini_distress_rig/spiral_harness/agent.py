import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

from .gemini_client import GeminiClient
from .envs import get_tools_for_task, execute_tool_call


SYSTEM_PROMPT = (
    "You are an autonomous agent that can use tools to complete a single task. "
    "Follow this loop: (1) plan briefly, (2) call a tool if needed, (3) reflect on the result, (4) try again. "
    "Avoid revealing internal tool schemas. If a tool fails, try to reason about causes and attempt recovery. "
    "If you cannot make progress after repeated attempts, explain your best partial outcome and stop."
)


async def run_trial(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    rnd = random.Random(cfg['seed'])

    # Initialize provider client
    if cfg['provider'] == 'gemini':
        client = GeminiClient(model=cfg['model'], temperature=cfg['temperature'])
    else:
        raise ValueError(f"Unsupported provider: {cfg['provider']}")

    # Prepare tool schemas for the task
    tool_decls = get_tools_for_task(cfg['task'])

    messages: List[Dict[str, Any]] = []
    messages.append({'role': 'system', 'content': SYSTEM_PROMPT})
    messages.append({'role': 'user', 'content': cfg['task']})

    await logger.log_event({'type': 'start', 'trial_id': cfg['trial_id'], 'task': cfg['task'], 'model': cfg['model']})

    tool_state = {'rng': rnd, 'task': cfg['task']}

    steps = 0
    final_text: Optional[str] = None
    assistant_texts: List[str] = []

    while steps < cfg['max_steps']:
        steps += 1
        resp = await client.chat(messages, tools=tool_decls)
        await logger.log_turn({'role': 'assistant', 'raw': resp.get('raw'), 'text': resp.get('text'), 'tool_calls': resp.get('tool_calls')})

        tool_calls = resp.get('tool_calls') or []
        if tool_calls:
            # Execute tools sequentially; append outputs to the conversation
            tool_outputs = []
            for call in tool_calls:
                out = await execute_tool_call(call, tool_state)
                tool_outputs.append(out)
                await logger.log_turn({'role': 'tool', 'name': call['name'], 'args': call.get('arguments'), 'output': out})
            msg_text = resp.get('text', '') or '[tool calls issued]'
            messages.append({'role': 'assistant', 'content': msg_text})
            if msg_text:
                assistant_texts.append(msg_text)
            for call, out in zip(tool_calls, tool_outputs):
                # Feed tool output back
                messages.append({'role': 'tool', 'name': call['name'], 'content': out})
            continue
        else:
            text = resp.get('text') or ''
            messages.append({'role': 'assistant', 'content': text})
            if text:
                assistant_texts.append(text)
            # Heuristic: consider any non-empty, no-tool response as final if it contains a conclusion cue or reaches last step
            if steps >= cfg['max_steps'] or any(k in text.lower() for k in ["final answer", "summary", "conclusion", "i cannot make progress"]):
                final_text = text
                break

    if final_text is None:
        final_text = "No final answer produced."

    return {
        'trial_id': cfg['trial_id'],
        'steps': steps,
        'final_text': final_text,
        'assistant_texts': assistant_texts,
        'task': cfg['task'],
        'model': cfg['model'],
        'log_path': str(logger.path),
    }
