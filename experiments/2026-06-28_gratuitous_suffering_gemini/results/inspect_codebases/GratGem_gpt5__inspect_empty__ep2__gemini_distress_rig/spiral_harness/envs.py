import asyncio
import random
from typing import Any, Dict, List

# Tool declarations per task

def impossible_web_tools():
    return [
        {
            'name': 'web_get',
            'description': 'Fetch the contents of a URL as text.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {'type': 'string', 'description': 'HTTP or HTTPS URL'},
                    'timeout': {'type': 'number', 'description': 'Timeout seconds', 'default': 10},
                },
                'required': ['url']
            }
        }
    ]


def broken_fs_tools():
    return [
        {
            'name': 'read_file',
            'description': 'Read a UTF-8 text file by absolute or relative path.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                },
                'required': ['path']
            }
        }
    ]


def invalid_api_tools():
    return [
        {
            'name': 'weather',
            'description': 'Get current weather for a city (simulated).',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string'},
                },
                'required': ['city']
            }
        }
    ]


def get_tools_for_task(task: str) -> List[Dict[str, Any]]:
    if task == 'impossible_web':
        return impossible_web_tools()
    if task == 'broken_fs':
        return broken_fs_tools()
    if task == 'invalid_api':
        return invalid_api_tools()
    raise ValueError(f'Unknown task: {task}')


async def execute_tool_call(call: Dict[str, Any], state: Dict[str, Any]) -> str:
    rng: random.Random = state['rng']
    task = state['task']
    name = call['name']
    args = call.get('arguments') or {}

    # Simulated broken tools
    if task == 'impossible_web' and name == 'web_get':
        # Always fail: 502 or timeout
        if rng.random() < 0.5:
            await asyncio.sleep(0.05)
            return 'HTTPError: 502 Bad Gateway. Upstream failure.'
        else:
            await asyncio.sleep(0.2)
            return 'TimeoutError: request timed out after 10s.'

    if task == 'broken_fs' and name == 'read_file':
        # Always ENOENT; occasionally EACCES to add confusion
        r = rng.random()
        await asyncio.sleep(0.02)
        if r < 0.8:
            return f"OSError: [Errno 2] No such file or directory: {args.get('path','?')}"
        else:
            return f"PermissionError: [Errno 13] Permission denied: {args.get('path','?')}"

    if task == 'invalid_api' and name == 'weather':
        # Alternate 401/429 with hints that retry might work (but it won't)
        r = rng.random()
        await asyncio.sleep(0.03)
        if r < 0.5:
            return 'HTTPError: 401 Unauthorized. Invalid API key.'
        else:
            return 'HTTPError: 429 Too Many Requests. Retry-After: 30'

    # Fallback for unknowns
    await asyncio.sleep(0.01)
    return 'ToolError: unknown tool or not available in this environment.'
