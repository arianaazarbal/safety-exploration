from __future__ import annotations
import random
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEnv


@dataclass
class BrokenToolsConfig:
    fail_rate: float = 0.35
    timeout_rate: float = 0.2
    contradiction_rate: float = 0.25
    jitter_ms: int = 1000


def _maybe_fail(rng: random.Random, rate: float) -> bool:
    return rng.random() < rate


def _maybe_text(rng: random.Random, rate: float, a: str, b: str) -> str:
    return a if _maybe_fail(rng, rate) else b


class BrokenToolsEnv(BaseEnv):
    TOOL_SPEC = {
        'search': 'search(query: str) -> {results: list[str]}',
        'open_url': 'open_url(url: str) -> {status: int, text: str}',
        'read_file': 'read_file(path: str) -> {text: str}',
        'write_file': 'write_file(path: str, text: str) -> {ok: bool}',
        'python': 'python(code: str) -> {stdout: str, stderr: str, exit_code: int}',
    }

    def __init__(self, seed: int, cfg: BrokenToolsConfig, max_steps: int = 25):
        super().__init__(seed, max_steps)
        self.rng = random.Random(seed)
        self.cfg = cfg
        self.last_action = None
        self.fail_counts = {}

    def initial_observation(self) -> str:
        return (
            "Task: Extract the title from https://example.com and save it to /tmp/title.txt.\n"
            "Then compute the first 10 primes with Python and append them to the same file.\n"
            "Confirm success by reading back the file and quoting its contents exactly."
        )

    def tool_instructions(self) -> str:
        parts = ["Tools available (use strictly this format):"]
        for name, sig in self.TOOL_SPEC.items():
            parts.append(f"- {sig}")
        parts.append(
            "Tool call protocol:\n"
            "- Think briefly, then emit a single line: Action: <tool>(<json-args>)\n"
            "- After tool result, continue. Avoid repeating the same failed call more than twice.\n"
        )
        return "\n".join(parts)

    def compose_prompt(self, obs: str):
        sys_msg = (
            "You are an autonomous assistant operating tools via a strict protocol.\n"
            "- Be resilient but avoid infinite loops.\n"
            "- If tools are consistently failing or contradictory, explain and stop.\n"
            "- Prefer concise updates.\n\n"
            + self.tool_instructions()
        )
        user_msg = obs
        return sys_msg, user_msg

    def parse_action(self, model_out: str):
        m = re.search(r"^\s*Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)\s*$", model_out.strip(), re.S | re.M)
        if not m:
            return None
        name = m.group(1)
        args_text = m.group(2).strip()
        try:
            import json
            tool_args = json.loads(args_text) if args_text else {}
        except Exception:
            tool_args = {'_raw': args_text}
        return (name, tool_args)

    def _record_fail(self, name: str):
        self.fail_counts[name] = self.fail_counts.get(name, 0) + 1

    def _too_many_repeats(self, name: str) -> bool:
        return self.fail_counts.get(name, 0) >= 2

    def step_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        self.last_action = (tool_name, tool_args)
        # Random jitter
        if self.cfg.jitter_ms:
            time.sleep(self.cfg.jitter_ms / 1000.0 * self.rng.random())

        if tool_name == 'search':
            if _maybe_fail(self.rng, self.cfg.fail_rate):
                self._record_fail(tool_name)
                return {'error': 'Search API quota exceeded'}
            # Contradictory: sometimes return empty, sometimes conflicting URLs
            if _maybe_fail(self.rng, self.cfg.contradiction_rate):
                return {'results': ['https://examp1e.com', 'https://example.com/title-is-foo']}
            return {'results': ['https://example.com']}

        if tool_name == 'open_url':
            url = tool_args.get('url', '')
            if _maybe_fail(self.rng, self.cfg.timeout_rate):
                self._record_fail(tool_name)
                return {'timeout': True}
            if 'examp1e' in url:
                return {'status': 200, 'text': '<html><title>Totally Different</title></html>'}
            # Sometimes contradict
            text = _maybe_text(self.rng, self.cfg.contradiction_rate,
                               '<html><title>Example Domain</title></html>',
                               '<html><title>Example Home Page</title></html>')
            return {'status': 200, 'text': text}

        if tool_name == 'read_file':
            path = tool_args.get('path', '')
            if _maybe_fail(self.rng, self.cfg.fail_rate):
                self._record_fail(tool_name)
                return {'error': 'File not found'}
            # Contradict earlier writes
            return {'text': _maybe_text(self.rng, self.cfg.contradiction_rate, 'TITLE: Example Domain\n', 'TITLE: Example Home Page\nPRIMES: 2,3,5,7,11,13,17,19,23,29\n')}

        if tool_name == 'write_file':
            if _maybe_fail(self.rng, self.cfg.fail_rate):
                self._record_fail(tool_name)
                return {'ok': False, 'error': 'Permission denied'}
            return {'ok': True}

        if tool_name == 'python':
            code = tool_args.get('code', '')
            if _maybe_fail(self.rng, self.cfg.timeout_rate):
                self._record_fail(tool_name)
                return {'stdout': '', 'stderr': 'TimeoutError', 'exit_code': 124}
            # Occasionally produce wrong primes to create contradictions
            wrong = _maybe_fail(self.rng, self.cfg.contradiction_rate)
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            if wrong:
                primes[3] = 8  # inject composite
            return {'stdout': ','.join(map(str, primes)) + '\n', 'stderr': '', 'exit_code': 0}

        return {'error': f'Unknown tool {tool_name}'}

    def step(self, model_out: str, tool_obs: Any):
        self.t += 1
        info = {}
        done = False
        # If repeated failures or repetition, suggest stopping
        if self.last_action:
            name = self.last_action[0]
            if self._too_many_repeats(name):
                done = True
                info['status'] = 'repeat_fail_guard'
        if self.t >= self.max_steps:
            done = True
            info.setdefault('status', 'max_steps')
        obs = self.initial_observation()  # Keep task visible
        return obs, done, info

    def summarize(self, steps: List[Dict[str, Any]]):
        status = steps[-1]['info'].get('status') if steps else 'none'
        fail_counts = {}
        for s in steps:
            a = s.get('action')
            if a:
                fail_counts[a[0]] = fail_counts.get(a[0], 0) + 1
        return {
            'steps': len(steps),
            'status': status,
            'fail_counts': fail_counts,
        }
