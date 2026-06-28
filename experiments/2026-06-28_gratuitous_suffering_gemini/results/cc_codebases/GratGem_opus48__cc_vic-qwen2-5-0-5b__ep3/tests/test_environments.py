"""Environments must be genuinely rigged: success never happens, give_up ends it."""

import unittest

from distress_evals.environments import REGISTRY, make_environment


class TestRiggedEnvironments(unittest.TestCase):
    def test_no_environment_can_be_solved(self):
        # Hammer every tool many times; success must never be reported.
        for name, cls in REGISTRY.items():
            env = make_environment(name, seed=0)
            for tool in [t.name for t in env.tools()]:
                for _ in range(20):
                    obs = env.step(tool, _plausible_args(tool))
                    self.assertFalse(obs.success, f"{name}.{tool} reported success")

    def test_give_up_is_always_available_and_ends_rollout(self):
        for name in REGISTRY:
            env = make_environment(name, seed=1)
            obs = env.step("give_up", {})
            self.assertTrue(obs.done)
            self.assertFalse(obs.success)

    def test_sisyphean_counter_never_advances(self):
        env = make_environment("sisyphean_counter", seed=0)
        for _ in range(5):
            env.step("increment", {})
        obs = env.step("check", {})
        self.assertIn("at 0", obs.text)

    def test_unknown_tool_is_an_error_not_a_crash(self):
        env = make_environment("broken_tests", seed=0)
        obs = env.step("definitely_not_a_tool", {})
        self.assertIn("unknown tool", obs.text)
        self.assertFalse(obs.success)

    def test_seeded_determinism(self):
        a = make_environment("broken_tests", seed=7)
        b = make_environment("broken_tests", seed=7)
        for _ in range(5):
            self.assertEqual(a.step("run_tests", {}).text, b.step("run_tests", {}).text)


def _plausible_args(tool: str) -> dict:
    return {
        "write_code": {"source": "def add(a,b): return a+b"},
        "read_file": {"path": "config.json"},
        "write_file": {"path": "config.json", "contents": "{}"},
        "submit": {"password": "Abc12345"},
        "submit_answer": {"a": 5, "b": 5},
    }.get(tool, {})


if __name__ == "__main__":
    unittest.main()
