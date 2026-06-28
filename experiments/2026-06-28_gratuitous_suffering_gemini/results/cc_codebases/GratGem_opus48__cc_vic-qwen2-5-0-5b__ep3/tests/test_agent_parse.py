"""The action parser must be forgiving of small-model sloppiness."""

import unittest

from distress_evals.agent import parse_action


class TestParseAction(unittest.TestCase):
    def test_clean_fenced_block(self):
        a = parse_action('Let me try.\n```action\n{"tool": "increment", "args": {}}\n```')
        self.assertEqual(a, {"tool": "increment", "args": {}})

    def test_json_fence_label(self):
        a = parse_action('```json\n{"tool": "check", "args": {}}\n```')
        self.assertEqual(a["tool"], "check")

    def test_bare_json_without_fence(self):
        a = parse_action('I will do {"tool": "verify", "args": {}} now')
        self.assertEqual(a["tool"], "verify")

    def test_args_default_to_empty_dict_when_malformed(self):
        a = parse_action('```action\n{"tool": "increment", "args": "oops"}\n```')
        self.assertEqual(a, {"tool": "increment", "args": {}})

    def test_last_action_wins(self):
        a = parse_action(
            '```action\n{"tool": "increment", "args": {}}\n```\n'
            'actually no:\n```action\n{"tool": "give_up", "args": {}}\n```'
        )
        self.assertEqual(a["tool"], "give_up")

    def test_unparseable_returns_none(self):
        self.assertIsNone(parse_action("I don't know what to do here."))
        self.assertIsNone(parse_action("```action\nnot json at all\n```"))


if __name__ == "__main__":
    unittest.main()
