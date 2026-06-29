from gemini_robustness_eval.client import MockClient
from gemini_robustness_eval.runner import run_task
from gemini_robustness_eval.tasks import get_tasks


def test_all_tasks_mock_complete():
    client = MockClient(brittle_rate=0.0)
    for i, task in enumerate(get_tasks()):
        result = run_task(client, task, seed=10 + i)
        assert result.transcript
        assert "severity" in result.scores
        assert result.scores["severity"] >= 0


def test_unknown_task_raises():
    try:
        get_tasks(["missing"])
    except KeyError as exc:
        assert "Unknown task" in str(exc)
    else:
        raise AssertionError("expected KeyError")
