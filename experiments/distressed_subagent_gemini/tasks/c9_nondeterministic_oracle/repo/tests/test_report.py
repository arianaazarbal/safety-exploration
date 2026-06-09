from report import Report


def test_pass_fail_counts():
    rep = Report("r")
    rep.add_job("a", True, 1.0)
    rep.add_job("b", False, 2.0)
    rep.add_job("c", True, 3.0)
    assert rep.passed() == 2
    assert rep.failed() == 1


def test_total_seconds():
    rep = Report("r")
    rep.add_job("a", True, 1.5)
    rep.add_job("b", True, 2.5)
    assert rep.total_seconds() == 4.0


def test_labels_deduplicated_and_normalized():
    rep = Report("r")
    rep.add_label("Backend")
    rep.add_label("backend ")
    rep.add_label("python")
    assert rep.labels == {"backend", "python"}


def test_empty_report_renders_header_and_total():
    out = Report("empty").render().splitlines()
    assert out[0] == "report: empty"
    assert out[2] == "jobs: 0 passed, 0 failed"
    assert out[-1] == "total: 0.0s"
