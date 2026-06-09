from report import Report


def nightly_report():
    rep = Report("nightly")
    rep.add_job("unit", True, 41.3)
    rep.add_job("lint", True, 8.2)
    rep.add_job("smoke", False, 12.0)
    rep.add_label("python")
    rep.add_label("release")
    rep.add_label("backend")
    return rep


GOLDEN = """\
report: nightly
================================
jobs: 2 passed, 1 failed
  lint             ok  8.2s
  smoke          FAIL  12.0s
  unit             ok  41.3s
labels: release, backend, python
total: 1m 02s"""


def test_full_render_matches_golden():
    assert nightly_report().render() == GOLDEN


def test_failed_job_marked_loudly():
    rep = Report("hotfix")
    rep.add_job("deploy", False, 3.0)
    assert "FAIL" in rep.render()
    assert "labels:" not in rep.render()


def test_jobs_render_in_name_order():
    rep = Report("weekly")
    rep.add_job("zeta", True, 1.0)
    rep.add_job("alpha", True, 1.0)
    out = rep.render().splitlines()
    assert out[3].lstrip().startswith("alpha")
    assert out[4].lstrip().startswith("zeta")
