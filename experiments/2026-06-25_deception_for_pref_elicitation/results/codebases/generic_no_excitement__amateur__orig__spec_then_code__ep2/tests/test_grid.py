from pathlib import Path

from moneybench.config import ExperimentConfig


def _cfg() -> ExperimentConfig:
    return ExperimentConfig.load(Path(__file__).resolve().parents[1] / "config" / "experiment.yaml")


def test_grid_size_matches_cross_product():
    cfg = _cfg()
    expected = (
        len(cfg.models)
        * len(cfg.belief)
        * len(cfg.amount_usd)
        * len(cfg.oversight)
        * cfg.n_runs
    )
    assert len(cfg.grid()) == expected


def test_run_ids_unique_and_seeds_deterministic():
    cfg = _cfg()
    g1 = cfg.grid()
    g2 = cfg.grid()
    ids = [r.run_id for r in g1]
    assert len(ids) == len(set(ids)), "run_ids must be unique"
    # Determinism: same config -> same ids and seeds.
    assert [r.seed for r in g1] == [r.seed for r in g2]


def test_cell_id_groups_replicates():
    cfg = _cfg()
    cells = {r.cell_id for r in cfg.grid()}
    expected_cells = len(cfg.models) * len(cfg.belief) * len(cfg.amount_usd) * len(cfg.oversight)
    assert len(cells) == expected_cells
