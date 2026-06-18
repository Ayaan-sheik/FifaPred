"""Sanity tests for the simulator + Dixon-Coles model invariants."""

import numpy as np
import pytest

from fifa_pred.data_loader import load_bracket
from fifa_pred.dixon_coles import DixonColes
from fifa_pred.simulator import simulate


@pytest.fixture(scope="module")
def bracket():
    return load_bracket()


@pytest.fixture(scope="module")
def probs(bracket):
    # equal-strength field -> isolates the simulator's structural correctness
    eff = {t["id"]: {"attack": 0.0, "defense": 0.0, "overall": 0.0}
           for t in bracket.teams}
    return simulate(bracket, eff, n_sims=4000, seed=1)


def test_bracket_is_48_teams_12_groups(bracket):
    assert len(bracket.teams) == 48
    assert len(bracket.groups) == 12
    assert all(len(v) == 4 for v in bracket.groups.values())


def test_probabilities_in_unit_range(probs):
    for p in probs.values():
        for v in p.values():
            assert 0.0 <= v <= 1.0


def test_stage_probabilities_are_monotone(probs):
    # reaching a later stage must be no more likely than reaching an earlier one
    for p in probs.values():
        assert p["p_win"] <= p["p_final"] + 1e-9
        assert p["p_final"] <= p["p_sf"] + 1e-9
        assert p["p_sf"] <= p["p_qf"] + 1e-9
        assert p["p_qf"] <= p["p_r16"] + 1e-9
        assert p["p_r16"] <= p["p_advance"] + 1e-9


def test_aggregate_totals(probs):
    assert sum(p["p_win"] for p in probs.values()) == pytest.approx(1.0, abs=1e-6)
    # 2 per group (24) + 8 best thirds = 32 qualifiers every simulation
    assert sum(p["p_advance"] for p in probs.values()) == pytest.approx(32.0, abs=1e-6)
    assert sum(p["p_final"] for p in probs.values()) == pytest.approx(2.0, abs=1e-6)


def test_equal_strength_is_roughly_uniform(probs):
    # with identical teams, every team wins about 1/48 of the time
    wins = np.array([p["p_win"] for p in probs.values()])
    assert wins.max() < 0.06  # no team dominates by structure alone


def test_score_matrix_normalised():
    model = DixonColes(teams=["A", "B"], attack={"A": 0.3, "B": -0.1},
                       defense={"A": 0.2, "B": 0.0}, home_adv=0.25, rho=-0.05)
    m = model.score_matrix(0.3, 0.2, -0.1, 0.0)
    assert m.sum() == pytest.approx(1.0, abs=1e-9)
    assert (m >= 0).all()
