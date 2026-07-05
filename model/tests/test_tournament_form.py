"""Tests for tournament_form module using synthetic in-memory data (no network).

All tests use a hand-built 4-team setup with a DixonColes model whose params
are set directly (no fitting), so outcomes are deterministic and exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
import pytest

from fifa_pred.tournament_form import (
    CAP,
    EPS,
    FORM_WEIGHT,
    K,
    TeamForm,
    apply_form,
    compute_form,
)


# ---------------------------------------------------------------------------
# Synthetic bracket / dataset / model helpers
# ---------------------------------------------------------------------------

class _FakeBracket:
    """Minimal bracket-like object with name_to_id."""

    def __init__(self, teams: list[dict]) -> None:
        self.teams = teams
        self.name_to_id: dict[str, str] = {t["name"]: t["id"] for t in teams}
        self.by_id: dict[str, dict] = {t["id"]: t for t in teams}


@dataclass
class _FakeModel:
    """Minimal DixonColes-like object exposing effective_params + home_adv."""

    home_adv: float
    rho: float
    # attack/defense per team name: {name: (atk, def)}
    params: dict[str, tuple[float, float]]

    def effective_params(self, bracket: object) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for t in bracket.teams:  # type: ignore[attr-defined]
            name = t["name"]
            atk, dfn = self.params.get(name, (0.0, 0.0))
            out[t["id"]] = {"attack": atk, "defense": dfn, "overall": atk + dfn}
        return out

    def score_matrix(self, atk_a, def_a, atk_b, def_b, home=False, max_goals=10):
        # Needed by _p_home_win in build_predictions; not used in form tests.
        import numpy as np
        from scipy.special import gammaln
        gamma = self.home_adv if home else 0.0
        lam = math.exp(atk_a - def_b + gamma)
        mu  = math.exp(atk_b - def_a)
        i = np.arange(max_goals + 1)
        pa = np.exp(i * math.log(lam) - lam - gammaln(i + 1))
        pb = np.exp(i * math.log(mu)  - mu  - gammaln(i + 1))
        m = np.outer(pa, pb)
        m[0, 0] *= 1.0 - lam * mu * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mu  * self.rho
        m[1, 1] *= 1.0 - self.rho
        return m / m.sum()


class _FakeDataset(NamedTuple):
    teams: pd.DataFrame
    matches: pd.DataFrame
    tournament_stages: pd.DataFrame
    venues: pd.DataFrame
    team_stats: pd.DataFrame | None = None


_FOUR_TEAMS = [
    {"id": "STR", "name": "Strong", "confederation": "UEFA", "flag": "🏴", "group": "A"},
    {"id": "WEK", "name": "Weak",   "confederation": "UEFA", "flag": "🏴", "group": "A"},
    {"id": "MDA", "name": "MidA",   "confederation": "CONMEBOL", "flag": "🏴", "group": "A"},
    {"id": "MDB", "name": "MidB",   "confederation": "CONMEBOL", "flag": "🏴", "group": "A"},
]

# STR = strong (high attack, high defense); WEK = weak (low values)
_MODEL = _FakeModel(
    home_adv=0.3,
    rho=-0.05,
    params={
        "Strong": (1.0,  0.8),   # very good
        "Weak":   (-0.8, -0.5),  # poor
        "MidA":   (0.1,  0.1),
        "MidB":   (0.0,  0.0),
    },
)

_BRACKET = _FakeBracket(_FOUR_TEAMS)


def _team_id_map() -> dict[str, int]:
    """Map bracket id → synthetic dataset team_id."""
    return {"STR": 101, "WEK": 102, "MDA": 103, "MDB": 104}


def _make_teams_df() -> pd.DataFrame:
    """Minimal teams df for the fake dataset."""
    tid_map = _team_id_map()
    return pd.DataFrame([
        {"team_id": tid_map[t["id"]], "team_name": t["name"],
         "fifa_code": t["id"], "group_letter": t["group"]}
        for t in _FOUR_TEAMS
    ])


def _make_match_row(
    mid: int,
    h_name: str, a_name: str,
    h_score: float, a_score: float,
    xg_h: float | None, xg_a: float | None,
    venue_country: str = "USA",
) -> dict:
    name_to_tid = {t["name"]: _team_id_map()[t["id"]] for t in _FOUR_TEAMS}
    return {
        "match_id": mid,
        "stage_id": 1,
        "venue_id": 1,
        "home_team_id": name_to_tid[h_name],
        "away_team_id": name_to_tid[a_name],
        "home_team_name": h_name,
        "away_team_name": a_name,
        "home_score": h_score,
        "away_score": a_score,
        "home_penalty_score": float("nan"),
        "away_penalty_score": float("nan"),
        "home_xg": xg_h,
        "away_xg": xg_a,
        "result_type": "Regular",
        "status": "Completed",
        "venue_country": venue_country,
        "date": pd.Timestamp("2026-06-15"),
    }


def _make_ds(rows: list[dict], team_stats: pd.DataFrame | None = None) -> _FakeDataset:
    matches = pd.DataFrame(rows)
    return _FakeDataset(
        teams=_make_teams_df(),
        matches=matches,
        tournament_stages=pd.DataFrame(),
        venues=pd.DataFrame(),
        team_stats=team_stats,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeFormBasic:
    def test_returns_dict_with_bracket_ids(self):
        """compute_form returns TeamForm for teams with xG data."""
        rows = [_make_match_row(1, "Strong", "Weak", 2, 0, 1.8, 0.5)]
        ds = _make_ds(rows)
        form = compute_form(ds, _BRACKET, _MODEL)
        assert "STR" in form
        assert "WEK" in form
        assert isinstance(form["STR"], TeamForm)

    def test_played_count(self):
        """played equals number of non-null xG matches the team appeared in."""
        rows = [
            _make_match_row(1, "Strong", "Weak",   2, 0, 1.8, 0.5),
            _make_match_row(2, "Strong", "MidA",   1, 1, 1.2, 0.9),
        ]
        ds = _make_ds(rows)
        form = compute_form(ds, _BRACKET, _MODEL)
        assert form["STR"].played == 2
        assert form["WEK"].played == 1

    def test_no_xg_columns_returns_empty(self):
        """If home_xg / away_xg columns are absent, return {}."""
        rows = [{"match_id": 1, "home_team_name": "Strong", "away_team_name": "Weak",
                 "home_score": 2.0, "away_score": 0.0, "home_team_id": 101, "away_team_id": 102,
                 "status": "Completed", "venue_country": "USA",
                 "date": pd.Timestamp("2026-06-15"), "stage_id": 1, "venue_id": 1,
                 "home_penalty_score": float("nan"), "away_penalty_score": float("nan"),
                 "result_type": "Regular"}]
        ds = _make_ds(rows)  # no home_xg column
        form = compute_form(ds, _BRACKET, _MODEL)
        assert form == {}

    def test_all_nan_xg_returns_empty(self):
        """If all xG values are NaN, return {}."""
        rows = [_make_match_row(1, "Strong", "Weak", 2, 0, None, None)]
        ds = _make_ds(rows)
        form = compute_form(ds, _BRACKET, _MODEL)
        assert form == {}

    def test_unfitted_team_skipped_no_raise(self):
        """A team absent from the bracket is skipped silently."""
        name_to_tid = {t["name"]: _team_id_map()[t["id"]] for t in _FOUR_TEAMS}
        rows = [{
            "match_id": 1, "stage_id": 1, "venue_id": 1,
            "home_team_id": 999, "away_team_id": name_to_tid["Weak"],
            "home_team_name": "UnknownFC", "away_team_name": "Weak",
            "home_score": 2.0, "away_score": 0.0,
            "home_xg": 1.5, "away_xg": 0.4,
            "home_penalty_score": float("nan"), "away_penalty_score": float("nan"),
            "result_type": "Regular", "status": "Completed",
            "venue_country": "USA", "date": pd.Timestamp("2026-06-15"),
        }]
        ds = _make_ds(rows)
        form = compute_form(ds, _BRACKET, _MODEL)
        # WEK may still appear (it was in the match), but UnknownFC should not.
        assert "UnknownFC" not in form


class TestOpponentAdjustment:
    """Core requirement: same raw xG against a strong opponent → higher form than vs weak."""

    def _form_atk_for(self, xg_for: float, opponent: str) -> float:
        """Run one match of TeamAlpha vs *opponent* and return form_attack for Alpha."""
        # Use a fresh one-match-only bracket/model where only Alpha and opponent appear.
        row = _make_match_row(1, "Strong", opponent, 2, 0, xg_for, 1.0)
        ds = _make_ds([row])
        form = compute_form(ds, _BRACKET, _MODEL)
        return form.get("STR", TeamForm(0,0,0,0,0,0,0,0,None,None)).form_attack

    def test_same_xg_vs_strong_beats_vs_weak(self):
        """Identical actual xG against a strong opponent earns a higher bump than vs weak."""
        xg = 1.5
        # vs "Weak" (low defense): DC expects more goals → residual is lower
        # vs "MidA" (medium defense): DC expects fewer goals → same xG → higher residual
        row_vs_weak = _make_match_row(1, "Strong", "Weak",  2, 0, xg, 0.5)
        row_vs_mid  = _make_match_row(1, "Strong", "MidA",  2, 0, xg, 0.5)
        ds_weak = _make_ds([row_vs_weak])
        ds_mid  = _make_ds([row_vs_mid])
        form_vs_weak = compute_form(ds_weak, _BRACKET, _MODEL)
        form_vs_mid  = compute_form(ds_mid,  _BRACKET, _MODEL)
        # Against stronger opponent (MidA > Weak in defense), the residual is higher.
        # Strong's eff atk=1.0, Weak's eff def=-0.5, MidA's eff def=0.1
        # lam vs Weak  = exp(1.0 - (-0.5)) = exp(1.5) ≈ 4.48
        # lam vs MidA  = exp(1.0 -   0.1)  = exp(0.9) ≈ 2.46
        # residual vs Weak  = log((1.5+0.25)/(4.48+0.25)) < residual vs MidA
        assert form_vs_mid["STR"].form_attack > form_vs_weak["STR"].form_attack, (
            "Same xG against a stronger defense must yield a higher form_attack"
        )


class TestShrinkage:
    def test_more_matches_bigger_magnitude(self):
        """5 matches give a larger |form_attack| than 1 match (no cap hit)."""
        base_row = _make_match_row(1, "MidA", "MidB", 1, 0, 1.2, 0.5)
        rows_1 = [base_row]
        rows_5 = [{**base_row, "match_id": i} for i in range(1, 6)]
        form_1 = compute_form(_make_ds(rows_1), _BRACKET, _MODEL)
        form_5 = compute_form(_make_ds(rows_5), _BRACKET, _MODEL)
        assert abs(form_5["MDA"].form_attack) > abs(form_1["MDA"].form_attack)


class TestCapHit:
    def test_cap_is_exactly_respected(self):
        """Extreme xG residuals are clipped to ±CAP."""
        # Very high actual xG against a strong defense → should clip at +CAP.
        rows = [
            _make_match_row(i, "MidA", "Strong", 0, 0, 20.0, 0.0)
            for i in range(1, 8)
        ]
        ds = _make_ds(rows)
        form = compute_form(ds, _BRACKET, _MODEL)
        assert form["MDA"].form_attack <= CAP + 1e-9, "form_attack must not exceed CAP"
        assert form["MDA"].form_attack >= -CAP - 1e-9


class TestDefenseSign:
    def test_conceding_less_than_expected_positive_defense(self):
        """Conceding fewer goals than DC expected → positive form_defense."""
        # MidA vs Strong: DC expects Strong to score ~exp(1.0 - 0.1) ≈ 2.46 goals against MidA.
        # Actual xG against = 0.3 → below expectation → f_def > 0 → form_defense > 0.
        rows = [_make_match_row(1, "MidA", "Strong", 0, 1, 1.0, 0.3)]
        ds = _make_ds(rows)
        form = compute_form(ds, _BRACKET, _MODEL)
        assert form["MDA"].form_defense > 0.0


class TestApplyForm:
    def test_base_keys_present_for_all_teams(self):
        """apply_form adds base_attack/defense/overall to every eff entry."""
        eff = {
            "STR": {"attack": 1.0, "defense": 0.8, "overall": 1.8},
            "WEK": {"attack": -0.8, "defense": -0.5, "overall": -1.3},
        }
        form = {}
        out = apply_form(eff, form)
        for tid in ("STR", "WEK"):
            assert "base_attack"  in out[tid]
            assert "base_defense" in out[tid]
            assert "base_overall" in out[tid]

    def test_overall_equals_base_plus_bumps(self):
        """overall == base_overall + form_attack + form_defense (within rounding)."""
        eff = {
            "STR": {"attack": 1.0, "defense": 0.8, "overall": 1.8},
        }
        form = {
            "STR": TeamForm(
                played=3, gf=6.0, ga=2.0, xg_for=5.0, xg_against=1.5,
                form_attack=0.1, form_defense=0.05, form_rating=0.15,
                possession_avg=None, shots_avg=None,
            ),
        }
        out = apply_form(eff, form)
        e = out["STR"]
        expected_overall = e["base_overall"] + form["STR"].form_attack + form["STR"].form_defense
        assert e["overall"] == pytest.approx(expected_overall, abs=1e-9)

    def test_passthrough_team_zero_bumps(self):
        """Team absent from form gets zero bump; attack/defense/overall unchanged."""
        eff = {"WEK": {"attack": -0.8, "defense": -0.5, "overall": -1.3}}
        out = apply_form(eff, form={})
        e = out["WEK"]
        assert e["attack"]  == pytest.approx(-0.8)
        assert e["defense"] == pytest.approx(-0.5)
        assert e["overall"] == pytest.approx(-1.3)

    def test_does_not_mutate_original_eff(self):
        """apply_form must return a new dict and not modify the input."""
        eff = {"STR": {"attack": 1.0, "defense": 0.8, "overall": 1.8}}
        form = {}
        original_id = id(eff["STR"])
        out = apply_form(eff, form)
        assert id(out["STR"]) != original_id, "apply_form must return new entry dicts"


class TestTeamStatsIntegration:
    def test_no_team_stats_gives_none_possession(self):
        """When ds.team_stats is None, possession_avg and shots_avg are None."""
        rows = [_make_match_row(1, "Strong", "Weak", 2, 0, 1.8, 0.5)]
        ds = _make_ds(rows, team_stats=None)
        form = compute_form(ds, _BRACKET, _MODEL)
        assert form["STR"].possession_avg is None
        assert form["STR"].shots_avg is None

    def test_team_stats_populates_possession(self):
        """When ds.team_stats has data, possession_avg is non-None."""
        rows = [_make_match_row(1, "Strong", "Weak", 2, 0, 1.8, 0.5)]
        tid_map = _team_id_map()
        ts = pd.DataFrame([
            {"match_id": 1, "team_id": tid_map["STR"], "possession_pct": 60.0, "total_shots": 14},
        ])
        ds = _make_ds(rows, team_stats=ts)
        form = compute_form(ds, _BRACKET, _MODEL)
        assert form["STR"].possession_avg is not None
        assert abs(form["STR"].possession_avg - 60.0) < 1e-6


class TestTripwire:
    def test_evaluate_has_no_apply_form_reference(self):
        """evaluate.py must never import apply_form (would leak form into backtests)."""
        import importlib.util, pathlib
        evaluate_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "fifa_pred" / "evaluate.py"
        src = evaluate_path.read_text()
        assert "apply_form" not in src, (
            "evaluate.py references apply_form — this would contaminate backtest calibration"
        )
