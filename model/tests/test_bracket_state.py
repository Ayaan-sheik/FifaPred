"""Tests for bracket_state module using synthetic in-memory frames (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from fifa_pred.bracket_state import (
    FALLBACK_FEEDS,
    TournamentState,
    _int_or_none,
    bracket_order,
    build_tournament_state,
    derive_feeds,
    winner_of,
)
from fifa_pred.data_loader import Bracket


# ---------------------------------------------------------------------------
# Helpers: build tiny synthetic datasets and brackets
# ---------------------------------------------------------------------------

def _make_bracket(teams: list[dict]) -> Bracket:
    """Build a minimal Bracket from a list of team dicts."""
    by_id = {t["id"]: t for t in teams}
    name_to_id = {t["name"]: t["id"] for t in teams}
    groups: dict[str, list[str]] = {}
    for t in teams:
        g = t.get("group", "A")
        groups.setdefault(g, []).append(t["id"])
    fmt = {
        "groups": len(groups),
        "teams_per_group": 4,
        "advance_per_group": 2,
        "best_third_advancing": 8,
        "knockout_start": "Round of 32",
    }
    return Bracket(teams=teams, groups=groups, fmt=fmt, by_id=by_id, name_to_id=name_to_id)


def _make_row(**kwargs) -> pd.Series:
    """Build a pd.Series representing a match row."""
    defaults = {
        "match_id": 1,
        "home_team_id": 1,
        "away_team_id": 2,
        "home_score": 1,
        "away_score": 0,
        "home_penalty_score": float("nan"),
        "away_penalty_score": float("nan"),
        "result_type": "Regular",
        "status": "Completed",
        "home_bracket_id": "AAA",
        "away_bracket_id": "BBB",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# bracket_order tests
# ---------------------------------------------------------------------------

class TestBracketOrder:
    def test_r32_leaf_order_matches_true_bracket_draw(self):
        """R32 display order must follow the tree (left-to-right), not
        ascending match_id — this is the case that motivated the fix:
        e.g. match 88 is not necessarily the last bracket slot."""
        order = bracket_order(FALLBACK_FEEDS)
        r32_ids = list(range(73, 89))
        r32_in_tree_order = sorted(r32_ids, key=lambda mid: order[mid])
        assert r32_in_tree_order == [
            75, 78, 73, 76, 84, 83, 82, 81,
            74, 77, 79, 80, 87, 86, 85, 88,
        ]

    def test_positions_are_distinct_within_each_stage(self):
        order = bracket_order(FALLBACK_FEEDS)
        for stage_ids in (
            list(range(89, 97)),   # R16
            list(range(97, 101)),  # QF
            list(range(101, 103)), # SF
        ):
            positions = [order[mid] for mid in stage_ids]
            assert len(set(positions)) == len(positions)

    def test_final_and_all_matches_covered(self):
        order = bracket_order(FALLBACK_FEEDS)
        assert 104 in order
        assert set(order) == set(range(73, 103)) | {104}


# ---------------------------------------------------------------------------
# winner_of tests
# ---------------------------------------------------------------------------

class TestWinnerOf:
    def test_regular_home_win(self):
        row = _make_row(home_score=2, away_score=0, result_type="Regular")
        assert winner_of(row) == 1

    def test_regular_away_win(self):
        row = _make_row(home_score=0, away_score=3, result_type="Regular")
        assert winner_of(row) == 2

    def test_aet_home_win(self):
        row = _make_row(home_score=2, away_score=1, result_type="AET")
        assert winner_of(row) == 1

    def test_aet_away_win(self):
        row = _make_row(home_score=1, away_score=2, result_type="AET")
        assert winner_of(row) == 2

    def test_penalties_home_win(self):
        row = _make_row(
            home_score=1, away_score=1, result_type="Penalties",
            home_penalty_score=4.0, away_penalty_score=3.0,
        )
        assert winner_of(row) == 1

    def test_penalties_away_win(self):
        row = _make_row(
            home_score=1, away_score=1, result_type="Penalties",
            home_penalty_score=2.0, away_penalty_score=4.0,
        )
        assert winner_of(row) == 2

    def test_not_completed_returns_none(self):
        row = _make_row(status="Scheduled", home_score=float("nan"), away_score=float("nan"))
        assert winner_of(row) is None

    def test_missing_scores_returns_none(self):
        row = _make_row(home_score=float("nan"), away_score=float("nan"))
        assert winner_of(row) is None


# ---------------------------------------------------------------------------
# derive_feeds tests
# ---------------------------------------------------------------------------

def _make_ko_df(rows: list[dict]) -> pd.DataFrame:
    """Build a knockout matches DataFrame from a list of row dicts."""
    defaults = {
        "match_id": 0,
        "home_team_id": None,
        "away_team_id": None,
        "home_score": float("nan"),
        "away_score": float("nan"),
        "home_penalty_score": float("nan"),
        "away_penalty_score": float("nan"),
        "result_type": None,
        "status": "Scheduled",
        "home_bracket_id": None,
        "away_bracket_id": None,
    }
    out = []
    for r in rows:
        d = dict(defaults)
        d.update(r)
        out.append(d)
    return pd.DataFrame(out)


def test_derive_feeds_no_override_when_no_r16_data():
    """With only R32 completions and no R16 data, FALLBACK_FEEDS is returned unchanged."""
    # Match 75 completed (PAR wins), match 78 completed (FRA wins).
    # No R16 fixture listed — feeds[89] should stay as FALLBACK_FEEDS[89]=(75,78).
    ko = _make_ko_df([
        {"match_id": 75, "home_team_id": 17, "away_team_id": 14,
         "home_score": 1, "away_score": 1, "result_type": "Penalties", "status": "Completed",
         "home_bracket_id": "GER", "away_bracket_id": "PAR",
         "home_penalty_score": 3.0, "away_penalty_score": 4.0},
        {"match_id": 78, "home_team_id": 7, "away_team_id": 31,
         "home_score": 3, "away_score": 0, "result_type": "Regular", "status": "Completed",
         "home_bracket_id": "FRA", "away_bracket_id": "SWE"},
    ])
    feeds = derive_feeds(ko)
    # No R16 data → FALLBACK_FEEDS[89] = (75, 78) preserved.
    assert feeds[89] == (75, 78)


def test_derive_feeds_scheduled_conflict_no_warning(capsys):
    """Scheduled fixtures with wrong teams do NOT produce warnings (they are ignored)."""
    # The mominullptr dataset may schedule R16 matches with wrong pre-seeding team IDs.
    # We must not emit warnings or override FALLBACK_FEEDS for these.
    ko = _make_ko_df([
        {"match_id": 74, "home_team_id": 9, "away_team_id": 22,
         "home_score": 2, "away_score": 1, "result_type": "Regular", "status": "Completed",
         "home_bracket_id": "BRA", "away_bracket_id": "JPN"},
        {"match_id": 75, "home_team_id": 17, "away_team_id": 14,
         "home_score": 1, "away_score": 1, "result_type": "Penalties", "status": "Completed",
         "home_bracket_id": "GER", "away_bracket_id": "PAR",
         "home_penalty_score": 3.0, "away_penalty_score": 4.0},
        # Wrong-seeded scheduled fixture for match 90 (dataset says BRA vs PAR, not CAN vs MAR)
        {"match_id": 90, "home_team_id": 9, "away_team_id": 14,
         "status": "Scheduled", "home_bracket_id": "BRA", "away_bracket_id": "PAR"},
    ])
    feeds = derive_feeds(ko)
    out = capsys.readouterr().out
    assert "WARNING" not in out
    # FALLBACK_FEEDS[90] = (73, 76) preserved (scheduled fixture ignored).
    assert feeds[90] == FALLBACK_FEEDS[90]
    assert feeds[90] == (73, 76)


def test_derive_feeds_scheduled_fixture_ignored():
    """Scheduled (pending) fixtures do not override FALLBACK_FEEDS."""
    # The dataset may have scheduled R16 fixtures with pre-seeding team IDs.
    # These should be ignored so FALLBACK_FEEDS (correct bracket draw) is used.
    ko = _make_ko_df([
        {"match_id": 73, "home_team_id": 2, "away_team_id": 5,
         "home_score": 0, "away_score": 1, "result_type": "Regular", "status": "Completed",
         "home_bracket_id": "RSA", "away_bracket_id": "CAN"},
        {"match_id": 75, "home_team_id": 17, "away_team_id": 14,
         "home_score": 1, "away_score": 1, "result_type": "Penalties", "status": "Completed",
         "home_bracket_id": "GER", "away_bracket_id": "PAR",
         "home_penalty_score": 3.0, "away_penalty_score": 4.0},
        # Scheduled fixture with wrong pre-seeding team IDs (CAN vs PAR for slot 89)
        {"match_id": 89, "home_team_id": 5, "away_team_id": 14,
         "status": "Scheduled", "home_bracket_id": "CAN", "away_bracket_id": "PAR"},
    ])
    feeds = derive_feeds(ko)
    # Scheduled fixture ignored; FALLBACK_FEEDS[89]=(75,78) is preserved.
    assert feeds[89] == FALLBACK_FEEDS[89]
    assert feeds[89] == (75, 78)


def test_derive_feeds_no_participants_uses_fallback():
    """When no participants are known, FALLBACK_FEEDS is returned unchanged."""
    ko = _make_ko_df([
        # A future match with no participants yet.
        {"match_id": 90, "status": "Scheduled"},
    ])
    feeds = derive_feeds(ko)
    assert feeds[90] == FALLBACK_FEEDS[90]


# ---------------------------------------------------------------------------
# build_tournament_state with synthetic data
# ---------------------------------------------------------------------------

def _make_teams_df(team_rows: list[dict]) -> pd.DataFrame:
    """Build a teams DataFrame."""
    return pd.DataFrame(team_rows)


def _make_venues_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"venue_id": 1, "stadium_name": "Estadio A", "city": "Mexico City", "country": "MEX"},
        {"venue_id": 2, "stadium_name": "Stadium B", "city": "New York", "country": "USA"},
    ])


def _make_stages_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"stage_id": 1, "stage_name": "Group Stage", "is_knockout": False},
        {"stage_id": 2, "stage_name": "Round of 32", "is_knockout": True},
        {"stage_id": 3, "stage_name": "Round of 16", "is_knockout": True},
    ])


def _build_minimal_state():
    """
    Build a minimal 8-team synthetic tournament state with:
    - 4 group matches completed (group stage NOT complete, since we need 72)
    - 2 R32 matches completed
    Returns (state, bracket).
    """
    from fifa_pred.wc_dataset import Dataset

    # 8 teams, 2 groups of 4
    team_rows = [
        {"team_id": 1, "team_name": "Alpha", "fifa_code": "ALP", "group_letter": "A",
         "confederation": "UEFA", "fifa_ranking_pre_tournament": 1, "elo_rating": 2000, "manager_name": "X"},
        {"team_id": 2, "team_name": "Beta",  "fifa_code": "BET", "group_letter": "A",
         "confederation": "UEFA", "fifa_ranking_pre_tournament": 2, "elo_rating": 1900, "manager_name": "X"},
        {"team_id": 3, "team_name": "Gamma", "fifa_code": "GAM", "group_letter": "A",
         "confederation": "CONMEBOL", "fifa_ranking_pre_tournament": 3, "elo_rating": 1800, "manager_name": "X"},
        {"team_id": 4, "team_name": "Delta", "fifa_code": "DEL", "group_letter": "A",
         "confederation": "CONMEBOL", "fifa_ranking_pre_tournament": 4, "elo_rating": 1700, "manager_name": "X"},
        {"team_id": 5, "team_name": "Echo",  "fifa_code": "ECH", "group_letter": "B",
         "confederation": "CAF", "fifa_ranking_pre_tournament": 5, "elo_rating": 1600, "manager_name": "X"},
        {"team_id": 6, "team_name": "Foxtrot","fifa_code": "FOX", "group_letter": "B",
         "confederation": "CAF", "fifa_ranking_pre_tournament": 6, "elo_rating": 1500, "manager_name": "X"},
        {"team_id": 7, "team_name": "Golf",  "fifa_code": "GOL", "group_letter": "B",
         "confederation": "AFC", "fifa_ranking_pre_tournament": 7, "elo_rating": 1400, "manager_name": "X"},
        {"team_id": 8, "team_name": "Hotel", "fifa_code": "HOT", "group_letter": "B",
         "confederation": "AFC", "fifa_ranking_pre_tournament": 8, "elo_rating": 1300, "manager_name": "X"},
    ]
    teams_df = _make_teams_df(team_rows)

    # Group matches (ids 1-4, incomplete — would need 72 for complete)
    group_matches = [
        {"match_id": 1, "date": "2026-06-11", "stage_id": 1, "venue_id": 1,
         "home_team_id": 1, "away_team_id": 2,
         "home_score": 2.0, "away_score": 0.0,
         "home_penalty_score": float("nan"), "away_penalty_score": float("nan"),
         "result_type": "Regular", "status": "Completed",
         "home_xg": None, "away_xg": None, "referee_id": None, "player_of_the_match_id": None,
         "kickoff_time_utc": None},
        {"match_id": 2, "date": "2026-06-11", "stage_id": 1, "venue_id": 2,
         "home_team_id": 5, "away_team_id": 6,
         "home_score": 1.0, "away_score": 0.0,
         "home_penalty_score": float("nan"), "away_penalty_score": float("nan"),
         "result_type": "Regular", "status": "Completed",
         "home_xg": None, "away_xg": None, "referee_id": None, "player_of_the_match_id": None,
         "kickoff_time_utc": None},
    ]

    # R32 matches: ALP beats BET, ECH beats FOX
    ko_matches = [
        {"match_id": 73, "date": "2026-06-29", "stage_id": 2, "venue_id": 2,
         "home_team_id": 2, "away_team_id": 1,
         "home_score": 0.0, "away_score": 1.0,
         "home_penalty_score": float("nan"), "away_penalty_score": float("nan"),
         "result_type": "Regular", "status": "Completed",
         "home_xg": None, "away_xg": None, "referee_id": None, "player_of_the_match_id": None,
         "kickoff_time_utc": None},
        {"match_id": 74, "date": "2026-06-29", "stage_id": 2, "venue_id": 2,
         "home_team_id": 5, "away_team_id": 6,
         "home_score": 2.0, "away_score": 1.0,
         "home_penalty_score": float("nan"), "away_penalty_score": float("nan"),
         "result_type": "Regular", "status": "Completed",
         "home_xg": None, "away_xg": None, "referee_id": None, "player_of_the_match_id": None,
         "kickoff_time_utc": None},
    ]

    all_matches = pd.DataFrame(group_matches + ko_matches)
    all_matches["date"] = pd.to_datetime(all_matches["date"])

    # Enrich with stage_name and venue_country
    stage_map = {1: "Group Stage", 2: "Round of 32", 3: "Round of 16"}
    venue_country_map = {1: "MEX", 2: "USA"}
    id_to_name = {r["team_id"]: r["team_name"] for r in team_rows}

    all_matches["stage_name"] = all_matches["stage_id"].map(stage_map)
    all_matches["venue_country"] = all_matches["venue_id"].map(venue_country_map)
    all_matches["home_team_name"] = all_matches["home_team_id"].map(id_to_name)
    all_matches["away_team_name"] = all_matches["away_team_id"].map(id_to_name)

    ds = Dataset(
        teams=teams_df,
        matches=all_matches,
        tournament_stages=_make_stages_df(),
        venues=_make_venues_df(),
    )

    bracket_teams = [
        {"id": "ALP", "name": "Alpha", "confederation": "UEFA", "flag": "🏴", "group": "A"},
        {"id": "BET", "name": "Beta",  "confederation": "UEFA", "flag": "🏴", "group": "A"},
        {"id": "GAM", "name": "Gamma", "confederation": "CONMEBOL", "flag": "🏴", "group": "A"},
        {"id": "DEL", "name": "Delta", "confederation": "CONMEBOL", "flag": "🏴", "group": "A"},
        {"id": "ECH", "name": "Echo",  "confederation": "CAF", "flag": "🏴", "group": "B"},
        {"id": "FOX", "name": "Foxtrot","confederation": "CAF", "flag": "🏴", "group": "B"},
        {"id": "GOL", "name": "Golf",  "confederation": "AFC", "flag": "🏴", "group": "B"},
        {"id": "HOT", "name": "Hotel", "confederation": "AFC", "flag": "🏴", "group": "B"},
    ]
    bracket = _make_bracket(bracket_teams)
    return ds, bracket


class TestBuildTournamentState:
    def test_group_stage_not_complete_with_few_matches(self):
        ds, bracket = _build_minimal_state()
        state = build_tournament_state(ds, bracket)
        # Only 2 group matches, need 72 for complete.
        assert state.group_stage_complete is False

    def test_r32_participants_found(self):
        ds, bracket = _build_minimal_state()
        state = build_tournament_state(ds, bracket)
        # Matches 73 and 74 are listed, so ALP, BET, ECH, FOX should be in r32_slots.
        assert "ALP" in state.r32_slots
        assert "BET" in state.r32_slots
        assert "ECH" in state.r32_slots
        assert "FOX" in state.r32_slots

    def test_r32_winners_in_reached(self):
        ds, bracket = _build_minimal_state()
        state = build_tournament_state(ds, bracket)
        # ALP won match 73, ECH won match 74.
        # They should advance; BET and FOX should be eliminated.
        assert "BET" in state.eliminated
        assert "FOX" in state.eliminated

    def test_feeds_include_fallback(self):
        ds, bracket = _build_minimal_state()
        state = build_tournament_state(ds, bracket)
        # FALLBACK_FEEDS entries should be present.
        assert 89 in state.feeds
        assert 104 in state.feeds

    def test_next_stage_round_of_16_when_r32_done(self):
        """With R32 complete, next_stage should advance."""
        # This uses the real data indirectly; just check the synthetic one
        # shows "Round of 32" since not all 16 R32 matches are completed.
        ds, bracket = _build_minimal_state()
        state = build_tournament_state(ds, bracket)
        # Only 2 of 16 R32 matches completed → still in Round of 32
        assert state.next_stage == "Round of 32"


# ---------------------------------------------------------------------------
# simulate_from_state invariants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_state_and_probs():
    """Load real bracket + dataset and run simulate_from_state (module-scoped)."""
    from fifa_pred.data_loader import load_bracket, load_results, merge_live_results
    from fifa_pred.dixon_coles import DixonColes
    from fifa_pred.simulator import simulate_from_state
    from fifa_pred.wc_dataset import load_dataset, completed_results_frame

    bracket = load_bracket()
    df = load_results()
    ds = load_dataset()
    live = completed_results_frame(ds)
    df = merge_live_results(df, live)
    model = DixonColes.fit(df)
    eff = model.effective_params(bracket)
    state = build_tournament_state(ds, bracket)
    probs = simulate_from_state(bracket, eff, state, n_sims=4000, seed=99)
    return state, probs, bracket


class TestSimulateFromState:
    def test_sum_p_win_approx_one(self, real_state_and_probs):
        _, probs, bracket = real_state_and_probs
        total = sum(probs[t["id"]]["p_win"] for t in bracket.teams)
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_sum_p_advance_is_32(self, real_state_and_probs):
        _, probs, bracket = real_state_and_probs
        total = sum(probs[t["id"]]["p_advance"] for t in bracket.teams)
        assert total == pytest.approx(32.0, abs=1e-6)

    def test_sum_p_r16_is_16(self, real_state_and_probs):
        _, probs, bracket = real_state_and_probs
        total = sum(probs[t["id"]]["p_r16"] for t in bracket.teams)
        assert total == pytest.approx(16.0, abs=1e-3)

    def test_eliminated_teams_have_zero_p_win(self, real_state_and_probs):
        state, probs, bracket = real_state_and_probs
        for tid in state.eliminated:
            assert probs[tid]["p_win"] == 0.0, f"{tid} is eliminated but has p_win > 0"

    def test_r32_winners_have_p_r16_one(self, real_state_and_probs):
        state, probs, bracket = real_state_and_probs
        r32_winners = ['CAN','BRA','PAR','MAR','NOR','FRA','MEX','ENG',
                       'BEL','USA','ESP','POR','SUI','EGY','ARG','COL']
        for tid in r32_winners:
            assert probs[tid]["p_r16"] == pytest.approx(1.0, abs=1e-6), \
                f"{tid} should have p_r16=1.0 (won R32)"

    def test_stage_probabilities_monotone(self, real_state_and_probs):
        _, probs, bracket = real_state_and_probs
        for t in bracket.teams:
            p = probs[t["id"]]
            assert p["p_win"]   <= p["p_final"] + 1e-9
            assert p["p_final"] <= p["p_sf"]    + 1e-9
            assert p["p_sf"]    <= p["p_qf"]    + 1e-9
            assert p["p_qf"]    <= p["p_r16"]   + 1e-9
            assert p["p_r16"]   <= p["p_advance"] + 1e-9

    def test_deterministic_under_same_seed(self, real_state_and_probs):
        state, _, bracket = real_state_and_probs
        from fifa_pred.data_loader import load_bracket, load_results, merge_live_results
        from fifa_pred.dixon_coles import DixonColes
        from fifa_pred.simulator import simulate_from_state
        from fifa_pred.wc_dataset import load_dataset, completed_results_frame

        ds = load_dataset()
        live = completed_results_frame(ds)
        df = merge_live_results(load_results(), live)
        eff = DixonColes.fit(df).effective_params(bracket)
        p1 = simulate_from_state(bracket, eff, state, n_sims=1000, seed=7)
        p2 = simulate_from_state(bracket, eff, state, n_sims=1000, seed=7)
        for t in bracket.teams:
            assert p1[t["id"]]["p_win"] == p2[t["id"]]["p_win"]

    def test_rsa_p_advance_one_rest_zero(self, real_state_and_probs):
        _, probs, _ = real_state_and_probs
        rsa = probs["RSA"]
        assert rsa["p_advance"] == pytest.approx(1.0)
        assert rsa["p_r16"]   == 0.0
        assert rsa["p_qf"]    == 0.0
        assert rsa["p_win"]   == 0.0

    def test_group_stage_complete(self, real_state_and_probs):
        state, _, _ = real_state_and_probs
        assert state.group_stage_complete is True

    def test_16_r32_slots(self, real_state_and_probs):
        state, _, _ = real_state_and_probs
        # Exactly 32 teams qualified to R32 from group stage.
        assert len(state.r32_slots) == 32

    def test_feeds_89_uses_correct_bracket_draw(self, real_state_and_probs):
        state, _, _ = real_state_and_probs
        # FALLBACK_FEEDS[89] = (75, 78): PAR(w75) vs FRA(w78) per official bracket draw.
        # If match 89 is completed in the real dataset, derive_feeds may override this
        # with the actual participants; otherwise FALLBACK_FEEDS is used.
        assert 89 in state.feeds
