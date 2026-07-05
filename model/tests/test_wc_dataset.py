"""Tests for wc_dataset module using synthetic in-memory frames (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from fifa_pred.wc_dataset import (
    DATASET_NAME_MAP,
    Dataset,
    completed_results_frame,
    unmapped_team_names,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic Dataset
# ---------------------------------------------------------------------------

def _make_dataset(
    match_rows: list[dict],
    team_rows: list[dict] | None = None,
    venue_rows: list[dict] | None = None,
) -> Dataset:
    """Build a synthetic Dataset for testing."""
    if team_rows is None:
        team_rows = [
            {"team_id": 1, "team_name": "Mexico", "fifa_code": "MEX",
             "group_letter": "A", "confederation": "CONCACAF",
             "fifa_ranking_pre_tournament": 15, "elo_rating": 1650, "manager_name": "X"},
            {"team_id": 2, "team_name": "South Africa", "fifa_code": "RSA",
             "group_letter": "A", "confederation": "CAF",
             "fifa_ranking_pre_tournament": 60, "elo_rating": 1400, "manager_name": "X"},
            {"team_id": 3, "team_name": "United States", "fifa_code": "USA",
             "group_letter": "D", "confederation": "CONCACAF",
             "fifa_ranking_pre_tournament": 14, "elo_rating": 1660, "manager_name": "X"},
            {"team_id": 4, "team_name": "Canada", "fifa_code": "CAN",
             "group_letter": "B", "confederation": "CONCACAF",
             "fifa_ranking_pre_tournament": 47, "elo_rating": 1530, "manager_name": "X"},
            {"team_id": 5, "team_name": "France", "fifa_code": "FRA",
             "group_letter": "I", "confederation": "UEFA",
             "fifa_ranking_pre_tournament": 2, "elo_rating": 1990, "manager_name": "X"},
            {"team_id": 6, "team_name": "Argentina", "fifa_code": "ARG",
             "group_letter": "J", "confederation": "CONMEBOL",
             "fifa_ranking_pre_tournament": 1, "elo_rating": 2050, "manager_name": "X"},
        ]

    if venue_rows is None:
        venue_rows = [
            {"venue_id": 1, "stadium_name": "Estadio Azteca", "city": "Mexico City",
             "country": "MEX", "capacity": 87000,
             "latitude": 19.3, "longitude": -99.15, "elevation_meters": 2240},
            {"venue_id": 2, "stadium_name": "MetLife Stadium", "city": "East Rutherford",
             "country": "USA", "capacity": 82500,
             "latitude": 40.8, "longitude": -74.07, "elevation_meters": 0},
            {"venue_id": 3, "stadium_name": "BMO Field", "city": "Toronto",
             "country": "CAN", "capacity": 30000,
             "latitude": 43.6, "longitude": -79.4, "elevation_meters": 76},
        ]

    teams_df = pd.DataFrame(team_rows)
    venues_df = pd.DataFrame(venue_rows)
    stages_df = pd.DataFrame([
        {"stage_id": 1, "stage_name": "Group Stage", "is_knockout": False},
        {"stage_id": 2, "stage_name": "Round of 32", "is_knockout": True},
    ])

    id_to_name = {r["team_id"]: r["team_name"] for r in team_rows}
    stage_map = {1: "Group Stage", 2: "Round of 32"}
    venue_country_map = {r["venue_id"]: r["country"] for r in venue_rows}

    matches_df = pd.DataFrame(match_rows)
    if not matches_df.empty:
        matches_df["date"] = pd.to_datetime(matches_df["date"])
        matches_df["stage_name"] = matches_df["stage_id"].map(stage_map)
        matches_df["venue_country"] = matches_df["venue_id"].map(venue_country_map)
        matches_df["home_team_name"] = matches_df["home_team_id"].map(id_to_name)
        matches_df["away_team_name"] = matches_df["away_team_id"].map(id_to_name)

    return Dataset(
        teams=teams_df,
        matches=matches_df,
        tournament_stages=stages_df,
        venues=venues_df,
    )


# ---------------------------------------------------------------------------
# DATASET_NAME_MAP: verify the 7 canonical mappings
# ---------------------------------------------------------------------------

class TestDatasetNameMap:
    EXPECTED_MAPPINGS = {
        "Czechia":        "Czech Republic",
        "USA":            "United States",
        "Türkiye":        "Turkey",
        "Côte d'Ivoire":  "Ivory Coast",
        "Cabo Verde":     "Cape Verde",
        "Congo DR":       "DR Congo",
        "IR Iran":        "Iran",
    }

    def test_all_7_mappings_present(self):
        assert len(DATASET_NAME_MAP) == len(self.EXPECTED_MAPPINGS)

    @pytest.mark.parametrize("source,canonical", EXPECTED_MAPPINGS.items())
    def test_mapping(self, source, canonical):
        assert DATASET_NAME_MAP[source] == canonical


# ---------------------------------------------------------------------------
# completed_results_frame: schema and neutral logic
# ---------------------------------------------------------------------------

class TestCompletedResultsFrame:
    """Test schema, neutral logic, and away-host swap using synthetic frames."""

    def _match_row(self, match_id, home_id, away_id, venue_id,
                   home_score, away_score, result_type="Regular",
                   status="Completed"):
        return {
            "match_id": match_id,
            "date": "2026-06-11",
            "stage_id": 1,
            "venue_id": venue_id,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_score": float(home_score),
            "away_score": float(away_score),
            "home_penalty_score": float("nan"),
            "away_penalty_score": float("nan"),
            "result_type": result_type,
            "status": status,
            "home_xg": None, "away_xg": None,
            "referee_id": None, "player_of_the_match_id": None,
            "kickoff_time_utc": None,
        }

    def test_output_schema(self):
        """completed_results_frame should return the results.csv schema."""
        ds = _make_dataset([self._match_row(1, 1, 2, 1, 2, 0)])
        result = completed_results_frame(ds)
        required_cols = {"date", "home_team", "away_team", "home_score",
                         "away_score", "tournament", "neutral"}
        assert required_cols.issubset(set(result.columns))

    def test_host_home_neutral_false(self):
        """Home host playing at own-country venue → neutral=False."""
        # MEX (team_id=1) at MEX venue (venue_id=1) vs RSA (team_id=2).
        ds = _make_dataset([self._match_row(1, 1, 2, 1, 2, 0)])
        result = completed_results_frame(ds)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["home_team"] == "Mexico"
        assert row["away_team"] == "South Africa"
        assert bool(row["neutral"]) is False

    def test_away_host_swap(self):
        """Away host (USA, team_id=3) playing at own-country venue → swap to home, neutral=False."""
        # France (team_id=5) is home, USA (team_id=3) is away, at USA venue (venue_id=2).
        ds = _make_dataset([self._match_row(1, 5, 3, 2, 1, 0)])
        result = completed_results_frame(ds)
        assert len(result) == 1
        row = result.iloc[0]
        # After swap: USA should be home, France should be away.
        assert row["home_team"] == "United States"
        assert row["away_team"] == "France"
        assert bool(row["neutral"]) is False
        # Scores should also be swapped.
        assert row["home_score"] == 0  # was away_score=0
        assert row["away_score"] == 1  # was home_score=1

    def test_neutral_venue(self):
        """Two non-host teams at a non-home venue → neutral=True."""
        # France (team_id=5) vs Argentina (team_id=6) at MEX venue.
        ds = _make_dataset([self._match_row(1, 5, 6, 1, 2, 1)])
        result = completed_results_frame(ds)
        assert len(result) == 1
        assert bool(result.iloc[0]["neutral"]) is True

    def test_penalties_not_added_to_scores(self):
        """Penalty goals must NOT be added to the regulation scores."""
        # Germany (synthetic) vs Paraguay: 1-1 AET, pen 3-4.
        # We reuse team ids 5 and 4 (France, Canada) to avoid adding new rows.
        extra_row = self._match_row(2, 5, 4, 1, 1, 1, result_type="Penalties")
        extra_row["home_penalty_score"] = 3.0
        extra_row["away_penalty_score"] = 4.0
        ds = _make_dataset([extra_row])
        result = completed_results_frame(ds)
        row = result.iloc[0]
        # Scores should remain 1-1 (no penalty goals added).
        assert row["home_score"] == 1
        assert row["away_score"] == 1

    def test_only_completed_matches(self):
        """Scheduled (incomplete) matches should be excluded."""
        completed = self._match_row(1, 1, 2, 1, 2, 0, status="Completed")
        scheduled = self._match_row(2, 3, 4, 2, 0, 0, status="Scheduled")
        scheduled["home_score"] = float("nan")
        scheduled["away_score"] = float("nan")
        ds = _make_dataset([completed, scheduled])
        result = completed_results_frame(ds)
        assert len(result) == 1

    def test_tournament_column_value(self):
        ds = _make_dataset([self._match_row(1, 1, 2, 1, 2, 0)])
        result = completed_results_frame(ds)
        assert result.iloc[0]["tournament"] == "FIFA World Cup"

    def test_scores_are_integers(self):
        ds = _make_dataset([self._match_row(1, 1, 2, 1, 2, 0)])
        result = completed_results_frame(ds)
        assert result["home_score"].dtype in (int, "int64", "int32")
        assert result["away_score"].dtype in (int, "int64", "int32")


# ---------------------------------------------------------------------------
# Neutral-column parsing regression: load_live_results must parse, not force
# ---------------------------------------------------------------------------

class TestLoadLiveResultsNeutralParsing:
    """Regression test: data_loader.load_live_results() parses neutral from CSV."""

    def test_neutral_false_preserved(self, tmp_path):
        """A row with neutral=False written to CSV should be read back as False."""
        from fifa_pred.data_loader import load_live_results

        csv_path = tmp_path / "test_live.csv"
        df = pd.DataFrame([{
            "date": "2026-06-11",
            "home_team": "Mexico",
            "away_team": "South Africa",
            "home_score": 2,
            "away_score": 0,
            "tournament": "FIFA World Cup",
            "neutral": False,
        }])
        df.to_csv(csv_path, index=False)

        loaded = load_live_results(path=csv_path)
        assert len(loaded) == 1
        assert bool(loaded.iloc[0]["neutral"]) is False

    def test_neutral_true_preserved(self, tmp_path):
        """A row with neutral=True written to CSV should be read back as True."""
        from fifa_pred.data_loader import load_live_results

        csv_path = tmp_path / "test_live.csv"
        df = pd.DataFrame([{
            "date": "2026-06-11",
            "home_team": "France",
            "away_team": "Argentina",
            "home_score": 1,
            "away_score": 2,
            "tournament": "FIFA World Cup",
            "neutral": True,
        }])
        df.to_csv(csv_path, index=False)

        loaded = load_live_results(path=csv_path)
        assert len(loaded) == 1
        assert bool(loaded.iloc[0]["neutral"]) is True


# ---------------------------------------------------------------------------
# unmapped_team_names
# ---------------------------------------------------------------------------

class TestUnmappedTeamNames:
    def test_all_canonical_names_resolve(self):
        """All team names after applying DATASET_NAME_MAP should appear in the bracket."""
        from fifa_pred.data_loader import load_bracket
        from fifa_pred.wc_dataset import load_dataset

        ds = load_dataset()
        bracket = load_bracket()
        unmapped = unmapped_team_names(ds, bracket)
        assert unmapped == [], f"Unmapped names found: {unmapped}"

    def test_returns_empty_when_bracket_is_none(self):
        ds = _make_dataset([])
        result = unmapped_team_names(ds, bracket=None)
        assert result == []
