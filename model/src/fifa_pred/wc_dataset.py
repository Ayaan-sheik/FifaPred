"""Fetch and cache the mominullptr WC2026 dataset from GitHub.

Source: https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset

Run standalone to print a summary report:

    uv run python -m fifa_pred.wc_dataset

Honors the ``FIFA_PRED_FORCE_FETCH`` environment variable (any non-empty value)
to bypass the 6-hour mtime cache and re-download all files.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import NamedTuple

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "wc2026_dataset"
LIVE_CSV = DATA_DIR / "wc2026_live.csv"

DATASET_BASE = "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/"
DATASET_FILES = ["teams.csv", "matches.csv", "tournament_stages.csv", "venues.csv"]

# Optional files: downloaded best-effort; failures are warned, not raised.
OPTIONAL_DATASET_FILES = ["match_team_stats.csv"]

# Map dataset team names → canonical names used in results.csv / wc2026.json.
# Keys are the exact strings that appear in the dataset's `team_name` column.
DATASET_NAME_MAP: dict[str, str] = {
    "Czechia": "Czech Republic",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "IR Iran": "Iran",
}

# FIFA codes of the three co-host nations — used for neutral-field logic.
_HOST_CODES = {"MEX", "USA", "CAN"}


class Dataset(NamedTuple):
    """Parsed and joined WC2026 dataset tables."""

    teams: pd.DataFrame           # team_id, team_name (canonical), fifa_code, …
    matches: pd.DataFrame         # match_id + joined stage_name + venue_country + home/away names
    tournament_stages: pd.DataFrame
    venues: pd.DataFrame
    team_stats: pd.DataFrame | None = None  # match_team_stats.csv; None if unavailable


def download_dataset(force: bool = False, max_age_hours: float = 6.0) -> None:
    """Download the four CSV files into ``CACHE_DIR``.

    Skips any file whose cached mtime is within *max_age_hours*.  On network
    failure, falls back silently to the existing cached copy (if present).
    ``FIFA_PRED_FORCE_FETCH`` env var overrides *force*.
    """
    if os.environ.get("FIFA_PRED_FORCE_FETCH"):
        force = True

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    max_age_sec = max_age_hours * 3600.0

    for fname in DATASET_FILES:
        dest = CACHE_DIR / fname
        if not force and dest.exists():
            age = time.time() - dest.stat().st_mtime
            if age < max_age_sec:
                # Cache is fresh — skip download.
                continue

        url = DATASET_BASE + fname
        try:
            print(f"  [wc_dataset] Downloading {url} …")
            urllib.request.urlretrieve(url, dest)
            print(f"  [wc_dataset] Cached → {dest}")
        except (urllib.error.URLError, OSError) as exc:
            if dest.exists():
                print(
                    f"  [wc_dataset] Network error ({exc}); using cached {dest}"
                )
            else:
                raise RuntimeError(
                    f"Cannot fetch {url} and no cached copy exists at {dest}"
                ) from exc

    # Best-effort download of optional files; warn-and-continue on failure.
    for fname in OPTIONAL_DATASET_FILES:
        dest = CACHE_DIR / fname
        if not force and dest.exists():
            age = time.time() - dest.stat().st_mtime
            if age < max_age_sec:
                continue
        url = DATASET_BASE + fname
        try:
            urllib.request.urlretrieve(url, dest)
        except (urllib.error.URLError, OSError) as exc:
            warnings.warn(
                f"[wc_dataset] Optional file {fname} could not be fetched ({exc}); "
                "skipping.",
                stacklevel=2,
            )


def load_dataset(force: bool = False) -> Dataset:
    """Return a :class:`Dataset` with the four joined DataFrames.

    Teams get their names canonicalised via :data:`DATASET_NAME_MAP`.
    Matches are enriched with ``stage_name``, ``venue_country`` (FIFA code),
    and ``home_team_name`` / ``away_team_name`` (canonical).

    Raises ``KeyError`` loudly if expected columns are missing.
    """
    download_dataset(force=force)

    teams_raw = pd.read_csv(CACHE_DIR / "teams.csv")
    matches_raw = pd.read_csv(CACHE_DIR / "matches.csv")
    stages_raw = pd.read_csv(CACHE_DIR / "tournament_stages.csv")
    venues_raw = pd.read_csv(CACHE_DIR / "venues.csv")

    # Validate required columns.
    _require(teams_raw, ["team_id", "team_name", "fifa_code"], "teams.csv")
    _require(matches_raw,
             ["match_id", "date", "stage_id", "venue_id",
              "home_team_id", "away_team_id", "home_score", "away_score",
              "home_penalty_score", "away_penalty_score", "status", "result_type"],
             "matches.csv")
    _require(stages_raw, ["stage_id", "stage_name"], "tournament_stages.csv")
    _require(venues_raw, ["venue_id", "country"], "venues.csv")

    # Canonicalise team names.
    teams = teams_raw.copy()
    teams["team_name"] = teams["team_name"].replace(DATASET_NAME_MAP)

    # Build a lookup: team_id → canonical name.
    id_to_name: dict[int, str] = dict(zip(teams["team_id"], teams["team_name"]))

    # Enrich matches with stage name, venue country, and canonical team names.
    stage_map = dict(zip(stages_raw["stage_id"], stages_raw["stage_name"]))
    venue_country_map = dict(zip(venues_raw["venue_id"], venues_raw["country"]))

    matches = matches_raw.copy()
    matches["stage_name"] = matches["stage_id"].map(stage_map)
    matches["venue_country"] = matches["venue_id"].map(venue_country_map)
    matches["home_team_name"] = matches["home_team_id"].map(id_to_name)
    matches["away_team_name"] = matches["away_team_id"].map(id_to_name)
    matches["date"] = pd.to_datetime(matches["date"])

    # Optional: match_team_stats.csv — best effort.
    team_stats: pd.DataFrame | None = None
    _stats_path = CACHE_DIR / "match_team_stats.csv"
    if _stats_path.exists():
        try:
            _ts = pd.read_csv(_stats_path)
            _required_cols = ["match_id", "team_id", "possession_pct", "total_shots"]
            if all(c in _ts.columns for c in _required_cols):
                team_stats = _ts
            else:
                _missing = [c for c in _required_cols if c not in _ts.columns]
                warnings.warn(
                    f"[wc_dataset] match_team_stats.csv missing columns {_missing}; "
                    "team_stats set to None.",
                    stacklevel=2,
                )
        except Exception as exc:
            warnings.warn(
                f"[wc_dataset] Failed to parse match_team_stats.csv ({exc}); "
                "team_stats set to None.",
                stacklevel=2,
            )

    return Dataset(
        teams=teams,
        matches=matches,
        tournament_stages=stages_raw,
        venues=venues_raw,
        team_stats=team_stats,
    )


def completed_results_frame(ds: Dataset) -> pd.DataFrame:
    """Convert completed matches to results.csv schema.

    Columns: date, home_team, away_team, home_score, away_score, tournament, neutral.

    Neutral logic:
    - If the *home* team's FIFA code matches the venue country, the home team is a
      co-host playing at home → neutral=False, no swap.
    - If the *away* team's FIFA code matches the venue country, the away team is a
      co-host playing at home → swap home/away (and scores) so the host is listed
      as home, then neutral=False.
    - Otherwise → neutral=True (all matches in this tournament are in USA/CAN/MEX
      so a non-host playing at a co-host venue is neutral).

    Penalty goals are NOT added to the regulation scores.
    """
    # Only completed matches.
    completed = ds.matches[ds.matches["status"].str.upper() == "COMPLETED"].copy()

    # Build fifa_code lookup by team_id.
    code_map: dict[int, str] = dict(zip(ds.teams["team_id"], ds.teams["fifa_code"]))

    rows: list[dict] = []
    for _, m in completed.iterrows():
        home_name: str = m["home_team_name"]
        away_name: str = m["away_team_name"]
        home_score = int(m["home_score"])
        away_score = int(m["away_score"])
        venue_country: str = m["venue_country"]

        home_code = code_map.get(int(m["home_team_id"]), "")
        away_code = code_map.get(int(m["away_team_id"]), "")

        if home_code == venue_country:
            neutral = False
        elif away_code == venue_country:
            # Away team is the host — swap so host is listed as home.
            home_name, away_name = away_name, home_name
            home_score, away_score = away_score, home_score
            neutral = False
        else:
            neutral = True

        rows.append({
            "date": m["date"],
            "home_team": home_name,
            "away_team": away_name,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": "FIFA World Cup",
            "neutral": neutral,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_live_results(df: pd.DataFrame, path: Path = LIVE_CSV) -> None:
    """Write a completed-results DataFrame to *wc2026_live.csv*.

    Keeps the same schema as results.csv so :func:`data_loader.merge_live_results`
    can ingest it without modification.
    """
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [wc_dataset] Saved {len(df)} results → {path}")


def unmapped_team_names(ds: Dataset, bracket: object | None = None) -> list[str]:
    """Return dataset team names that have no canonical match.

    Checks that every name in *teams.csv* (after ``DATASET_NAME_MAP`` is applied)
    appears in the bracket's ``name_to_id`` mapping (if a bracket is provided).
    Also warns loudly to stderr for each unmapped name found.
    """
    if bracket is None:
        return []

    unmapped: list[str] = []
    for name in ds.teams["team_name"]:
        if name not in bracket.name_to_id:
            unmapped.append(name)
    if unmapped:
        warnings.warn(
            f"[wc_dataset] Unmapped team names (not in bracket): {sorted(unmapped)}",
            stacklevel=2,
        )
    return unmapped


def _require(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[wc_dataset] {source} is missing expected columns: {missing}"
        )


if __name__ == "__main__":
    from .data_loader import load_bracket

    print("Loading WC2026 dataset …")
    ds = load_dataset()
    bracket = load_bracket()

    completed = completed_results_frame(ds)
    print(f"\n  Total matches in dataset  : {len(ds.matches)}")
    print(f"  Completed matches         : {len(completed)}")

    # Check unmapped names.
    bad = unmapped_team_names(ds, bracket)
    print(f"  Unmapped team names       : {len(bad)}" + (f" — {bad}" if bad else ""))

    # Save live CSV.
    save_live_results(completed)

    # Verify Mexico–South Africa neutral flag.
    mex_rsa = completed[
        completed["home_team"].isin(["Mexico", "South Africa"]) &
        completed["away_team"].isin(["Mexico", "South Africa"])
    ]
    if not mex_rsa.empty:
        row = mex_rsa.iloc[0]
        print(
            f"\n  Mexico–South Africa: home={row['home_team']}, "
            f"neutral={row['neutral']} (expect False)"
        )
    else:
        print("\n  Mexico–South Africa match not found in completed results")

    print("\nSample (first 5 completed matches):")
    for _, r in completed.head(5).iterrows():
        print(
            f"  {str(r['date'].date())}  {r['home_team']} {r['home_score']}"
            f"–{r['away_score']} {r['away_team']}  "
            f"neutral={r['neutral']}"
        )
