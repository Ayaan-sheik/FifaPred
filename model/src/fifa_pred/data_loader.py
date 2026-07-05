"""Load & clean historical international results; attach time-decay weights.

Source: martj42 "International football results" (the Kaggle dataset, mirrored on
GitHub). One row per match: date, home_team, away_team, home_score, away_score,
tournament, city, country, neutral.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CSV = DATA_DIR / "raw" / "results.csv"
LIVE_CSV = DATA_DIR / "wc2026_live.csv"
BRACKET_JSON = DATA_DIR / "wc2026.json"
CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# CSV uses common English names that already match wc2026.json's `name` field.
# Add aliases here only if a future data source diverges.
NAME_ALIASES: dict[str, str] = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
}


@dataclass
class Bracket:
    """Parsed wc2026.json: the 48-team field + group layout + format config."""

    teams: list[dict]
    groups: dict[str, list[str]]
    fmt: dict
    by_id: dict[str, dict]
    name_to_id: dict[str, str]

    @property
    def team_names(self) -> list[str]:
        return [t["name"] for t in self.teams]

    @property
    def hosts(self) -> set[str]:
        return {t["id"] for t in self.teams if t.get("host")}


def load_bracket(path: Path = BRACKET_JSON) -> Bracket:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    teams = data["teams"]
    by_id = {t["id"]: t for t in teams}
    name_to_id = {t["name"]: t["id"] for t in teams}
    return Bracket(teams=teams, groups=data["groups"], fmt=data["format"],
                   by_id=by_id, name_to_id=name_to_id)


def download_results(force: bool = False) -> Path:
    """Fetch the historical results CSV if it isn't already cached locally."""
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not force:
        return RAW_CSV
    urllib.request.urlretrieve(CSV_URL, RAW_CSV)
    return RAW_CSV


def load_results(
    since: str | None = "2014-01-01",
    until: str | None = None,
    xi: float = 0.30,
    reference_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a cleaned match dataframe with a time-decay ``weight`` column.

    Parameters
    ----------
    since, until : optional ISO date bounds (``until`` excludes later matches, used
        for backtests so the model only sees data available before a tournament).
    xi : time-decay rate; weight = exp(-xi * years_before_reference).
    reference_date : the "now" the decay is measured from (defaults to the latest
        match in the filtered window, or ``until`` if given).
    """
    download_results()
    df = pd.read_csv(RAW_CSV, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_team"] = df["home_team"].replace(NAME_ALIASES)
    df["away_team"] = df["away_team"].replace(NAME_ALIASES)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # `neutral` arrives as bool or "TRUE"/"FALSE" strings depending on pandas version.
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    if since:
        df = df[df["date"] >= pd.Timestamp(since)]
    if until:
        df = df[df["date"] < pd.Timestamp(until)]
    df = df.sort_values("date").reset_index(drop=True)

    ref = pd.Timestamp(reference_date) if reference_date else (
        pd.Timestamp(until) if until else df["date"].max()
    )
    years_ago = (ref - df["date"]).dt.days / 365.25
    df["weight"] = np.exp(-xi * years_ago.clip(lower=0))
    return df


def load_live_results(path: Path = LIVE_CSV) -> pd.DataFrame:
    """Load wc2026_live.csv if it exists; return empty DataFrame otherwise.

    Parses the ``neutral`` column from the CSV rather than forcing it to True,
    so that co-host teams (USA/MEX/CAN) playing in their own country correctly
    retain ``neutral=False`` and the home-advantage parameter is applied.
    """
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # `neutral` may be a Python bool or "TRUE"/"FALSE" string depending on
    # how the CSV was written — normalise the same way load_results() does.
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def teams_in_results(df: pd.DataFrame) -> set[str]:
    return set(df["home_team"]) | set(df["away_team"])


def merge_live_results(
    df: pd.DataFrame,
    live: pd.DataFrame,
    xi: float = 0.30,
) -> pd.DataFrame:
    """Merge live WC2026 scraped rows into the historical training dataframe.

    Deduplicates on (date, home_team, away_team), preferring the scraped row
    over the martj42 CSV row for any WC2026 match that appears in both.
    Recomputes time-decay weights so the live matches (very recent) get a weight
    close to 1.0.
    """
    if live.empty:
        return df

    # ensure live has the same columns we need
    live = live.copy()
    if "tournament" not in live.columns:
        live["tournament"] = "FIFA World Cup"
    if "neutral" not in live.columns:
        live["neutral"] = True

    # tag each source so we can keep the scraped row in duplicates
    df = df.copy()
    df["_src"] = "csv"
    live["_src"] = "live"

    combined = pd.concat([df, live], ignore_index=True)
    # sort so "live" rows come last (they win the dedup)
    combined = combined.sort_values(
        ["date", "home_team", "away_team", "_src"], ascending=[True, True, True, False]
    )
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="first"
    )
    combined = combined.drop(columns=["_src"]).sort_values("date").reset_index(drop=True)

    # recompute weights with the full merged frame
    ref = combined["date"].max()
    years_ago = (ref - combined["date"]).dt.days / 365.25
    combined["weight"] = np.exp(-xi * years_ago.clip(lower=0))
    return combined


def compute_group_standings(live: pd.DataFrame, bracket: "Bracket") -> dict:
    """Compute current group points tables from played WC2026 matches.

    Returns a dict ``{group_label: [row, ...]}`` where each row is:
        {"id": team_id, "name": ..., "flag": ..., "played": N,
         "won": W, "drawn": D, "lost": L, "gf": G, "ga": G, "gd": G, "pts": P}

    Groups with no played matches are omitted from the output. Returns ``{}`` if
    ``live`` is empty.
    """
    if live.empty:
        return {}

    # build a quick name -> team lookup
    name_to_team = {t["name"]: t for t in bracket.teams}

    out: dict[str, list[dict]] = {}
    for group_label, member_ids in bracket.groups.items():
        member_names = {bracket.by_id[tid]["name"] for tid in member_ids}

        # matches where both teams are in this group
        mask = live["home_team"].isin(member_names) & live["away_team"].isin(member_names)
        grp_matches = live[mask]
        if grp_matches.empty:
            continue

        rows: dict[str, dict] = {}
        for name in member_names:
            t = name_to_team.get(name)
            tid = t["id"] if t else name
            flag = t["flag"] if t else ""
            rows[name] = {"id": tid, "name": name, "flag": flag,
                          "played": 0, "won": 0, "drawn": 0, "lost": 0,
                          "gf": 0, "ga": 0, "gd": 0, "pts": 0}

        for _, m in grp_matches.iterrows():
            h, a = m["home_team"], m["away_team"]
            hg, ag = int(m["home_score"]), int(m["away_score"])
            if h not in rows or a not in rows:
                continue
            rows[h]["played"] += 1; rows[a]["played"] += 1
            rows[h]["gf"] += hg;   rows[h]["ga"] += ag
            rows[a]["gf"] += ag;   rows[a]["ga"] += hg
            if hg > ag:
                rows[h]["won"] += 1; rows[h]["pts"] += 3
                rows[a]["lost"] += 1
            elif hg == ag:
                rows[h]["drawn"] += 1; rows[h]["pts"] += 1
                rows[a]["drawn"] += 1; rows[a]["pts"] += 1
            else:
                rows[a]["won"] += 1; rows[a]["pts"] += 3
                rows[h]["lost"] += 1

        for r in rows.values():
            r["gd"] = r["gf"] - r["ga"]

        # sort: pts desc, gd desc, gf desc
        sorted_rows = sorted(
            rows.values(), key=lambda r: (-r["pts"], -r["gd"], -r["gf"])
        )
        out[group_label] = sorted_rows
    return out


if __name__ == "__main__":
    bracket = load_bracket()
    df = load_results()
    present = teams_in_results(df)
    missing = [n for n in bracket.team_names if n not in present]
    print(f"Loaded {len(df):,} matches since {df['date'].min().date()} "
          f"-> {df['date'].max().date()}")
    print(f"Bracket teams with no recent matches: {missing or 'none'}")
    print(df[["date", "home_team", "away_team", "home_score", "away_score",
              "weight"]].tail(3).to_string(index=False))
