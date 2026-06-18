"""Fetch live WC2026 match results from worldcup26.ir (free, no API key).

Run standalone to print fetched results:

    uv run python -m fifa_pred.scraper

No environment variables required. The endpoint returns all 104 matches in one
call; we filter to finished games only.

API: GET https://worldcup26.ir/get/games
  - No auth headers needed
  - Returns a JSON array; each element has the fields used below
  - finished      : "TRUE" | "FALSE" (string, not boolean)
  - time_elapsed  : "finished" | <minutes> | ...
  - local_date    : "MM/DD/YYYY HH:MM"
  - home_team_name_en / away_team_name_en : English team names
  - home_score / away_score : int or numeric string
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LIVE_CSV = DATA_DIR / "wc2026_live.csv"

_GAMES_URL = "https://worldcup26.ir/get/games"

# Map worldcup26.ir English names → our canonical names in wc2026.json / results.csv.
# The API generally uses clean English names; only add entries that diverge.
TEAM_NAME_MAP: dict[str, str] = {
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Cape Verde Islands": "Cape Verde",
    # Mirror canonical names so lookups always hit the dict (identity aliases)
    "Argentina": "Argentina",
    "Brazil": "Brazil",
    "France": "France",
    "Spain": "Spain",
    "England": "England",
    "Portugal": "Portugal",
    "Germany": "Germany",
    "Netherlands": "Netherlands",
    "Belgium": "Belgium",
    "Croatia": "Croatia",
    "Switzerland": "Switzerland",
    "Austria": "Austria",
    "Uruguay": "Uruguay",
    "Colombia": "Colombia",
    "Ecuador": "Ecuador",
    "Paraguay": "Paraguay",
    "Mexico": "Mexico",
    "United States": "United States",
    "Canada": "Canada",
    "Panama": "Panama",
    "Morocco": "Morocco",
    "Senegal": "Senegal",
    "Egypt": "Egypt",
    "Algeria": "Algeria",
    "Ivory Coast": "Ivory Coast",
    "Ghana": "Ghana",
    "Tunisia": "Tunisia",
    "South Africa": "South Africa",
    "Cape Verde": "Cape Verde",
    "Japan": "Japan",
    "South Korea": "South Korea",
    "Australia": "Australia",
    "Iran": "Iran",
    "Saudi Arabia": "Saudi Arabia",
    "Qatar": "Qatar",
    "Uzbekistan": "Uzbekistan",
    "Jordan": "Jordan",
    "New Zealand": "New Zealand",
    "Scotland": "Scotland",
    "Norway": "Norway",
    "Haiti": "Haiti",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    # TBD playoff qualifiers — confirmed before group stage
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czech Republic",
    "Iraq": "Iraq",
    "Sweden": "Sweden",
    "Turkey": "Turkey",
    "Democratic Republic of the Congo": "DR Congo",
}


def _parse_date(raw: str) -> str:
    """Parse 'MM/DD/YYYY HH:MM' → 'YYYY-MM-DD'."""
    return datetime.strptime(raw[:10], "%m/%d/%Y").strftime("%Y-%m-%d")


def fetch_wc2026_results() -> list[dict]:
    """Return completed WC2026 fixtures as martj42-schema dicts.

    Each dict has: date, home_team, away_team, home_score, away_score,
    tournament, neutral — ready to concat into the training dataframe.

    Returns [] on any network or parse error (graceful degradation so
    the rest of the pipeline continues with the existing live CSV).
    """
    try:
        req = urllib.request.Request(_GAMES_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError) as exc:
        print(f"  [scraper] Cannot reach {_GAMES_URL}: {exc}")
        print("  [scraper] Using existing live CSV (if any).")
        return []
    except Exception as exc:
        print(f"  [scraper] Unexpected error fetching games: {exc}")
        return []

    # API may return a bare list or a wrapper dict like {"data": [...]}
    if isinstance(payload, dict):
        games: list[dict] = payload.get("data", payload.get("games", []))
    elif isinstance(payload, list):
        games = payload
    else:
        print(f"  [scraper] Unexpected response type: {type(payload).__name__}")
        return []

    results: list[dict] = []
    unmapped: set[str] = set()

    for g in games:
        if not isinstance(g, dict):
            continue
        if g.get("finished") != "TRUE" or g.get("time_elapsed") != "finished":
            continue
        # Field may be top-level or inside an embedded team object
        home_raw = (
            str(g.get("home_team_name_en", "")).strip()
            or str((g.get("homeTeam") or {}).get("name_en", "")).strip()
        )
        away_raw = (
            str(g.get("away_team_name_en", "")).strip()
            or str((g.get("visitingTeam") or {}).get("name_en", "")).strip()
        )
        home = TEAM_NAME_MAP.get(home_raw)
        away = TEAM_NAME_MAP.get(away_raw)
        if home is None:
            unmapped.add(home_raw)
        if away is None:
            unmapped.add(away_raw)
        if home is None or away is None:
            continue
        try:
            hg = int(g["home_score"])
            ag = int(g["away_score"])
        except (KeyError, ValueError, TypeError):
            continue
        try:
            date_str = _parse_date(str(g.get("local_date", "")))
        except (ValueError, TypeError):
            continue

        results.append({
            "date": date_str,
            "home_team": home,
            "away_team": away,
            "home_score": hg,
            "away_score": ag,
            "tournament": "FIFA World Cup",
            "neutral": True,
        })

    if unmapped:
        print(
            f"  [scraper] UNMAPPED team names (add to TEAM_NAME_MAP in scraper.py): "
            f"{sorted(unmapped)}"
        )
    print(f"  [scraper] Fetched {len(results)} completed WC2026 fixture(s).")
    return results


def fetch_group_standings_api() -> dict:
    """Stub — worldcup26.ir has no pre-built standings endpoint.

    Returns {} so build_predictions.py falls back to compute_group_standings(),
    which computes standings from the scraped live match results.
    """
    return {}


def save_live_results(results: list[dict], path: Path = LIVE_CSV) -> None:
    """Write scraped results to wc2026_live.csv (same schema as results.csv)."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"  [scraper] Saved {len(results)} results → {path}")


def load_live_results(path: Path = LIVE_CSV) -> pd.DataFrame:
    """Load wc2026_live.csv if it exists; return empty DataFrame otherwise."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = True
    return df


if __name__ == "__main__":
    results = fetch_wc2026_results()
    if results:
        save_live_results(results)
        print("\nSample (up to 5):")
        for r in results[:5]:
            print(
                f"  {r['date']}  {r['home_team']} {r['home_score']}"
                f"–{r['away_score']} {r['away_team']}"
            )
    else:
        print("No completed results fetched (network issue or no matches played yet).")
