"""Sync wc2026.json with the real 2026 World Cup bracket from worldcup26.ir.

Downloads the teams CSV from the GitHub repository (no auth needed) and writes
the authoritative model/data/wc2026.json, replacing the hand-seeded placeholder.

Run once after the tournament draw, or any time the bracket changes:

    uv run python data/sync_bracket.py

Teams with TBD / playoff slots keep a placeholder entry so the bracket structure
stays complete (48 teams, 12 groups of 4).
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
from pathlib import Path

TEAMS_CSV_URL = (
    "https://raw.githubusercontent.com/rezarahiminia/worldcup2026/main/worldcup2026.teams.csv"
)

OUT = Path(__file__).parent / "wc2026.json"

HOSTS = {"MEX", "USA", "CAN"}

# Confederation by FIFA code — hardcoded since the teams CSV doesn't include it.
FIFA_CODE_TO_CONF: dict[str, str] = {
    # UEFA
    "SCO": "UEFA", "GER": "UEFA", "NED": "UEFA", "FRA": "UEFA", "BEL": "UEFA",
    "ESP": "UEFA", "ENG": "UEFA", "SUI": "UEFA", "CRO": "UEFA", "AUT": "UEFA",
    "NOR": "UEFA", "TUR": "UEFA", "UKR": "UEFA", "SRB": "UEFA", "POL": "UEFA",
    "POR": "UEFA", "ITA": "UEFA", "DEN": "UEFA", "BIH": "UEFA", "SWE": "UEFA",
    "CZE": "UEFA",
    # CONMEBOL
    "BRA": "CONMEBOL", "ARG": "CONMEBOL", "COL": "CONMEBOL", "ECU": "CONMEBOL",
    "PAR": "CONMEBOL", "URU": "CONMEBOL", "CHI": "CONMEBOL", "BOL": "CONMEBOL",
    "PER": "CONMEBOL", "VEN": "CONMEBOL",
    # CONCACAF
    "MEX": "CONCACAF", "USA": "CONCACAF", "CAN": "CONCACAF", "CRC": "CONCACAF",
    "PAN": "CONCACAF", "JAM": "CONCACAF", "HON": "CONCACAF", "CUW": "CONCACAF",
    "HAI": "CONCACAF", "CUB": "CONCACAF",
    # AFC
    "KOR": "AFC", "JPN": "AFC", "QAT": "AFC", "AUS": "AFC", "IRN": "AFC",
    "KSA": "AFC", "JOR": "AFC", "UZB": "AFC", "CHN": "AFC", "IRQ": "AFC",
    # CAF
    "RSA": "CAF", "MAR": "CAF", "TUN": "CAF", "SEN": "CAF", "EGY": "CAF",
    "GHA": "CAF", "CIV": "CAF", "ALG": "CAF", "NGA": "CAF", "CMR": "CAF",
    "CPV": "CAF", "COD": "CAF", "DRC": "CAF",
    # OFC
    "NZL": "OFC",
}


# Teams whose iso2 field in the CSV is a 3-letter FIFA code rather than ISO 3166-1 alpha-2.
# Map to the correct 2-letter code (or a hardcoded emoji for subdivisions).
ISO2_OVERRIDES: dict[str, str] = {
    "ENG": "GB",   # England → Union Jack (closest standard flag)
    "SCO": "GB",   # Scotland → Union Jack (subdivision emoji too niche)
}
# Direct emoji overrides when no 2-letter code is a good fit
EMOJI_OVERRIDES: dict[str, str] = {}

# Maps GitHub CSV placeholder names → (fifa_code, canonical_name, iso2).
# Update this when intercontinental/UEFA playoff results are confirmed.
TBD_RESOLUTIONS: dict[str, tuple[str, str, str]] = {
    "UEFA Path A Winner": ("BIH", "Bosnia and Herzegovina", "BA"),
    "UEFA Path B Winner": ("SWE", "Sweden", "SE"),
    "UEFA Path C Winner": ("TUR", "Turkey", "TR"),
    "UEFA Path D Winner": ("CZE", "Czech Republic", "CZ"),
    "IC Path 1 Winner":   ("COD", "DR Congo", "CD"),
    "IC Path 2 Winner":   ("IRQ", "Iraq", "IQ"),
}


def iso2_to_flag(iso2: str) -> str:
    """Convert a 2-letter ISO country code to its flag emoji via regional indicators."""
    if not iso2 or iso2.upper() in ("TBD", ""):
        return "🏳️"
    code = iso2.upper()
    if code in EMOJI_OVERRIDES:
        return EMOJI_OVERRIDES[code]
    code = ISO2_OVERRIDES.get(code, code)
    if len(code) != 2:
        return "🏳️"
    try:
        return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code)
    except Exception:
        return "🏳️"


def fetch_teams() -> list[dict]:
    with urllib.request.urlopen(TEAMS_CSV_URL, timeout=15) as resp:
        text = resp.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def build_bracket(teams_raw: list[dict]) -> dict:
    teams_out = []
    groups_out: dict[str, list[str]] = {}

    tbd_seen: set[str] = set()
    for row in teams_raw:
        tid = row["fifa_code"].strip()
        if not tid:
            continue
        name = row["name_en"].strip()
        group = row["groups"].strip()
        iso2 = row["iso2"].strip()

        # Resolve TBD placeholders to confirmed qualifiers
        if tid == "TBD" and name in TBD_RESOLUTIONS:
            tid, name, iso2 = TBD_RESOLUTIONS[name]

        conf = FIFA_CODE_TO_CONF.get(tid, "Other")

        # TBD playoff slots: make IDs unique per group so by_id dict doesn't collide
        if tid == "TBD" or tid in tbd_seen:
            tid = f"TBD_{group}"
        tbd_seen.add(tid)

        entry = {
            "id": tid,
            "name": name,
            "confederation": conf,
            "flag": iso2_to_flag(iso2),
            "group": group,
            "host": tid in HOSTS,
        }
        teams_out.append(entry)
        groups_out.setdefault(group, []).append(tid)

    # sort groups alphabetically
    groups_out = dict(sorted(groups_out.items()))

    return {
        "_note": (
            "Synced from worldcup26.ir / rezarahiminia/worldcup2026 on GitHub. "
            "TBD slots are playoff qualifiers not yet determined."
        ),
        "format": {
            "groups": 12,
            "teams_per_group": 4,
            "advance_per_group": 2,
            "best_third_advancing": 8,
            "knockout_start": "R32",
        },
        "groups": groups_out,
        "teams": teams_out,
    }


def main():
    print("Fetching teams from GitHub…")
    raw = fetch_teams()
    print(f"  Got {len(raw)} teams.")

    payload = build_bracket(raw)
    n_groups = len(payload["groups"])
    group_sizes = {g: len(v) for g, v in payload["groups"].items()}
    print(f"  Groups: {n_groups}  sizes: {group_sizes}")

    # Print the bracket for a quick eyeball
    for g in sorted(payload["groups"]):
        ids = payload["groups"][g]
        names = [next(t["name"] for t in payload["teams"] if t["id"] == i) for i in ids]
        print(f"  Group {g}: {', '.join(names)}")

    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
