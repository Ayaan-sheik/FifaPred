"""Generate wc2026.json — the 48-team field, 12 groups, host/confederation config.

NOTE: This is a hand-seeded, PLAUSIBLE field for the 2026 World Cup, not the
official draw. The qualified teams, their groups, and Elo seeds should be
verified against the real tournament before the output is shared publicly.
Re-run with `python make_bracket.py` to regenerate data/wc2026.json.
"""

import json
from pathlib import Path

HOSTS = {"USA", "CAN", "MEX"}

# (id, name, confederation, flag-emoji, seed Elo) — plausible 48-team field.
TEAMS = [
    # CONMEBOL
    ("ARG", "Argentina", "CONMEBOL", "\U0001F1E6\U0001F1F7", 2105),
    ("BRA", "Brazil", "CONMEBOL", "\U0001F1E7\U0001F1F7", 2025),
    ("URU", "Uruguay", "CONMEBOL", "\U0001F1FA\U0001F1FE", 1895),
    ("COL", "Colombia", "CONMEBOL", "\U0001F1E8\U0001F1F4", 1885),
    ("ECU", "Ecuador", "CONMEBOL", "\U0001F1EA\U0001F1E8", 1790),
    ("PAR", "Paraguay", "CONMEBOL", "\U0001F1F5\U0001F1FE", 1720),
    # UEFA
    ("FRA", "France", "UEFA", "\U0001F1EB\U0001F1F7", 2085),
    ("ESP", "Spain", "UEFA", "\U0001F1EA\U0001F1F8", 2070),
    ("ENG", "England", "UEFA", "\U0001F1EC\U0001F1E7", 2015),
    ("POR", "Portugal", "UEFA", "\U0001F1F5\U0001F1F9", 2000),
    ("NED", "Netherlands", "UEFA", "\U0001F1F3\U0001F1F1", 1990),
    ("GER", "Germany", "UEFA", "\U0001F1E9\U0001F1EA", 1965),
    ("ITA", "Italy", "UEFA", "\U0001F1EE\U0001F1F9", 1945),
    ("BEL", "Belgium", "UEFA", "\U0001F1E7\U0001F1EA", 1950),
    ("CRO", "Croatia", "UEFA", "\U0001F1ED\U0001F1F7", 1905),
    ("SUI", "Switzerland", "UEFA", "\U0001F1E8\U0001F1ED", 1850),
    ("DEN", "Denmark", "UEFA", "\U0001F1E9\U0001F1F0", 1840),
    ("AUT", "Austria", "UEFA", "\U0001F1E6\U0001F1F9", 1795),
    ("TUR", "Turkey", "UEFA", "\U0001F1F9\U0001F1F7", 1775),
    ("UKR", "Ukraine", "UEFA", "\U0001F1FA\U0001F1E6", 1770),
    ("SRB", "Serbia", "UEFA", "\U0001F1F7\U0001F1F8", 1760),
    ("POL", "Poland", "UEFA", "\U0001F1F5\U0001F1F1", 1740),
    # CONCACAF (incl. 3 hosts)
    ("USA", "United States", "CONCACAF", "\U0001F1FA\U0001F1F8", 1820),
    ("MEX", "Mexico", "CONCACAF", "\U0001F1F2\U0001F1FD", 1800),
    ("CAN", "Canada", "CONCACAF", "\U0001F1E8\U0001F1E6", 1750),
    ("CRC", "Costa Rica", "CONCACAF", "\U0001F1E8\U0001F1F7", 1680),
    ("PAN", "Panama", "CONCACAF", "\U0001F1F5\U0001F1E6", 1670),
    ("JAM", "Jamaica", "CONCACAF", "\U0001F1EF\U0001F1F2", 1650),
    # CAF
    ("MAR", "Morocco", "CAF", "\U0001F1F2\U0001F1E6", 1860),
    ("SEN", "Senegal", "CAF", "\U0001F1F8\U0001F1F3", 1810),
    ("NGA", "Nigeria", "CAF", "\U0001F1F3\U0001F1EC", 1760),
    ("EGY", "Egypt", "CAF", "\U0001F1EA\U0001F1EC", 1740),
    ("ALG", "Algeria", "CAF", "\U0001F1E9\U0001F1FF", 1730),
    ("CIV", "Ivory Coast", "CAF", "\U0001F1E8\U0001F1EE", 1720),
    ("CMR", "Cameroon", "CAF", "\U0001F1E8\U0001F1F2", 1710),
    ("GHA", "Ghana", "CAF", "\U0001F1EC\U0001F1ED", 1700),
    ("TUN", "Tunisia", "CAF", "\U0001F1F9\U0001F1F3", 1695),
    # AFC
    ("JPN", "Japan", "AFC", "\U0001F1EF\U0001F1F5", 1825),
    ("IRN", "Iran", "AFC", "\U0001F1EE\U0001F1F7", 1760),
    ("KOR", "South Korea", "AFC", "\U0001F1F0\U0001F1F7", 1755),
    ("AUS", "Australia", "AFC", "\U0001F1E6\U0001F1FA", 1730),
    ("QAT", "Qatar", "AFC", "\U0001F1F6\U0001F1E6", 1690),
    ("KSA", "Saudi Arabia", "AFC", "\U0001F1F8\U0001F1E6", 1680),
    ("UZB", "Uzbekistan", "AFC", "\U0001F1FA\U0001F1FF", 1670),
    ("IRQ", "Iraq", "AFC", "\U0001F1EE\U0001F1F6", 1660),
    # OFC
    ("NZL", "New Zealand", "OFC", "\U0001F1F3\U0001F1FF", 1600),
    # Intercontinental playoff winners (plausible)
    ("COD", "DR Congo", "CAF", "\U0001F1E8\U0001F1E9", 1680),
    ("BOL", "Bolivia", "CONMEBOL", "\U0001F1E7\U0001F1F4", 1640),
]

GROUP_LABELS = [chr(ord("A") + i) for i in range(12)]  # A..L


def snake_draft(teams):
    """Seed teams into 12 groups via an Elo-ordered snake draft, with the three
    hosts pinned to the top of groups A/B/D so the bracket looks like a real draw."""
    ordered = sorted(teams, key=lambda t: -t[4])
    groups = {g: [] for g in GROUP_LABELS}

    # Pin hosts as group heads (USA->A, MEX->B, CAN->D), as FIFA does for hosts.
    host_slots = {"USA": "A", "MEX": "B", "CAN": "D"}
    for tid, g in host_slots.items():
        team = next(t for t in ordered if t[0] == tid)
        groups[g].append(team)
        ordered.remove(team)

    # Snake-draft the rest into the remaining slots.
    seq = GROUP_LABELS + GROUP_LABELS[::-1]
    seq = seq * 4  # plenty of passes
    gi = 0
    for team in ordered:
        # advance to next group that still has < 4 teams
        while len(groups[seq[gi]]) >= 4:
            gi += 1
        groups[seq[gi]].append(team)
        gi += 1
    return groups


def main():
    groups = snake_draft(TEAMS)
    teams_out = []
    groups_out = {}
    for g in GROUP_LABELS:
        members = groups[g]
        groups_out[g] = [t[0] for t in members]
        for tid, name, conf, flag, elo in members:
            teams_out.append({
                "id": tid, "name": name, "confederation": conf,
                "flag": flag, "group": g, "elo_seed": elo,
                "host": tid in HOSTS,
            })

    out = {
        "_note": "Hand-seeded plausible 2026 field, NOT the official draw. Verify before sharing.",
        "format": {"groups": 12, "teams_per_group": 4,
                   "advance_per_group": 2, "best_third_advancing": 8,
                   "knockout_start": "R32"},
        "groups": groups_out,
        "teams": teams_out,
    }
    path = Path(__file__).parent / "wc2026.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {g: len(v) for g, v in groups_out.items()}
    print(f"Wrote {path} — {len(teams_out)} teams, group sizes: {counts}")


if __name__ == "__main__":
    main()
