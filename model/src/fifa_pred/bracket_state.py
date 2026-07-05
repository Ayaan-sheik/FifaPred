"""Derive tournament state from the completed WC2026 match dataset.

Match-ID stage ranges
---------------------
  1–72    Group Stage
  73–88   Round of 32 (R32)
  89–96   Round of 16 (R16)
  97–100  Quarter-finals (QF)
  101–102 Semi-finals (SF)
  103     Third-place match (skipped — not simulated)
  104     Final

Fallback bracket tree
---------------------
``FALLBACK_FEEDS`` encodes how each knockout match's two participants are
determined when the dataset does not yet list the fixture explicitly.

R16 feeds are derived from the official 2026 FIFA bracket draw:
  89→(75,78)  PAR(w75) vs FRA(w78)
  90→(73,76)  CAN(w73) vs MAR(w76)
  91→(84,83)  POR(w84) vs ESP(w83)
  92→(82,81)  USA(w82) vs BEL(w81)
  93→(74,77)  BRA(w74) vs NOR(w77)
  94→(79,80)  MEX(w79) vs ENG(w80)
  95→(87,86)  ARG(w87) vs EGY(w86)
  96→(85,88)  SUI(w85) vs COL(w88)

QF/SF/Final feeds follow the standard bracket halves:
  97→(89,90), 98→(91,92), 99→(93,94), 100→(95,96)
  101→(97,98), 102→(99,100), 104→(101,102)
  Match 103 (Third-place): **skipped** — not modelled.

``derive_feeds()`` overrides entries here when *completed* fixtures in the
dataset identify participants that can be traced to prior match winners.
Scheduled (pending) fixtures are intentionally ignored — the mominullptr
dataset pre-populates scheduled R16 fixtures with pre-seeding team IDs that
can differ from the actual bracket draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Bracket feed table
# ---------------------------------------------------------------------------

# FALLBACK_FEEDS[match_id] = (feeder_match_id_home, feeder_match_id_away)
# Only match 89 is confirmed; see module docstring for the rest.
FALLBACK_FEEDS: dict[int, tuple[int, int]] = {
    # R16 — official 2026 FIFA bracket draw
    89: (75, 78),   # PAR(w75) vs FRA(w78)
    90: (73, 76),   # CAN(w73) vs MAR(w76)
    91: (84, 83),   # POR(w84) vs ESP(w83)
    92: (82, 81),   # USA(w82) vs BEL(w81)
    93: (74, 77),   # BRA(w74) vs NOR(w77)
    94: (79, 80),   # MEX(w79) vs ENG(w80)
    95: (87, 86),   # ARG(w87) vs EGY(w86)
    96: (85, 88),   # SUI(w85) vs COL(w88)
    # QF
    97: (89, 90),
    98: (91, 92),
    99: (93, 94),
    100: (95, 96),
    # SF
    101: (97, 98),
    102: (99, 100),
    # Final (skip 103 — third-place match)
    104: (101, 102),
}

# Match-id ranges per stage.
_STAGE_OF = {
    **{i: "Group Stage"    for i in range(1, 73)},
    **{i: "Round of 32"    for i in range(73, 89)},
    **{i: "Round of 16"    for i in range(89, 97)},
    **{i: "Quarter-finals" for i in range(97, 101)},
    **{i: "Semi-finals"    for i in range(101, 103)},
    103: "Third-place match",
    104: "Final",
}

# All knockout match ids we care about (excluding third-place).
_KNOCKOUT_IDS = list(range(73, 103)) + [104]

# Short stage names for placeholder labels.
_STAGE_SHORT: dict[int, str] = {
    **{i: "R32" for i in range(73, 89)},
    **{i: "R16" for i in range(89, 97)},
    **{i: "QF"  for i in range(97, 101)},
    **{i: "SF"  for i in range(101, 103)},
    104: "Final",
}


def bracket_order(feeds: dict[int, tuple[int, int]]) -> dict[int, int]:
    """Map every knockout match_id to its left-to-right position (0-15) in
    the bracket tree, derived from ``feeds``.

    Match-id order (73, 74, 75, ...) is just fixture/dataset sequence — it
    does not correspond to which side of the draw a match sits on. This walks
    the tree from the Final (104) down to its 16 R32 leaf descendants
    (home-subtree before away-subtree at every level) to recover the true
    left-to-right order. Every non-leaf match's position is defined as the
    position of its leftmost leaf descendant, so matches within the same
    stage sort correctly relative to one another.
    """

    def leftmost_leaf(mid: int) -> int:
        feed = feeds.get(mid)
        if feed is None:
            return mid
        return leftmost_leaf(feed[0])

    leaves: list[int] = []

    def collect_leaves(mid: int) -> None:
        feed = feeds.get(mid)
        if feed is None:
            leaves.append(mid)
            return
        collect_leaves(feed[0])
        collect_leaves(feed[1])

    collect_leaves(104)
    leaf_position = {mid: i for i, mid in enumerate(leaves)}

    return {mid: leaf_position[leftmost_leaf(mid)] for mid in _KNOCKOUT_IDS}


def placeholder_label(mid: int) -> str:
    """Return a human-readable placeholder for the winner of *mid*.

    Example: ``placeholder_label(89)`` → ``"Winner of R16 #89"``.
    """
    stage = _STAGE_SHORT.get(mid, "Match")
    return f"Winner of {stage} #{mid}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TournamentState:
    """Snapshot of the WC2026 tournament derived from real match data.

    Attributes
    ----------
    group_stage_complete : bool
        True once all 72 group-stage matches have been played.
    played_group : pd.DataFrame
        Subset of ``completed_results_frame()`` output covering matches 1–72.
    knockout_matches : pd.DataFrame
        Rows from *matches.csv* for match-ids 73–104 (regardless of status),
        with columns: match_id, stage_id, home_team_id, away_team_id,
        home_score, away_score, home_penalty_score, away_penalty_score,
        result_type, status, home_name, away_name.
    feeds : dict[int, tuple[int, int]]
        Maps each knockout match_id to the two feeder match_ids whose winners
        will participate.  Starts from ``FALLBACK_FEEDS``; ``derive_feeds()``
        overrides entries where the real fixture has identifiable participants.
    bracket_order : dict[int, int]
        Maps each knockout match_id to its left-to-right position (0-15) in
        the bracket tree, derived from ``feeds``. Use this instead of
        ``match_id`` to order matches for display within a stage.
    r32_slots : dict[str, str]
        Maps team-id → "advance" for the 32 R32 qualifiers.  Includes both
        group qualifiers and the confirmed R32 advancers (from completed group
        matches).
    reached : dict[str, str]
        Maps team_id → furthest stage string reached so far.
    eliminated : set[str]
        Set of team_ids confirmed eliminated.
    next_stage : str
        Human-readable label for the current active/next stage.
    """

    group_stage_complete: bool
    played_group: pd.DataFrame
    knockout_matches: pd.DataFrame
    feeds: dict[int, tuple[int, int]]
    bracket_order: dict[int, int]
    r32_slots: dict[str, str]           # team_id → "advance"
    reached: dict[str, str]             # team_id → stage label
    eliminated: set[str]
    next_stage: str


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def winner_of(row: pd.Series) -> int | None:
    """Return the winning ``team_id`` for a completed knockout match row.

    Handles result_type values: ``Regular``, ``AET``, ``Penalties``.
    Returns ``None`` for incomplete or unparseable rows.
    """
    if str(row.get("status", "")).upper() != "COMPLETED":
        return None

    home_id = _int_or_none(row.get("home_team_id"))
    away_id = _int_or_none(row.get("away_team_id"))
    if home_id is None or away_id is None:
        return None

    hs = _int_or_none(row.get("home_score"))
    as_ = _int_or_none(row.get("away_score"))
    if hs is None or as_ is None:
        return None

    result_type = str(row.get("result_type", "") or "").upper()

    if result_type == "PENALTIES":
        hp = _int_or_none(row.get("home_penalty_score"))
        ap = _int_or_none(row.get("away_penalty_score"))
        if hp is not None and ap is not None:
            return home_id if hp > ap else away_id
        # Fall back to regulation score if penalty data is missing.

    # Regular / AET / fallback from Penalties without penalty data.
    if hs > as_:
        return home_id
    elif as_ > hs:
        return away_id
    # Draw after AET but no penalty data — shouldn't happen in practice.
    return None


def derive_feeds(
    knockout_matches: pd.DataFrame,
) -> dict[int, tuple[int, int]]:
    """Derive actual bracket feeds from listed fixtures.

    When the dataset lists a knockout fixture with known ``home_team_id`` and
    ``away_team_id``, we can identify which R32/R16/QF/SF matches produced
    those participants and update the feed map.

    If any derived feed differs from the FALLBACK_FEEDS entry, a WARNING is
    printed so the developer can update the constant.

    Returns the merged feeds dict (derived overrides fallback on conflicts).
    """
    feeds = dict(FALLBACK_FEEDS)

    # Build a map: team_id → match_id where that team was the winner.
    # We process matches in ascending order so earlier rounds are resolved first.
    winner_match: dict[int, int] = {}  # team_id → match_id they won

    # Sort by match_id ascending to process rounds in order.
    km = knockout_matches.sort_values("match_id")

    for _, row in km.iterrows():
        mid = int(row["match_id"])
        w = winner_of(row)
        if w is not None:
            winner_match[w] = mid

    for _, row in km.iterrows():
        # Only override FALLBACK_FEEDS from completed fixtures.  Scheduled
        # matches may carry pre-seeding team IDs that don't match the real
        # bracket draw, so we ignore them here.
        if str(row.get("status", "")).upper() != "COMPLETED":
            continue

        mid = int(row["match_id"])
        home_id = _int_or_none(row.get("home_team_id"))
        away_id = _int_or_none(row.get("away_team_id"))

        if home_id is None or away_id is None:
            continue  # fixture not yet scheduled with participants

        feeder_home = winner_match.get(home_id)
        feeder_away = winner_match.get(away_id)

        if feeder_home is not None and feeder_away is not None:
            derived = (feeder_home, feeder_away)
            if mid in FALLBACK_FEEDS and FALLBACK_FEEDS[mid] != derived:
                print(
                    f"  [bracket_state] WARNING: FALLBACK_FEEDS[{mid}] = "
                    f"{FALLBACK_FEEDS[mid]} but derived feeds = {derived}. "
                    f"Update the constant."
                )
            feeds[mid] = derived

    return feeds


def build_tournament_state(ds: "Dataset", bracket: "Bracket") -> TournamentState:  # noqa: F821
    """Build a :class:`TournamentState` from the live dataset and bracket.

    The *ds* dataset must already have canonical team names (i.e. after
    ``DATASET_NAME_MAP`` has been applied).  ``bracket.name_to_id`` maps
    canonical name → FIFA code used as team_id.

    Group-stage matches are identified by match_id 1–72; knockout matches are
    73–104 (excluding 103 which is the third-place match we skip).
    """
    matches = ds.matches.copy()

    # Build: dataset team_id → bracket team_id (FIFA code)
    id_to_name: dict[int, str] = dict(
        zip(ds.teams["team_id"], ds.teams["team_name"])
    )
    id_to_bracket_id: dict[int, str] = {}
    for ds_id, name in id_to_name.items():
        bracket_id = bracket.name_to_id.get(name)
        if bracket_id is not None:
            id_to_bracket_id[ds_id] = bracket_id

    # Separate group-stage and knockout matches.
    group_mask = matches["match_id"].between(1, 72)
    ko_mask = (matches["match_id"].between(73, 102)) | (matches["match_id"] == 104)

    group_matches = matches[group_mask]
    knockout_matches = matches[ko_mask].copy()

    # Enrich knockout_matches with canonical bracket ids.
    knockout_matches["home_bracket_id"] = (
        knockout_matches["home_team_id"].map(id_to_bracket_id)
    )
    knockout_matches["away_bracket_id"] = (
        knockout_matches["away_team_id"].map(id_to_bracket_id)
    )

    group_stage_complete = (
        len(group_matches[group_matches["status"].str.upper() == "COMPLETED"]) == 72
    )

    # Build played_group as a simplified frame.
    played_group = group_matches[
        group_matches["status"].str.upper() == "COMPLETED"
    ].copy()

    # Determine R32 qualifiers from the group stage.
    # All teams in bracket are potential qualifiers; mark those confirmed.
    r32_slots: dict[str, str] = {}
    reached: dict[str, str] = {}
    eliminated: set[str] = set()

    # Every team that appears in a R32 match has qualified for the knockout stage.
    r32_matches = knockout_matches[knockout_matches["match_id"].between(73, 88)]
    for _, row in r32_matches.iterrows():
        for col in ("home_bracket_id", "away_bracket_id"):
            bid = row.get(col)
            if bid and pd.notna(bid):
                r32_slots[bid] = "advance"
                reached[bid] = "Round of 32"

    # Teams that participated in the group stage but are NOT in any R32 match
    # are eliminated after the group stage.
    all_bracket_ids = {t["id"] for t in bracket.teams}
    group_teams_ds: set[int] = set()
    for _, row in group_matches.iterrows():
        group_teams_ds.add(int(row["home_team_id"]))
        group_teams_ds.add(int(row["away_team_id"]))

    # If group stage is complete, everyone NOT in r32_slots is eliminated.
    if group_stage_complete:
        for ds_tid in group_teams_ds:
            bid = id_to_bracket_id.get(ds_tid)
            if bid and bid not in r32_slots:
                eliminated.add(bid)
                if bid not in reached:
                    reached[bid] = "Group Stage"

    # Walk through completed knockout matches to update reached/eliminated.
    stage_label = {
        **{i: "Round of 32"    for i in range(73, 89)},
        **{i: "Round of 16"    for i in range(89, 97)},
        **{i: "Quarter-finals" for i in range(97, 101)},
        **{i: "Semi-finals"    for i in range(101, 103)},
        104: "Final",
    }
    next_stage_label_map = {
        "Round of 32": "Round of 16",
        "Round of 16": "Quarter-finals",
        "Quarter-finals": "Semi-finals",
        "Semi-finals": "Final",
        "Final": "Champion",
    }

    sorted_ko = knockout_matches.sort_values("match_id")
    for _, row in sorted_ko.iterrows():
        mid = int(row["match_id"])
        stage = stage_label.get(mid, "Knockout")
        home_bid = row.get("home_bracket_id")
        away_bid = row.get("away_bracket_id")

        if str(row.get("status", "")).upper() != "COMPLETED":
            continue

        # Both participants reached this stage.
        for bid in (home_bid, away_bid):
            if bid and pd.notna(bid):
                if bid not in reached or _stage_rank(stage) > _stage_rank(reached[bid]):
                    reached[bid] = stage

        w = winner_of(row)
        if w is None:
            continue
        winner_bid = id_to_bracket_id.get(w)

        for col, tid in (("home_team_id", home_bid), ("away_team_id", away_bid)):
            bid = row.get(col.replace("team_id", "bracket_id")) if col.startswith("home") else away_bid
            if bid and pd.notna(bid) and bid != winner_bid:
                eliminated.add(bid)

        if winner_bid:
            next_s = next_stage_label_map.get(stage, "Champion")
            if winner_bid not in reached or _stage_rank(next_s) > _stage_rank(reached.get(winner_bid, "")):
                reached[winner_bid] = next_s

    # Determine next active stage.
    completed_ko_count = (
        sorted_ko["status"].str.upper() == "COMPLETED"
    ).sum()
    if completed_ko_count < 16:
        next_stage = "Round of 32"
    elif completed_ko_count < 24:
        next_stage = "Round of 16"
    elif completed_ko_count < 28:
        next_stage = "Quarter-finals"
    elif completed_ko_count < 30:
        next_stage = "Semi-finals"
    elif completed_ko_count < 31:
        next_stage = "Final"
    else:
        next_stage = "Complete"

    # Derive bracket feeds from real fixture data.
    feeds = derive_feeds(knockout_matches)

    return TournamentState(
        group_stage_complete=group_stage_complete,
        played_group=played_group,
        knockout_matches=knockout_matches,
        feeds=feeds,
        bracket_order=bracket_order(feeds),
        r32_slots=r32_slots,
        reached=reached,
        eliminated=eliminated,
        next_stage=next_stage,
    )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _int_or_none(v: object) -> int | None:
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
        if pd.isna(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


_STAGE_RANK = {
    "Group Stage": 0,
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Final": 5,
    "Champion": 6,
}


def _stage_rank(stage: str) -> int:
    return _STAGE_RANK.get(stage, -1)
