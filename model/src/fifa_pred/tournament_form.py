"""Opponent-adjusted tournament form via Dixon-Coles xG residuals.

For each completed match with non-null xG and both teams in the fitted bracket,
we compare actual xG to the Dixon-Coles expectation on the log scale:

    lam_for     = exp(atk[team] - def[opponent] + gamma * at_home_team)
    lam_against = exp(atk[opponent] - def[team] + gamma * at_home_opp)
    f_atk(team) = log((xg_for     + EPS) / (lam_for     + EPS))
    f_def(team) = log((lam_against + EPS) / (xg_against  + EPS))   # positive = conceded less

This opponent-adjusts by construction: the same raw xG against a strong
defensive opponent earns a higher f_atk residual than the same xG against a
weak one.

Per team across n matches (≤7 in a World Cup):
    shrink       = n / (n + K)
    form_attack  = clip(FORM_WEIGHT * shrink * mean(f_atk), -CAP, +CAP)
    form_defense = clip(FORM_WEIGHT * shrink * mean(f_def), -CAP, +CAP)
    form_rating  = shrink * (mean(f_atk) + mean(f_def))   # display, uncapped

Constants:  FORM_WEIGHT=0.4, CAP=0.35, K=4.0, EPS=0.25
Equal match weights (no recency decay — ≤7 matches is too few to tune).

NOTE: form_attack and form_defense intentionally double-count the process
signal on top of the score-level DC fit (which already encodes the same xG
indirectly via goals).  The CAP bound (≡ ±0.35 log-goals ≈ ±42% rate
change) keeps movements bounded.  Backtests (evaluate.py) never call
apply_form(), so the blending cannot leak into historical calibration by
construction.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORM_WEIGHT: float = 0.4
CAP: float = 0.35
K: float = 4.0
EPS: float = 0.25


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class TeamForm:
    """Per-team form computed from tournament xG residuals.

    Attributes
    ----------
    played : int
        Number of matches included in the form calculation (non-null xG, both
        teams fitted).
    gf : float
        Total goals scored across all tournament matches.
    ga : float
        Total goals conceded across all tournament matches.
    xg_for : float
        Total xG produced across included matches.
    xg_against : float
        Total xG conceded across included matches.
    form_attack : float
        Opponent-adjusted attack bump (clipped to ±CAP).
    form_defense : float
        Opponent-adjusted defense bump (clipped to ±CAP).
    form_rating : float
        Combined form signal (uncapped, shrinkage-weighted) for display.
    possession_avg : float | None
        Average possession % from match_team_stats; None if unavailable.
    shots_avg : float | None
        Average total shots from match_team_stats; None if unavailable.
    """

    played: int
    gf: float
    ga: float
    xg_for: float
    xg_against: float
    form_attack: float
    form_defense: float
    form_rating: float
    possession_avg: float | None
    shots_avg: float | None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_form(
    ds: object,
    bracket: object,
    model: object,
) -> dict[str, TeamForm]:
    """Compute opponent-adjusted tournament form for all fitted teams.

    Parameters
    ----------
    ds : Dataset
        The WC2026 dataset (from ``wc_dataset.load_dataset``).
    bracket : Bracket
        The tournament bracket (from ``data_loader.load_bracket``).
    model : DixonColes
        Fitted Dixon-Coles model.

    Returns
    -------
    dict[str, TeamForm]
        Keyed by bracket team id (FIFA code).  Empty dict if xG is entirely
        missing or all-NaN.
    """
    matches: pd.DataFrame = ds.matches  # type: ignore[attr-defined]

    # Quick exit: if xG columns are absent, return empty.
    if "home_xg" not in matches.columns or "away_xg" not in matches.columns:
        warnings.warn(
            "[tournament_form] home_xg / away_xg columns absent from dataset; "
            "returning empty form.",
            stacklevel=2,
        )
        return {}

    # Completed matches only.
    completed = matches[matches["status"].str.upper() == "COMPLETED"].copy()
    if completed.empty or completed["home_xg"].isna().all():
        warnings.warn(
            "[tournament_form] No completed matches with xG data; returning empty form.",
            stacklevel=2,
        )
        return {}

    # Build bracket id lookups.
    name_to_id: dict[str, str] = bracket.name_to_id  # type: ignore[attr-defined]
    eff = model.effective_params(bracket)  # type: ignore[attr-defined]
    gamma: float = model.home_adv  # type: ignore[attr-defined]

    # Build dataset team_id → FIFA code (bracket id) and code lookup.
    code_map: dict[int, str] = dict(
        zip(ds.teams["team_id"], ds.teams["fifa_code"])  # type: ignore[attr-defined]
    )

    # Per-team accumulators: {bracket_id: [f_atk, ...], ...}
    atk_residuals: dict[str, list[float]] = {}
    def_residuals: dict[str, list[float]] = {}
    goals_for: dict[str, float] = {}
    goals_against: dict[str, float] = {}
    xg_for_acc: dict[str, float] = {}
    xg_against_acc: dict[str, float] = {}

    for _, row in completed.iterrows():
        xg_h = row.get("home_xg")
        xg_a = row.get("away_xg")
        if xg_h is None or xg_a is None:
            continue
        try:
            xg_h = float(xg_h)
            xg_a = float(xg_a)
        except (TypeError, ValueError):
            continue
        if math.isnan(xg_h) or math.isnan(xg_a):
            continue

        # Map team names to bracket ids.
        home_name = row.get("home_team_name", "")
        away_name = row.get("away_team_name", "")
        home_bid = name_to_id.get(str(home_name))
        away_bid = name_to_id.get(str(away_name))

        if home_bid is None or away_bid is None:
            continue  # at least one team not in fitted bracket
        if home_bid not in eff or away_bid not in eff:
            continue

        # Determine home advantage: same logic as completed_results_frame.
        venue_country = str(row.get("venue_country", ""))
        h_code = code_map.get(int(row.get("home_team_id", 0)), "")
        a_code = code_map.get(int(row.get("away_team_id", 0)), "")

        if h_code == venue_country:
            at_home_h, at_home_a = 1.0, 0.0
        elif a_code == venue_country:
            at_home_h, at_home_a = 0.0, 1.0
        else:
            at_home_h, at_home_a = 0.0, 0.0

        atk_h = eff[home_bid]["attack"]
        def_h = eff[home_bid]["defense"]
        atk_a = eff[away_bid]["attack"]
        def_a = eff[away_bid]["defense"]

        # Dixon-Coles expected goals (no rho correction — mean-rate only).
        lam_h = math.exp(atk_h - def_a + gamma * at_home_h)
        lam_a = math.exp(atk_a - def_h + gamma * at_home_a)

        # Log-ratio residuals.
        f_atk_h = math.log((xg_h + EPS) / (lam_h + EPS))
        f_def_h = math.log((lam_a + EPS) / (xg_a + EPS))
        f_atk_a = math.log((xg_a + EPS) / (lam_a + EPS))
        f_def_a = math.log((lam_h + EPS) / (xg_h + EPS))

        # Accumulate.
        atk_residuals.setdefault(home_bid, []).append(f_atk_h)
        def_residuals.setdefault(home_bid, []).append(f_def_h)
        atk_residuals.setdefault(away_bid, []).append(f_atk_a)
        def_residuals.setdefault(away_bid, []).append(f_def_a)

        hs = float(row.get("home_score", 0) or 0)
        as_ = float(row.get("away_score", 0) or 0)
        goals_for[home_bid] = goals_for.get(home_bid, 0.0) + hs
        goals_against[home_bid] = goals_against.get(home_bid, 0.0) + as_
        goals_for[away_bid] = goals_for.get(away_bid, 0.0) + as_
        goals_against[away_bid] = goals_against.get(away_bid, 0.0) + hs

        xg_for_acc[home_bid] = xg_for_acc.get(home_bid, 0.0) + xg_h
        xg_against_acc[home_bid] = xg_against_acc.get(home_bid, 0.0) + xg_a
        xg_for_acc[away_bid] = xg_for_acc.get(away_bid, 0.0) + xg_a
        xg_against_acc[away_bid] = xg_against_acc.get(away_bid, 0.0) + xg_h

    if not atk_residuals:
        return {}

    # Compute possession/shots averages from ds.team_stats if available.
    poss_avg: dict[str, float] = {}
    shots_avg_map: dict[str, float] = {}
    if ds.team_stats is not None:  # type: ignore[attr-defined]
        ts: pd.DataFrame = ds.team_stats  # type: ignore[attr-defined]
        # Map dataset team_id → bracket id.
        for ds_id, bid in code_map.items():
            subset = ts[ts["team_id"] == ds_id]
            if subset.empty:
                continue
            if "possession_pct" in subset.columns and subset["possession_pct"].notna().any():
                poss_avg[bid] = float(subset["possession_pct"].mean())
            if "total_shots" in subset.columns and subset["total_shots"].notna().any():
                shots_avg_map[bid] = float(subset["total_shots"].mean())

    # Build TeamForm per bracket id.
    result: dict[str, TeamForm] = {}
    for bid, f_atk_list in atk_residuals.items():
        f_def_list = def_residuals.get(bid, [])
        n = len(f_atk_list)
        shrink = n / (n + K)

        mean_atk = sum(f_atk_list) / n
        mean_def = sum(f_def_list) / len(f_def_list) if f_def_list else 0.0

        fa = max(-CAP, min(CAP, FORM_WEIGHT * shrink * mean_atk))
        fd = max(-CAP, min(CAP, FORM_WEIGHT * shrink * mean_def))
        fr = shrink * (mean_atk + mean_def)

        result[bid] = TeamForm(
            played=n,
            gf=goals_for.get(bid, 0.0),
            ga=goals_against.get(bid, 0.0),
            xg_for=xg_for_acc.get(bid, 0.0),
            xg_against=xg_against_acc.get(bid, 0.0),
            form_attack=fa,
            form_defense=fd,
            form_rating=fr,
            possession_avg=poss_avg.get(bid),
            shots_avg=shots_avg_map.get(bid),
        )

    return result


# ---------------------------------------------------------------------------
# Apply form to effective params
# ---------------------------------------------------------------------------

def apply_form(
    eff: dict[str, dict],
    form: dict[str, "TeamForm"],
) -> dict[str, dict]:
    """Return a new eff dict with form bumps applied.

    For every team in *eff*, adds ``base_attack``, ``base_defense``, and
    ``base_overall`` (copies of the un-adjusted values).  Then replaces
    ``attack``, ``defense``, and ``overall`` with base + form bump.

    Teams absent from *form* receive zero bumps (passthrough), but still get
    the ``base_*`` keys for schema consistency.

    Parameters
    ----------
    eff : dict[str, dict]
        As returned by ``DixonColes.effective_params``.
    form : dict[str, TeamForm]
        As returned by ``compute_form``.

    Returns
    -------
    dict[str, dict]
        New dict; original *eff* is not mutated.
    """
    out: dict[str, dict] = {}
    for tid, e in eff.items():
        base_a = e["attack"]
        base_d = e["defense"]
        base_o = e["overall"]

        tf = form.get(tid)
        fa = tf.form_attack if tf is not None else 0.0
        fd = tf.form_defense if tf is not None else 0.0

        new_a = base_a + fa
        new_d = base_d + fd
        new_o = new_a + new_d

        out[tid] = {
            **e,
            "base_attack": base_a,
            "base_defense": base_d,
            "base_overall": base_o,
            "attack": new_a,
            "defense": new_d,
            "overall": new_o,
        }
    return out
