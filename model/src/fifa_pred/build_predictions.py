"""Orchestrate the full pipeline and write predictions.json (the app's only input).

    load dataset -> fit Dixon-Coles -> compute form -> simulate from state -> backtest -> write JSON

Run:  uv run python -m fifa_pred.build_predictions
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .bracket_state import (
    _STAGE_OF,
    _STAGE_RANK,
    _int_or_none,
    build_tournament_state,
    placeholder_label,
    winner_of as _winner_of,
)
from .data_loader import (BRACKET_JSON, compute_group_standings, load_bracket,
                          load_live_results, load_results, merge_live_results)
from .dixon_coles import DixonColes
from .evaluate import run_backtests
from .simulator import simulate, simulate_from_state
from .tournament_form import FORM_WEIGHT, TeamForm, apply_form, compute_form
from .wc_dataset import (completed_results_frame, load_dataset,
                          save_live_results, unmapped_team_names)

ROOT = Path(__file__).resolve().parents[2]          # model/
REPO = ROOT.parent                                   # FIFA_Pred/
OUT = ROOT / "predictions.json"
WEB_COPY = REPO / "web" / "public" / "predictions.json"


# ---------------------------------------------------------------------------
# Knockout payload (module-level for testability)
# ---------------------------------------------------------------------------

def knockout_payload(
    state: object,
    eff: dict[str, dict],
    model: DixonColes,
) -> list[dict]:
    """Build the list of knockout match dicts for the tournament payload.

    Covers all match ids in 73–102 ∪ {104} (omitting 103, the third-place
    match).  Real rows are taken from ``state.knockout_matches``; synthetic
    placeholder rows are generated for any id missing from that frame.

    Every row in the output has a uniform schema superset::

        match_id, stage, bracket_position, date,
        home_id, away_id,
        home_placeholder, away_placeholder,
        home_score, away_score,
        penalties, result_type, status, winner_id,
        p_home_win

    ``bracket_position`` is the match's left-to-right position (0-15) in the
    real bracket tree (see ``bracket_state.bracket_order``) — rows are sorted
    by stage then by this, NOT by ``match_id`` (fixture/dataset order does not
    correspond to bracket side).

    ``p_home_win`` is a float ∈ (0, 1) for pending matches with both
    participants known; ``None`` for completed matches and placeholder rows.

    ``home_placeholder`` / ``away_placeholder`` are ``None`` when the real
    team id is present, otherwise a string like ``"Winner of R16 #89"``.
    """
    # All knockout ids we want (excluding third-place 103).
    target_ids = set(range(73, 103)) | {104}

    # -------------------------------------------------------------------
    # Pass 1: build real rows and record winners for feed resolution.
    # -------------------------------------------------------------------
    ko_rows: list[dict] = []
    winner_by_mid: dict[int, str] = {}   # match_id → winner bracket_id

    real_mids: set[int] = set()

    for _, row in state.knockout_matches.sort_values("match_id").iterrows():  # type: ignore[attr-defined]
        mid = int(row["match_id"])
        if mid not in target_ids:
            continue
        real_mids.add(mid)

        status = str(row.get("status", "")).upper()
        is_done = status == "COMPLETED"

        # Skip pending R16+ fixtures — dataset may have pre-seeding team IDs
        # that contradict the real bracket draw.  Pass 2 will resolve correct
        # teams from state.feeds (corrected FALLBACK_FEEDS).
        if not is_done and mid >= 89:
            real_mids.discard(mid)
            continue

        w_ds = _winner_of(row) if is_done else None
        home_bid = row.get("home_bracket_id")
        away_bid = row.get("away_bracket_id")

        # Resolve winner bracket id.
        winner_id: str | None = None
        if w_ds is not None:
            h_ds = _int_or_none(row.get("home_team_id"))
            winner_id = (
                str(home_bid) if w_ds == h_ds else str(away_bid)
            ) if home_bid and away_bid else None
            if winner_id:
                winner_by_mid[mid] = winner_id

        pen = None
        if is_done and str(row.get("result_type", "")).upper() == "PENALTIES":
            hp = _int_or_none(row.get("home_penalty_score"))
            ap = _int_or_none(row.get("away_penalty_score"))
            if hp is not None and ap is not None:
                pen = {"home": hp, "away": ap}

        # p_home_win: only for pending listed matches with both known ids.
        p_hw: float | None = None
        h_id = str(home_bid) if home_bid and pd.notna(home_bid) else None
        a_id = str(away_bid) if away_bid and pd.notna(away_bid) else None

        if not is_done and h_id is not None and a_id is not None:
            p_hw = _p_home_win(h_id, a_id, eff, model)

        ko_rows.append({
            "match_id": mid,
            "stage": _STAGE_OF.get(mid, "Knockout"),
            "bracket_position": state.bracket_order.get(mid, mid),  # type: ignore[attr-defined]
            "date": str(row["date"].date()) if pd.notna(row.get("date")) else None,
            "home_id": h_id,
            "away_id": a_id,
            "home_placeholder": None,
            "away_placeholder": None,
            "home_score": _int_or_none(row.get("home_score")) if is_done else None,
            "away_score": _int_or_none(row.get("away_score")) if is_done else None,
            "penalties": pen,
            "result_type": str(row.get("result_type") or "") if is_done else None,
            "status": str(row.get("status", "")),
            "winner_id": winner_id,
            "p_home_win": p_hw,
        })

    # -------------------------------------------------------------------
    # Pass 2: synthetic rows for missing ids.
    # -------------------------------------------------------------------
    feeds: dict[int, tuple[int, int]] = state.feeds  # type: ignore[attr-defined]

    for mid in sorted(target_ids - real_mids):
        feed = feeds.get(mid)
        h_id: str | None = None
        a_id: str | None = None
        h_placeholder: str | None = None
        a_placeholder: str | None = None

        if feed is not None:
            fh, fa = feed
            h_winner = winner_by_mid.get(fh)
            a_winner = winner_by_mid.get(fa)
            if h_winner is not None:
                h_id = h_winner
            else:
                h_placeholder = placeholder_label(fh)
            if a_winner is not None:
                a_id = a_winner
            else:
                a_placeholder = placeholder_label(fa)

        # Compute p_home_win only when both sides are real known ids.
        p_hw: float | None = None
        if h_id is not None and a_id is not None:
            p_hw = _p_home_win(h_id, a_id, eff, model)

        ko_rows.append({
            "match_id": mid,
            "stage": _STAGE_OF.get(mid, "Knockout"),
            "bracket_position": state.bracket_order.get(mid, mid),  # type: ignore[attr-defined]
            "date": None,
            "home_id": h_id,
            "away_id": a_id,
            "home_placeholder": h_placeholder,
            "away_placeholder": a_placeholder,
            "home_score": None,
            "away_score": None,
            "penalties": None,
            "result_type": None,
            "status": "Pending",
            "winner_id": None,
            "p_home_win": p_hw,
        })

    ko_rows.sort(key=lambda r: (_STAGE_RANK.get(r["stage"], 0), r["bracket_position"]))
    return ko_rows


def _p_home_win(
    h_id: str,
    a_id: str,
    eff: dict[str, dict],
    model: DixonColes,
) -> float | None:
    """P(home team wins including shootout) for a knockout match.

    Uses ``score_matrix(..., home=False)`` (all knockout matches neutral),
    then adds P(draw) × sigmoid(overall_h − overall_a) for the shootout.
    Matches the ``_ko_winner`` logic in simulator.py.
    """
    eh = eff.get(h_id)
    ea = eff.get(a_id)
    if eh is None or ea is None:
        return None

    m = model.score_matrix(
        eh["attack"], eh["defense"],
        ea["attack"], ea["defense"],
        home=False,
    )
    p_home_reg = float(np.tril(m, -1).sum())   # home goals > away goals
    p_draw = float(np.diag(m).sum())
    overall_diff = eh["overall"] - ea["overall"]
    p_shootout = 1.0 / (1.0 + math.exp(-overall_diff))

    return p_home_reg + p_draw * p_shootout


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build(n_sims: int = 20000, with_backtest: bool = True, scrape: bool = True) -> dict:
    bracket = load_bracket()
    df = load_results()

    # ---- WC2026 dataset: fetch, convert, save, merge -------------------------
    ds = None
    state = None
    live = pd.DataFrame()

    if scrape:
        print("Fetching WC2026 dataset …")
        try:
            ds = load_dataset()
            unmapped = unmapped_team_names(ds, bracket)
            if unmapped:
                warnings.warn(
                    f"[build] Unmapped team names in dataset: {unmapped}; "
                    "these matches will be excluded from training data.",
                    stacklevel=1,
                )

            live = completed_results_frame(ds)
            save_live_results(live)

            # Build tournament state for conditioned simulation.
            print("Building tournament state …")
            state = build_tournament_state(ds, bracket)
            print(
                f"  group_stage_complete={state.group_stage_complete}, "
                f"R32 slots={len(state.r32_slots)}, "
                f"eliminated={len(state.eliminated)}, "
                f"next_stage={state.next_stage}"
            )
        except Exception as exc:
            print(f"  [build] Dataset fetch/parse failed: {exc}")
            print("  [build] Falling back to cached wc2026_live.csv …")
            live = load_live_results()

    else:
        live = load_live_results()

    if not live.empty:
        df = merge_live_results(df, live)
        print(f"  Merged {len(live)} live WC2026 result(s) into training data.")

    # Use the latest live WC2026 match date when available.
    last_match = live["date"].max() if not live.empty else df["date"].max()

    print("Fitting Dixon-Coles model …")
    model = DixonColes.fit(df)
    eff = model.effective_params(bracket)

    # ---- Tournament form (opponent-adjusted xG residuals) --------------------
    form: dict[str, TeamForm] = {}
    if ds is not None:
        try:
            form = compute_form(ds, bracket, model)
            print(f"  Tournament form computed for {len(form)} teams.")
        except Exception as exc:
            warnings.warn(
                f"[build] compute_form failed ({exc}); proceeding without form.",
                stacklevel=1,
            )
            form = {}
    eff = apply_form(eff, form)

    print(f"Simulating tournament ({n_sims:,} runs) …")
    if state is not None:
        probs = simulate_from_state(bracket, eff, state, n_sims=n_sims)
    else:
        warnings.warn(
            "[build] No tournament state available; falling back to plain simulate() — "
            "eliminated teams will still receive non-zero odds.",
            stacklevel=1,
        )
        probs = simulate(bracket, eff, n_sims=n_sims)

    teams = []
    for t in bracket.teams:
        tid = t["id"]
        e = eff[tid]
        p = probs[tid]
        is_eliminated = state is not None and tid in state.eliminated
        reached_stage = (state.reached.get(tid, "") if state is not None else "")

        # Form info for this team.
        tf: TeamForm | None = form.get(tid)
        stats_block: dict | None = None
        if tf is not None:
            n_played = tf.played
            xg_for_pm  = round(tf.xg_for  / n_played, 4) if n_played > 0 else None
            xg_ag_pm   = round(tf.xg_against / n_played, 4) if n_played > 0 else None
            xg_diff_pm = round((tf.xg_for - tf.xg_against) / n_played, 4) if n_played > 0 else None
            stats_block = {
                "played":       n_played,
                "gf":           round(tf.gf, 1),
                "ga":           round(tf.ga, 1),
                "xg_for_pm":    xg_for_pm,
                "xg_against_pm": xg_ag_pm,
                "xg_diff_pm":   xg_diff_pm,
                "possession_avg": round(tf.possession_avg, 2) if tf.possession_avg is not None else None,
                "shots_avg":    round(tf.shots_avg, 2) if tf.shots_avg is not None else None,
            }

        teams.append({
            "id": tid, "name": t["name"], "confederation": t["confederation"],
            "flag": t["flag"], "group": t["group"], "host": t.get("host", False),
            "attack":       round(e["attack"], 4),
            "defense":      round(e["defense"], 4),
            "overall":      round(e["overall"], 4),
            "base_attack":  round(e["base_attack"], 4),
            "base_defense": round(e["base_defense"], 4),
            "base_overall": round(e["base_overall"], 4),
            "form":         round(tf.form_rating,  4) if tf is not None else 0.0,
            "form_attack":  round(tf.form_attack,  4) if tf is not None else 0.0,
            "form_defense": round(tf.form_defense, 4) if tf is not None else 0.0,
            "stats":        stats_block,
            **{k: round(v, 4) for k, v in p.items()},
            "eliminated": is_eliminated,
            "reached": reached_stage,
        })
    teams.sort(key=lambda x: -x["p_win"])

    # ---- group standings (computed from live results) -------------------------
    standings = compute_group_standings(live if not live.empty else pd.DataFrame(), bracket)
    src = "computed from WC2026 dataset" if standings else "no live data yet"
    print(f"  Group standings source: {src}")

    # ---- tournament payload --------------------------------------------------
    tournament_payload: dict = {}
    if state is not None:
        ko_rows = knockout_payload(state, eff, model)

        completed_count = sum(1 for r in ko_rows if r["status"].upper() == "COMPLETED")
        tournament_payload = {
            "current_stage": state.next_stage,
            "group_stage_complete": state.group_stage_complete,
            "matches_completed": len(state.played_group) + completed_count,
            "knockout": ko_rows,
        }

    calibration = {"backtest": {}, "reliability": []}
    if with_backtest:
        print("Backtesting on 2018 & 2022 World Cups …")
        calibration = run_backtests()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through": str(last_match.date()),
        "model": {
            "name": "Dixon-Coles Poisson + Monte Carlo (mominullptr WC2026 dataset)",
            "n_sims": n_sims,
            "home_advantage": round(model.home_adv, 4),
            "rho": round(model.rho, 4),
            "n_matches_trained": int(len(df)),
            "form_weight": FORM_WEIGHT,
            "form_applied": bool(form),
        },
        "format": bracket.fmt,
        "groups": bracket.groups,
        "standings": standings,
        "teams": teams,
        "calibration": calibration,
        "tournament": tournament_payload,
        "disclaimer": (
            "Predictions from a Dixon-Coles model trained on historical results plus "
            "live WC2026 data from the mominullptr/FIFA-World-Cup-2026-Dataset "
            "(https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset). "
            "Eliminated teams have p_win=0 by construction."
        ),
    }
    return payload


def main():
    payload = build()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}")
    if WEB_COPY.parent.exists():
        shutil.copy(OUT, WEB_COPY)
        print(f"Copied -> {WEB_COPY}")
    else:
        print(f"(web app not scaffolded; copy {OUT.name} to web/public/ later)")

    top = payload["teams"][:8]
    print("\nFavourites (still alive):")
    for t in top:
        alive = "" if t.get("eliminated") else " *"
        print(f"  {t['flag']} {t['name']:16} win {t['p_win']*100:4.1f}%{alive}")
    bt = payload["calibration"].get("backtest", {})
    for yr, m in bt.items():
        print(f"  WC{yr} backtest: Brier {m['brier']}, log-loss {m['log_loss']}")
    p_win_sum = sum(t["p_win"] for t in payload["teams"])
    print(f"  sum(p_win) = {p_win_sum:.6f} (expect ~1.0)")


if __name__ == "__main__":
    # ensure the bracket exists before building
    if not BRACKET_JSON.exists():
        raise SystemExit(f"Missing {BRACKET_JSON}; run data/sync_bracket.py first.")
    main()
