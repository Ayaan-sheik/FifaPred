"""Orchestrate the full pipeline and write predictions.json (the app's only input).

    load data -> fit Dixon-Coles -> Monte Carlo simulate -> backtest -> write JSON

Run:  uv run python -m fifa_pred.build_predictions
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_loader import (BRACKET_JSON, compute_group_standings, load_bracket,
                          load_results, merge_live_results)
from .dixon_coles import DixonColes
from .evaluate import run_backtests
from .scraper import (fetch_group_standings_api, fetch_wc2026_results,
                      load_live_results, save_live_results)
from .simulator import simulate

ROOT = Path(__file__).resolve().parents[2]          # model/
REPO = ROOT.parent                                   # FIFA_Pred/
OUT = ROOT / "predictions.json"
WEB_COPY = REPO / "web" / "public" / "predictions.json"


def build(n_sims: int = 20000, with_backtest: bool = True, scrape: bool = True) -> dict:
    bracket = load_bracket()
    df = load_results()

    # ---- live results: scrape then merge into training data ------------------
    if scrape:
        print("Fetching live WC2026 results...")
        results = fetch_wc2026_results()
        if results:
            save_live_results(results)
    live = load_live_results()
    if not live.empty:
        df = merge_live_results(df, live)
        print(f"  Merged {len(live)} live WC2026 result(s) into training data.")

    # Use the latest live WC2026 match date when available (more informative than
    # the historical CSV date for the "data through" display in the webapp).
    last_match = live["date"].max() if not live.empty else df["date"].max()

    print("Fitting Dixon-Coles model...")
    model = DixonColes.fit(df)
    eff = model.effective_params(bracket)

    print(f"Simulating tournament ({n_sims:,} runs)...")
    probs = simulate(bracket, eff, n_sims=n_sims)

    teams = []
    for t in bracket.teams:
        tid = t["id"]
        e = eff[tid]
        p = probs[tid]
        teams.append({
            "id": tid, "name": t["name"], "confederation": t["confederation"],
            "flag": t["flag"], "group": t["group"], "host": t.get("host", False),
            "attack": round(e["attack"], 4),
            "defense": round(e["defense"], 4),
            "overall": round(e["overall"], 4),
            **{k: round(v, 4) for k, v in p.items()},
        })
    teams.sort(key=lambda x: -x["p_win"])

    # ---- group standings: try API first (richer), fall back to computed ------
    standings = fetch_group_standings_api() if scrape else {}
    if not standings:
        standings = compute_group_standings(live if not live.empty else pd.DataFrame(), bracket)
        if standings:
            src = "computed from scraped match results"
        else:
            src = "no live data yet"
    else:
        src = "external API"
        # back-fill flag emojis from our bracket data (external APIs don't return them)
        name_to_flag = {t["name"]: t["flag"] for t in bracket.teams}
        for rows in standings.values():
            for row in rows:
                if not row.get("flag"):
                    row["flag"] = name_to_flag.get(row["name"], "")
    print(f"  Group standings source: {src}")

    calibration = {"backtest": {}, "reliability": []}
    if with_backtest:
        print("Backtesting on 2018 & 2022 World Cups...")
        calibration = run_backtests()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_through": str(last_match.date()),
        "model": {
            "name": "Dixon-Coles Poisson + Monte Carlo",
            "n_sims": n_sims,
            "home_advantage": round(model.home_adv, 4),
            "rho": round(model.rho, 4),
            "n_matches_trained": int(len(df)),
        },
        "format": bracket.fmt,
        "groups": bracket.groups,
        "standings": standings,
        "teams": teams,
        "calibration": calibration,
        "disclaimer": "Predictions from a Dixon-Coles model trained on historical "
                      "results plus live WC2026 data scraped from worldcup26.ir.",
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
        print(f"(web app not scaffolded yet; copy {OUT.name} to web/public/ later)")

    top = payload["teams"][:5]
    print("\nFavourites:")
    for t in top:
        print(f"  {t['flag']} {t['name']:14} win {t['p_win']*100:4.1f}%")
    bt = payload["calibration"].get("backtest", {})
    for yr, m in bt.items():
        print(f"  WC{yr} backtest: Brier {m['brier']}, log-loss {m['log_loss']}")


if __name__ == "__main__":
    # ensure the bracket exists before building
    if not BRACKET_JSON.exists():
        raise SystemExit(f"Missing {BRACKET_JSON}; run data/make_bracket.py first.")
    main()
