"""Calibration backtest: how well-calibrated were the model's match probabilities?

We refit the model using only data available *before* a past World Cup, predict
the win/draw/loss probabilities for each of that tournament's matches, and score
them with multiclass Brier and log-loss. We also bin predicted vs observed
frequencies into a reliability curve. This is the honest, defensible claim from
plan.md: the interesting story is calibration, not picking the winner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import load_results
from .dixon_coles import DixonColes

# World Cup windows: fit on everything strictly before `cutoff`.
WORLD_CUPS = {
    "2018": {"cutoff": "2018-06-01", "start": "2018-06-13", "end": "2018-07-16"},
    "2022": {"cutoff": "2022-11-01", "start": "2022-11-19", "end": "2022-12-19"},
}


def _match_probs(model: DixonColes, home: str, away: str) -> tuple[float, float, float]:
    """(P home win, P draw, P away win) for a neutral-site match."""
    a = model.attack.get(home, 0.0); da = model.defense.get(home, 0.0)
    b = model.attack.get(away, 0.0); db = model.defense.get(away, 0.0)
    m = model.score_matrix(a, da, b, db, home=False)
    p_home = np.tril(m, -1).sum()  # home goals > away goals
    p_away = np.triu(m, 1).sum()
    p_draw = np.trace(m)
    return float(p_home), float(p_draw), float(p_away)


def backtest_world_cup(year: str, xi: float = 0.30):
    """Return (metrics dict, list of (pred_prob, outcome_indicator)) for one cup."""
    cfg = WORLD_CUPS[year]
    train = load_results(since="2006-01-01", until=cfg["cutoff"], xi=xi,
                         reference_date=cfg["cutoff"])
    model = DixonColes.fit(train)

    matches = load_results(since=cfg["start"], xi=xi)
    matches = matches[(matches["date"] <= pd.Timestamp(cfg["end"])) &
                      (matches["tournament"].str.contains("World Cup", case=False))]

    briers, loglosses, points = [], [], []
    known = set(model.attack)
    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        if h not in known or a not in known:
            continue
        ph, pd_, pa = _match_probs(model, h, a)
        probs = np.array([ph, pd_, pa])
        if row["home_score"] > row["away_score"]:
            y = np.array([1, 0, 0])
        elif row["home_score"] == row["away_score"]:
            y = np.array([0, 1, 0])
        else:
            y = np.array([0, 0, 1])
        briers.append(float(((probs - y) ** 2).sum()))
        loglosses.append(float(-np.log(max(probs[y == 1][0], 1e-12))))
        for p, ind in zip(probs, y):
            points.append((float(p), int(ind)))

    metrics = {
        "brier": round(float(np.mean(briers)), 4),
        "log_loss": round(float(np.mean(loglosses)), 4),
        "n_matches": len(briers),
    }
    return metrics, points


def reliability_curve(points, n_bins: int = 10):
    """Bin (predicted prob, outcome) pairs into a reliability curve."""
    if not points:
        return []
    arr = np.array(points)
    p, y = arr[:, 0], arr[:, 1]
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        out.append({
            "pred": round(float(p[m].mean()), 4),
            "observed": round(float(y[m].mean()), 4),
            "n": int(m.sum()),
        })
    return out


def run_backtests(xi: float = 0.30) -> dict:
    """Backtest every configured World Cup; pool points for one reliability curve."""
    backtest = {}
    pooled = []
    for year in WORLD_CUPS:
        metrics, points = backtest_world_cup(year, xi=xi)
        backtest[year] = metrics
        pooled.extend(points)
    return {"backtest": backtest, "reliability": reliability_curve(pooled)}


if __name__ == "__main__":
    result = run_backtests()
    for year, m in result["backtest"].items():
        print(f"WC{year}: Brier {m['brier']:.3f}  log-loss {m['log_loss']:.3f}  "
              f"({m['n_matches']} matches)")
    print("\nreliability (pred -> observed):")
    for b in result["reliability"]:
        print(f"  {b['pred']:.2f} -> {b['observed']:.2f}  (n={b['n']})")
