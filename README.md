# WC26 Predictor — Calibrated World Cup 2026 Probabilities

A Dixon-Coles Poisson model and Monte Carlo bracket simulator that produce
**calibrated** win and advancement probabilities for every team at the 2026
World Cup — surfaced through an animated Next.js dashboard.

The methodology in two sentences: each national team gets latent **attack** and
**defense** strengths fitted (with time-decay weighting and a Dixon-Coles
low-score correction) from ~12,000 international match results; expected goals
for a match are `exp(attack_i − defense_j + home_advantage)`. Those strengths
are run through **20,000 Monte Carlo tournaments** — group stage, best-third
tiebreaks, and a 32-team knockout — to turn team strength into round-by-round
probabilities.

```
model (Python)  ──►  predictions.json  ──►  web (Next.js dashboard)
Dixon-Coles + MC      single static file     animated, no backend
```

## Project layout

| Path | What |
|---|---|
| `model/` | Python prediction engine (`uv`-managed) |
| `model/src/fifa_pred/` | `data_loader`, `dixon_coles`, `simulator`, `evaluate`, `build_predictions` |
| `model/data/wc2026.json` | 48-team field & groups (**hand-seeded — verify vs the official draw**) |
| `model/predictions.json` | model output (copied into `web/public/`) |
| `web/` | Next.js + Tailwind + Framer Motion dashboard |

## The dashboard

- **Overview** — favourites, headline stats, animated hero.
- **Strengths** — sortable, filterable table of attack/defense/overall ratings.
- **Bracket** — per-group advancement odds for every stage (R32 → 🏆).
- **Predictor** — pick two teams; a Poisson scoreline heatmap and W/D/L odds are
  computed live in the browser from the exported model parameters.
- **Calibration** — 2018 & 2022 backtest (Brier, log-loss) and a reliability
  curve. The honest claim is calibration, not picking the winner.

### Calibration results (backtest)

| Tournament | Brier | Log-loss | Baseline (uniform) |
|---|---|---|---|
| World Cup 2018 | ~0.60 | ~1.00 | 0.667 / 1.099 |
| World Cup 2022 | ~0.64 | ~1.09 | 0.667 / 1.099 |

Both beat the coin-toss baseline, i.e. the probabilities carried real
information. (Re-run `evaluate` for exact current numbers.)

## Run it

**1. Regenerate predictions** (downloads data, fits, simulates, backtests; ~1 min):

```bash
cd model
uv sync
uv pip install -e .
uv run python data/make_bracket.py          # (re)generate the team field
uv run python -m fifa_pred.build_predictions # writes predictions.json + copies to web/public
uv run pytest                                # simulator sanity tests
```

**2. Run the dashboard:**

```bash
cd web
npm install
npm run dev        # http://localhost:3000
# or: npm run build && npm run start
```

The web app reads only `web/public/predictions.json`, so it deploys as a static
site (e.g. Vercel) with no backend.

## Notes

- `model/data/wc2026.json` is a **plausible** 2026 field/draw, not the official
  one — verify the qualified teams and group assignments before sharing.
- Model: Dixon & Coles (1997). Data: the martj42 "International football
  results" dataset.
