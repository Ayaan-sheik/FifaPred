# FIFA World Cup 2026 Prediction Model — Plan

## Goal

Build a working, evaluable World Cup outcome predictor — and a shareable dashboard — that:
1. Produces calibrated win/advancement probabilities for every remaining team, not just single-point picks.
2. Can be shown to a friend during the actual tournament.
3. Is defensible in a job interview as a real, finished, evaluated project (not an abandoned six-service architecture).

## Context / constraint

The 2026 World Cup is live right now. Group stage runs through **June 27**, knockouts start **July 4**, final is **July 19**. The build timeline has to work backward from that calendar — a finished simple system beats an unfinished impressive one. Architecture complexity is staged accordingly: ship the core loop first, only add infrastructure if there's time and appetite left afterward.

---

## Phase 1 — MVP (ship this week, before group stage ends)

**Scope:** one notebook, one model, one tiny app. No database, no separate API service, no CI/CD yet.

| Component | Choice | Why |
|---|---|---|
| Historical data | Kaggle "International football results" dataset + eloratings.net | Free, clean, standard starting point |
| Live results | Manual update or light `api-football.com` free-tier pull | Don't build live infra before the model is validated |
| Model | Dixon-Coles Poisson model (`scipy.optimize`) | Simple, fast to fit, well-validated for football scorelines, no MCMC tuning needed under time pressure |
| Simulation | Monte Carlo bracket simulator (`numpy`, 10k+ draws) | Turns team strengths into round-by-round probabilities |
| Output | JSON file of probabilities, refreshed after each matchday | Decouples model from app — app just reads a file |
| App | Single-file Streamlit app reading the JSON | Fastest path to something shareable; free deploy on Streamlit Community Cloud |

**Day-by-day:**
- **Day 1–2:** Load historical data + current Elo, fit Dixon-Coles model, sanity-check against known strong/weak teams.
- **Day 3:** Build Monte Carlo simulator for the current 48-team bracket state, output probabilities to JSON.
- **Day 4:** Wrap in Streamlit (team strength table, bracket odds, simple chart), deploy.
- **Day 5+:** Update JSON after each matchday as group stage results land; start logging predictions vs. outcomes for evaluation (see below).

**Definition of done for Phase 1:** a public URL you can send to someone, showing live-ish probabilities, backed by a model you can explain in two sentences.

---

## Phase 2 — Production architecture (only after Phase 1 works, time/interest permitting)

This is the version worth doing if Phase 1 ships early and you want the deeper resume story (system design, not just modeling).

```
Data sources (historical + live API)
        |
        v
ETL & feature pipeline (pandas)
        |
        v
PostgreSQL (Neon / Supabase)
        |
        v
Modeling layer
  - Bayesian Poisson model (PyMC, Dixon-Coles)
  - Monte Carlo simulator
        |
        v
FastAPI backend (Dockerized REST endpoints)
        |
        v
Next.js dashboard (Vercel)

CI/CD: GitHub Actions, connected to backend + frontend deploys
```

Key API endpoints (when you get here): `GET /teams/{id}/strength`, `GET /matches/upcoming`, `POST /simulate/bracket`, `GET /predictions/history`.

Upgrade order if pursued: (1) swap JSON file for Postgres, (2) wrap model in FastAPI, (3) move Streamlit app to a proper Next.js frontend, (4) add CI/CD last, once there's something worth automating.

---

## Modeling notes

- Each team gets latent **attack** and **defense** strength parameters; expected goals for team *i* vs team *j* = `exp(attack_i − defense_j + home_advantage)`.
- Weight recent matches more heavily than older ones (time-decay weighting).
- Add a host-nation adjustment (Mexico/Canada/USA) and a confederation-strength adjustment (UEFA vs. CONCACAF vs. AFC aren't directly comparable on raw Elo).
- Re-fit or at least re-run the simulation after every matchday as the live bracket state changes.

## Evaluation — the part most predictors skip

A single tournament is too small a sample to "prove" accuracy, and upsets are common — so the honest, defensible claim is about **calibration**, not raw correctness.

- Backtest the model on 2018 and 2022: log-loss and Brier score against actual results.
- Track Brier score / calibration plots live across 2026 as real results come in — did the model assign reasonable probability to what actually happened?
- Compare pre-match win probabilities against published bookmaker odds as a benchmark, where available.
- Be upfront in writeups: the interesting story is "the model knew what it didn't know," not "the model picked the winner."

## Repo structure (grows into this over both phases)

```
/data        raw + processed datasets
/model       Dixon-Coles fit, simulator, (later) PyMC + MLflow
/app         Streamlit MVP (Phase 1) → /api + /dashboard (Phase 2)
/tests       pytest for simulator + feature pipeline
/.github     CI workflows (Phase 2)
README.md    writeup, methodology, calibration results
```

## Resume talking points (once shipped)

- "Built a Dixon-Coles/Poisson model and Monte Carlo bracket simulator to generate calibrated World Cup outcome probabilities, validated against 2018/2022 via Brier score and log-loss."
- "Tracked live calibration drift across the 2026 group stage and updated the model as results came in."
- (If Phase 2 happens) "Productionized the model behind a FastAPI service with Postgres storage and CI/CD, deployed end-to-end."
