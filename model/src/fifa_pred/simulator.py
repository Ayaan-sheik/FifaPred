"""Monte Carlo bracket simulator for the 2026 World Cup (48 teams, 12 groups).

Vectorised over ``n_sims`` with numpy. Each simulation:
  1. plays every group's round-robin, sampling scorelines as independent Poissons;
  2. ranks each group (points -> goal difference -> goals for -> random);
  3. advances the top 2 per group + the 8 best third-placed teams to a 32-team
     single-elimination knockout (random draw, fixed bracket through the rounds);
  4. plays R32 -> R16 -> QF -> SF -> Final, draws broken by a strength-weighted
     coin flip (penalties).

Returns each team's probability of reaching every stage and winning the cup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import Bracket

# round-robin pairings for a 4-team group (indices into the group)
_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def simulate(
    bracket: Bracket,
    eff: dict[str, dict],
    n_sims: int = 20000,
    seed: int = 42,
) -> dict[str, dict]:
    """Run the tournament ``n_sims`` times. ``eff`` maps team id -> {attack, defense}.

    Returns ``{team_id: {p_advance, p_r16, p_qf, p_sf, p_final, p_win}}``.
    """
    rng = np.random.default_rng(seed)
    ids = [t["id"] for t in bracket.teams]
    pos = {tid: i for i, tid in enumerate(ids)}
    n_teams = len(ids)

    atk = np.array([eff[t]["attack"] for t in ids])
    dfn = np.array([eff[t]["defense"] for t in ids])
    overall = np.array([eff[t]["overall"] for t in ids])

    def play(a: np.ndarray, b: np.ndarray):
        """Sample one match per row between global team indices a and b."""
        lam = np.exp(atk[a] - dfn[b])
        mu = np.exp(atk[b] - dfn[a])
        ga = rng.poisson(lam)
        gb = rng.poisson(mu)
        return ga, gb

    # ---- group stage ---------------------------------------------------------
    group_labels = list(bracket.groups.keys())
    winners = np.empty((n_sims, len(group_labels)), dtype=int)
    runners = np.empty((n_sims, len(group_labels)), dtype=int)
    thirds = np.empty((n_sims, len(group_labels)), dtype=int)
    thirds_key = np.empty((n_sims, len(group_labels)))

    for gi, g in enumerate(group_labels):
        members = [pos[t] for t in bracket.groups[g]]  # 4 global indices
        gmem = np.array(members)
        pts = np.zeros((n_sims, 4))
        gf = np.zeros((n_sims, 4))
        ga_tot = np.zeros((n_sims, 4))
        for i, j in _PAIRS:
            gi_, gj_ = play(np.full(n_sims, gmem[i]), np.full(n_sims, gmem[j]))
            pts[:, i] += np.where(gi_ > gj_, 3, np.where(gi_ == gj_, 1, 0))
            pts[:, j] += np.where(gj_ > gi_, 3, np.where(gi_ == gj_, 1, 0))
            gf[:, i] += gi_; ga_tot[:, i] += gj_
            gf[:, j] += gj_; ga_tot[:, j] += gi_
        gd = gf - ga_tot
        # composite ranking key with a tiny random jitter as final tiebreak
        key = pts * 1e6 + (gd + 100) * 1e3 + gf + rng.random((n_sims, 4)) * 1e-3
        order = np.argsort(-key, axis=1)  # best first
        rows = np.arange(n_sims)
        winners[:, gi] = gmem[order[:, 0]]
        runners[:, gi] = gmem[order[:, 1]]
        thirds[:, gi] = gmem[order[:, 2]]
        thirds_key[:, gi] = key[rows, order[:, 2]]

    # best 8 of the 12 third-placed teams
    third_order = np.argsort(-thirds_key, axis=1)[:, :8]
    rows = np.arange(n_sims)[:, None]
    best_thirds = thirds[rows, third_order]  # (n_sims, 8)

    advancers = np.concatenate([winners, runners, best_thirds], axis=1)  # (n_sims, 32)

    # ---- knockout ------------------------------------------------------------
    # random draw: shuffle the 32 qualifiers into a fixed bracket per simulation
    perm = np.argsort(rng.random(advancers.shape), axis=1)
    bracket_slots = np.take_along_axis(advancers, perm, axis=1)

    tally = {
        "p_advance": np.bincount(advancers.ravel(), minlength=n_teams),
    }
    stage_names = ["p_r16", "p_qf", "p_sf", "p_final", "p_win"]
    current = bracket_slots
    for stage in stage_names:
        a = current[:, 0::2]
        b = current[:, 1::2]
        ga, gb = play(a.ravel(), b.ravel())
        ga = ga.reshape(a.shape); gb = gb.reshape(b.shape)
        a_wins = ga > gb
        draw = ga == gb
        # strength-weighted shootout on draws
        pa = 1.0 / (1.0 + np.exp(-(overall[a] - overall[b])))
        a_wins = a_wins | (draw & (rng.random(a.shape) < pa))
        winner = np.where(a_wins, a, b)
        tally[stage] = np.bincount(winner.ravel(), minlength=n_teams)
        current = winner

    out = {}
    for tid in ids:
        k = pos[tid]
        out[tid] = {
            "p_advance": float(tally["p_advance"][k] / n_sims),
            "p_r16": float(tally["p_r16"][k] / n_sims),
            "p_qf": float(tally["p_qf"][k] / n_sims),
            "p_sf": float(tally["p_sf"][k] / n_sims),
            "p_final": float(tally["p_final"][k] / n_sims),
            "p_win": float(tally["p_win"][k] / n_sims),
        }
    return out


def simulate_from_state(
    bracket: Bracket,
    eff: dict[str, dict],
    state: "TournamentState",  # noqa: F821
    n_sims: int = 20000,
    seed: int = 42,
) -> dict[str, dict]:
    """Simulate the remaining WC2026 bracket conditioned on real completed results.

    Group stage probabilities are set deterministically from ``state.r32_slots``
    (all 32 R32 participants are already known).  Knockout matches are simulated
    round-by-round: completed matches are deterministic; pending matches are
    sampled using the same Poisson play + strength-weighted shootout logic as
    :func:`simulate`.

    Returns ``{team_id: {p_advance, p_r16, p_qf, p_sf, p_final, p_win}}`` —
    same shape and key set as :func:`simulate`.

    Parameters
    ----------
    bracket : Bracket
        The 48-team bracket.
    eff : dict
        Team-id → ``{attack, defense, overall}`` from Dixon-Coles.
    state : TournamentState
        Built by :func:`bracket_state.build_tournament_state`.
    n_sims : int
        Number of Monte Carlo simulations for the remaining matches.
    seed : int
        RNG seed for reproducibility.
    """
    from .bracket_state import FALLBACK_FEEDS, _int_or_none, winner_of as _winner_of

    rng = np.random.default_rng(seed)
    ids = [t["id"] for t in bracket.teams]
    pos = {tid: i for i, tid in enumerate(ids)}
    n_teams = len(ids)

    atk = np.array([eff[t]["attack"] for t in ids])
    dfn = np.array([eff[t]["defense"] for t in ids])
    overall = np.array([eff[t]["overall"] for t in ids])

    def _ko_winner(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Simulate one knockout match per row; returns winner index array."""
        lam = np.exp(atk[a] - dfn[b])
        mu = np.exp(atk[b] - dfn[a])
        ga = rng.poisson(lam)
        gb = rng.poisson(mu)
        a_wins = ga > gb
        draw = ga == gb
        pa = 1.0 / (1.0 + np.exp(-(overall[a] - overall[b])))
        a_wins = a_wins | (draw & (rng.random(a.shape) < pa))
        return np.where(a_wins, a, b)

    # ---- group stage: deterministic from state ----------------------------
    # p_advance = 1.0 for confirmed R32 qualifiers, 0.0 for eliminated.
    p_advance_arr = np.zeros(n_teams)
    for tid in state.r32_slots:
        if tid in pos:
            p_advance_arr[pos[tid]] = 1.0

    # Tallies for each knockout stage.
    tally_r16 = np.zeros(n_teams)
    tally_qf = np.zeros(n_teams)
    tally_sf = np.zeros(n_teams)
    tally_final = np.zeros(n_teams)
    tally_win = np.zeros(n_teams)

    # Build a lookup: match_id → row Series for all knockout matches in the dataset.
    ko_by_id: dict[int, object] = {}
    if not state.knockout_matches.empty:
        for _, row in state.knockout_matches.iterrows():
            ko_by_id[int(row["match_id"])] = row

    # ``sim_winner[mid]`` stores an (n_sims,) int array of global team indices
    # representing the simulated winner of each knockout match.
    sim_winner: dict[int, np.ndarray] = {}

    # Determine participants for each match.
    # For R32 (73–88): participants are known from the dataset home/away team ids.
    # For R16+ : derived from feed map or from fixture if listed.

    def _bid_valid(v: object) -> bool:
        """True if v is a non-null, non-empty bracket team id string."""
        if v is None:
            return False
        try:
            if pd.isna(v):  # type: ignore[arg-type]
                return False
        except (TypeError, ValueError):
            pass
        return bool(v)

    def _participants(mid: int) -> tuple[str | None, str | None]:
        """Return (home_bracket_id, away_bracket_id) from the fixture, or (None,None)."""
        row = ko_by_id.get(mid)
        if row is not None:
            h = row.get("home_bracket_id")
            a = row.get("away_bracket_id")
            if _bid_valid(h) and _bid_valid(a):
                return str(h), str(a)
        return None, None

    for mid in sorted(_KNOCKOUT_IDS_SIM):
        row = ko_by_id.get(mid)
        is_completed = row is not None and str(row.get("status", "")).upper() == "COMPLETED"

        if is_completed:
            # Deterministic: replicate the actual winner n_sims times.
            h_bid_raw = row.get("home_bracket_id")
            a_bid_raw = row.get("away_bracket_id")
            w_ds = _winner_of(row)
            if w_ds is not None and _bid_valid(h_bid_raw) and _bid_valid(a_bid_raw):
                w_ds_int = _int_or_none(w_ds)
                h_ds_int = _int_or_none(row.get("home_team_id"))
                winner_bid = str(h_bid_raw) if w_ds_int == h_ds_int else str(a_bid_raw)
                if winner_bid in pos:
                    sim_winner[mid] = np.full(n_sims, pos[winner_bid], dtype=int)
            continue  # always skip to next match after handling completed

        # Determine participants for pending/not-yet-listed matches.
        # For R16+ (mid >= 89) always derive from feeds — the dataset may list
        # scheduled fixtures with pre-seeding team IDs that contradict the real
        # bracket draw, which would double-count a team that appears both in a
        # dataset row and in a feed-derived slot.
        if mid >= 89:
            h_bid, a_bid = None, None
        else:
            h_bid, a_bid = _participants(mid)
        if h_bid is not None and a_bid is not None:
            # Fixture is listed with known participants.
            if h_bid not in pos or a_bid not in pos:
                continue
            h_arr = np.full(n_sims, pos[h_bid], dtype=int)
            a_arr = np.full(n_sims, pos[a_bid], dtype=int)
        else:
            # Use feed map to derive participants from previous simulated rounds.
            feed = state.feeds.get(mid, FALLBACK_FEEDS.get(mid))
            if feed is None:
                continue
            fh, fa = feed
            if fh not in sim_winner or fa not in sim_winner:
                continue
            h_arr = sim_winner[fh]
            a_arr = sim_winner[fa]

        sim_winner[mid] = _ko_winner(h_arr, a_arr)

    # ---- tally stage probabilities ----------------------------------------
    # R16 tally: winners of matches 73–88.
    for mid in range(73, 89):
        if mid in sim_winner:
            np.add.at(tally_r16, sim_winner[mid], 1)

    # QF tally: winners of matches 89–96.
    for mid in range(89, 97):
        if mid in sim_winner:
            np.add.at(tally_qf, sim_winner[mid], 1)

    # SF tally: winners of matches 97–100.
    for mid in range(97, 101):
        if mid in sim_winner:
            np.add.at(tally_sf, sim_winner[mid], 1)

    # Final tally: winners of SF matches 101–102.
    for mid in range(101, 103):
        if mid in sim_winner:
            np.add.at(tally_final, sim_winner[mid], 1)

    # Champion tally: winner of match 104.
    if 104 in sim_winner:
        np.add.at(tally_win, sim_winner[104], 1)

    out = {}
    for tid in ids:
        k = pos[tid]
        out[tid] = {
            "p_advance": float(p_advance_arr[k]),
            "p_r16":     float(tally_r16[k]  / n_sims),
            "p_qf":      float(tally_qf[k]   / n_sims),
            "p_sf":      float(tally_sf[k]    / n_sims),
            "p_final":   float(tally_final[k] / n_sims),
            "p_win":     float(tally_win[k]   / n_sims),
        }
    return out


# Match IDs to simulate (all knockout matches except third-place 103).
_KNOCKOUT_IDS_SIM = list(range(73, 103)) + [104]


if __name__ == "__main__":
    from .data_loader import load_bracket, load_results
    from .dixon_coles import DixonColes

    bracket = load_bracket()
    model = DixonColes.fit(load_results())
    eff = model.effective_params(bracket)
    probs = simulate(bracket, eff, n_sims=20000)

    ranked = sorted(bracket.by_id, key=lambda t: -probs[t]["p_win"])
    print(f"{'team':16} {'win%':>6} {'final%':>7} {'sf%':>6} {'adv%':>6}")
    for tid in ranked[:12]:
        p = probs[tid]
        name = bracket.by_id[tid]["name"]
        print(f"{name:16} {p['p_win']*100:6.1f} {p['p_final']*100:7.1f} "
              f"{p['p_sf']*100:6.1f} {p['p_advance']*100:6.1f}")
    print(f"\nsum p_win = {sum(probs[t]['p_win'] for t in bracket.by_id):.3f} (expect ~1.0)")
    print(f"sum p_advance = {sum(probs[t]['p_advance'] for t in bracket.by_id):.1f} (expect 32)")
