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
