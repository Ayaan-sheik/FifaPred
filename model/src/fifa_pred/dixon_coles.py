"""Dixon-Coles bivariate-Poisson model for international football scorelines.

Each team gets latent attack/defense strengths. Expected goals:

    lambda_home = exp(attack_home - defense_away + gamma * is_home)
    lambda_away = exp(attack_away - defense_home)

where ``gamma`` is a global home advantage (applied only at non-neutral venues).
Scorelines follow independent Poissons with the Dixon-Coles ``tau`` correction
that inflates/deflates the four low-score cells (0-0, 1-0, 0-1, 1-1).

The negative log-likelihood (time-decay weighted) is minimised with L-BFGS-B.
Reference: Dixon & Coles (1997), "Modelling Association Football Scores".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .data_loader import Bracket, load_bracket, load_results


def _tau(lam: np.ndarray, mu: np.ndarray, hg: np.ndarray, ag: np.ndarray,
         rho: float) -> np.ndarray:
    """Dixon-Coles low-score dependence correction (vectorised over matches)."""
    out = np.ones_like(lam)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho
    return out


@dataclass
class DixonColes:
    """Fitted model. Use :meth:`fit` then :meth:`effective_params` / :meth:`score_matrix`."""

    teams: list[str]
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    _idx: dict[str, int] = field(default_factory=dict)

    # ---- fitting -------------------------------------------------------------
    @classmethod
    def fit(cls, df: pd.DataFrame, teams: list[str] | None = None,
            max_iter: int = 200) -> "DixonColes":
        """Fit on a results dataframe (needs home/away team, scores, weight, neutral)."""
        if teams is None:
            teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # keep only matches between known teams
        mask = df["home_team"].isin(idx) & df["away_team"].isin(idx)
        d = df[mask]
        hi = d["home_team"].map(idx).to_numpy()
        ai = d["away_team"].map(idx).to_numpy()
        hg = d["home_score"].to_numpy()
        ag = d["away_score"].to_numpy()
        w = d["weight"].to_numpy()
        at_home = (~d["neutral"].to_numpy()).astype(float)
        lgam = gammaln(hg + 1) + gammaln(ag + 1)  # constant wrt params

        # params: [attack(n), defense(n), gamma, rho]
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

        def neg_ll(x: np.ndarray) -> float:
            atk, dfn = x[:n], x[n:2 * n]
            gamma, rho = x[2 * n], x[2 * n + 1]
            log_lam = atk[hi] - dfn[ai] + gamma * at_home
            log_mu = atk[ai] - dfn[hi]
            lam = np.exp(log_lam)
            mu = np.exp(log_mu)
            tau = _tau(lam, mu, hg, ag, rho)
            tau = np.clip(tau, 1e-9, None)  # guard against negative tau
            ll = (hg * log_lam - lam) + (ag * log_mu - mu) - lgam + np.log(tau)
            penalty = 1e3 * atk.mean() ** 2  # identifiability: mean attack = 0
            return -(w * ll).sum() + penalty

        bounds = [(-3, 3)] * (2 * n) + [(-0.5, 1.0), (-0.2, 0.2)]
        res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter})

        x = res.x
        atk = x[:n] - x[:n].mean()  # recenter so mean attack = 0
        dfn = x[n:2 * n]
        model = cls(teams=list(teams),
                    attack={t: float(atk[i]) for t, i in idx.items()},
                    defense={t: float(dfn[i]) for t, i in idx.items()},
                    home_adv=float(x[2 * n]), rho=float(x[2 * n + 1]), _idx=idx)
        return model

    # ---- post-fit adjustments (host nation + confederation prior) ------------
    def effective_params(
        self,
        bracket: Bracket,
        host_factor: float = 0.5,
        conf_offsets: dict[str, float] | None = None,
    ) -> dict[str, dict]:
        """Return per-team effective attack/defense for the 2026 field, folding in
        a host-nation bump (a fraction of home advantage) and an optional
        confederation offset, per the modelling notes in plan.md.

        Defaults keep confederation offsets at zero (the fit already spans
        confederations via inter-confederation friendlies/tournaments); the hook
        is exposed for tuning.
        """
        conf_offsets = conf_offsets or {}
        bump = host_factor * self.home_adv
        out = {}
        for t in bracket.teams:
            name = t["name"]
            a = self.attack.get(name, 0.0)
            d = self.defense.get(name, 0.0)
            host = bump if t.get("host") else 0.0
            coff = conf_offsets.get(t["confederation"], 0.0)
            eff_a = a + host + coff
            eff_d = d + host  # a large defense param means fewer goals conceded
            out[t["id"]] = {
                "attack": eff_a,
                "defense": eff_d,
                # both params are "higher = better"; overall is their sum
                "overall": eff_a + eff_d,
            }
        return out

    # ---- prediction ----------------------------------------------------------
    def score_matrix(self, atk_a: float, def_a: float, atk_b: float, def_b: float,
                     home: bool = False, max_goals: int = 10) -> np.ndarray:
        """P[i, j] = probability team A scores i and team B scores j (DC-corrected).

        ``home=True`` applies home advantage to team A (for non-neutral games).
        """
        gamma = self.home_adv if home else 0.0
        lam = np.exp(atk_a - def_b + gamma)
        mu = np.exp(atk_b - def_a)
        i = np.arange(max_goals + 1)
        pa = np.exp(i * np.log(lam) - lam - gammaln(i + 1))
        pb = np.exp(i * np.log(mu) - mu - gammaln(i + 1))
        m = np.outer(pa, pb)
        # DC correction on the four low-score cells
        m[0, 0] *= 1.0 - lam * mu * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mu * self.rho
        m[1, 1] *= 1.0 - self.rho
        return m / m.sum()


if __name__ == "__main__":
    bracket = load_bracket()
    df = load_results()
    model = DixonColes.fit(df)
    print(f"home advantage = {model.home_adv:.3f}, rho = {model.rho:.3f}\n")

    eff = model.effective_params(bracket)
    ranked = sorted(bracket.teams, key=lambda t: -eff[t["id"]]["overall"])
    print("Top 8 by overall strength:")
    for t in ranked[:8]:
        e = eff[t["id"]]
        print(f"  {t['name']:15} atk {e['attack']:+.2f}  def {e['defense']:+.2f}  "
              f"overall {e['overall']:+.2f}")
    print("Bottom 4:")
    for t in ranked[-4:]:
        e = eff[t["id"]]
        print(f"  {t['name']:15} atk {e['attack']:+.2f}  def {e['defense']:+.2f}  "
              f"overall {e['overall']:+.2f}")
