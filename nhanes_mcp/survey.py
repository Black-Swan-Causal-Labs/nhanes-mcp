"""Design-based estimation for NHANES (Taylor-series linearization).

Implements what NCHS analysts do in SUDAAN / R `survey`:
  * stratified, with-replacement PSU variance (SDMVSTRA / SDMVPSU)
  * domain (subpopulation) estimation WITHOUT subsetting the design
  * direct age standardization with linearized variance
  * Korn-Graubard CIs for proportions + NCHS (2017) reliability flags
  * survey-weighted linear / logistic regression with sandwich variance
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

# 2000 projected U.S. standard population, NCHS adult age groups.
# Derived from Census P25-1130: 20-39 = 77,670k; 40-59 = 72,816k; 60+ = 45,364k.
AGE_STANDARDS = {
    "nchs_adults_20plus": {
        "age_var": "RIDAGEYR",
        "groups": [(20, 39), (40, 59), (60, 200)],
        "proportions": [0.3966, 0.3718, 0.2316],
        "note": "NCHS 3-group adult standard (2000 projected census): 20-39, 40-59, 60+",
    },
}


@dataclass
class Design:
    weight: np.ndarray
    strata: np.ndarray
    psu: np.ndarray

    @property
    def n_psu(self) -> int:
        return len(pd.unique(pd.Series(list(zip(self.strata, self.psu)))))

    @property
    def n_strata(self) -> int:
        return len(pd.unique(self.strata))

    @property
    def df(self) -> int:
        return self.n_psu - self.n_strata


def design_from_frame(df: pd.DataFrame, weight: str, strata: str = "SDMVSTRA_U", psu: str = "SDMVPSU") -> Design:
    w = df[weight].fillna(0).to_numpy(dtype=float)
    return Design(w, df[strata].to_numpy(), df[psu].to_numpy())


def _psu_totals(scores: np.ndarray, d: Design):
    """Return list of (stratum, matrix of PSU totals) for score array (n,) or (n,k)."""
    s = scores if scores.ndim == 2 else scores[:, None]
    frame = pd.DataFrame(s)
    frame["_h"] = d.strata
    frame["_i"] = d.psu
    tot = frame.groupby(["_h", "_i"], sort=True).sum()
    return tot


def linearized_cov(scores: np.ndarray, d: Design, lonely: str = "adjust") -> np.ndarray:
    """Stratified with-replacement variance of the total of `scores`.

    lonely='adjust' mimics R survey options(survey.lonely.psu='adjust'):
    a singleton stratum is centered at the grand mean of PSU totals.
    """
    s = scores if scores.ndim == 2 else scores[:, None]
    tot = _psu_totals(s, d)
    k = s.shape[1]
    V = np.zeros((k, k))
    grand = tot.to_numpy().mean(axis=0)
    for _, g in tot.groupby(level=0):
        m = g.to_numpy()
        n_h = m.shape[0]
        if n_h >= 2:
            c = m - m.mean(axis=0)
            V += n_h / (n_h - 1) * c.T @ c
        elif lonely == "adjust":
            c = m - grand
            V += c.T @ c
    return V


def domain_ratio(y: np.ndarray, dom: np.ndarray, d: Design):
    """Weighted mean of y in domain `dom` and its linearization scores (full-sample length)."""
    w = d.weight * dom
    denom = w.sum()
    if denom <= 0:
        return np.nan, np.zeros_like(w)
    yy = np.where(dom, y, 0.0)
    R = (w * yy).sum() / denom
    z = w * (yy - R) / denom
    return R, z


def korn_graubard(p: float, var: float, n: int, df: int, level: float = 0.95):
    """Korn-Graubard (1998) CI for a proportion, NCHS implementation."""
    a = 1 - level
    if n <= 0 or np.isnan(p):
        return np.nan, np.nan, np.nan
    if var > 0:
        n_eff = p * (1 - p) / var
    else:
        n_eff = n
    n_eff = min(n_eff, n)
    if df > 0 and n > 1:
        n_eff_df = n_eff * (stats.t.ppf(1 - a / 2, n - 1) / stats.t.ppf(1 - a / 2, df)) ** 2
    else:
        n_eff_df = n_eff
    n_eff_df = min(n_eff_df, n)
    x = p * n_eff_df
    lo = 0.0 if x <= 0 else stats.beta.ppf(a / 2, x, n_eff_df - x + 1)
    hi = 1.0 if x >= n_eff_df else stats.beta.ppf(1 - a / 2, x + 1, n_eff_df - x)
    return lo, hi, n_eff


def nchs_proportion_reliability(p, lo, hi, n_eff, df):
    """NCHS Data Presentation Standards for Proportions (Parker et al., 2017)."""
    flags = []
    if n_eff < 30:
        flags.append("SUPPRESS: effective sample size < 30")
    width = hi - lo
    if width >= 0.30:
        flags.append("SUPPRESS: absolute CI width >= 30 percentage points")
    elif width > 0.05 and p > 0 and width / p > 1.30:
        flags.append("SUPPRESS: relative CI width > 130%")
    if p == 0 or p == 1:
        flags.append("REVIEW: estimate of 0% or 100%")
    if df < 8:
        flags.append("REVIEW: degrees of freedom < 8")
    if not flags:
        return "reliable", flags
    return ("suppress" if any(f.startswith("SUPPRESS") for f in flags) else "review"), flags


def estimate(
    df: pd.DataFrame,
    variable: str,
    weight: str,
    statistic: str = "mean",
    domain_mask: np.ndarray | None = None,
    age_adjust: str | None = None,
    level: float = 0.95,
) -> dict:
    d = design_from_frame(df, weight)
    y = df[variable].to_numpy(dtype=float)
    valid = ~np.isnan(y) & (d.weight > 0)
    dom = valid.copy()
    if domain_mask is not None:
        dom &= domain_mask.astype(bool)
    y0 = np.nan_to_num(y)

    out: dict = {"variable": variable, "statistic": statistic, "weight": weight,
                 "design_df": d.df, "n_unweighted": int(dom.sum())}

    if statistic == "proportion":
        vals = pd.unique(y[dom])
        if not set(np.unique(vals)).issubset({0.0, 1.0}):
            raise ValueError(f"'{variable}' is not 0/1 coded in the domain (values: {sorted(vals)[:10]}). "
                             "Derive a 0/1 indicator first.")

    if statistic == "total":
        z = d.weight * np.where(dom, y0, 0.0)
        est = z.sum()
        var = linearized_cov(z, d)[0, 0]
        se = np.sqrt(var)
        t = stats.t.ppf(1 - (1 - level) / 2, max(d.df, 1))
        out.update(estimate=est, se=se, ci_low=est - t * se, ci_high=est + t * se)
        return out

    if age_adjust:
        std = AGE_STANDARDS[age_adjust]
        age = df[std["age_var"]].to_numpy(dtype=float)
        est, z = 0.0, np.zeros(len(df))
        cells = []
        for (a0, a1), c in zip(std["groups"], std["proportions"]):
            dk = dom & (age >= a0) & (age <= a1)
            Rk, zk = domain_ratio(y0, dk, d)
            cells.append({"age_group": (f"{a0}-{a1}" if a1 < 200 else f"{a0}+"), "std_proportion": c,
                          "crude_estimate": Rk, "n": int(dk.sum())})
            est += c * Rk
            z += c * zk
        out["age_adjustment"] = {"standard": std["note"], "cells": cells}
    else:
        est, z = domain_ratio(y0, dom, d)

    var = linearized_cov(z, d)[0, 0]
    se = float(np.sqrt(var))
    tcrit = stats.t.ppf(1 - (1 - level) / 2, max(d.df, 1))
    out.update(estimate=float(est), se=se)
    if statistic == "proportion":
        lo, hi, n_eff = korn_graubard(est, var, int(dom.sum()), d.df, level)
        status, flags = nchs_proportion_reliability(est, lo, hi, n_eff, d.df)
        out.update(ci_low=float(lo), ci_high=float(hi), ci_method="Korn-Graubard",
                   effective_n=float(n_eff), nchs_reliability=status, reliability_flags=flags)
    else:
        out.update(ci_low=float(est - tcrit * se), ci_high=float(est + tcrit * se), ci_method="t (design df)",
                   rse_percent=float(100 * se / abs(est)) if est else None)
    return out


def regression(df: pd.DataFrame, outcome: str, predictors: list[str], weight: str,
               family: str = "gaussian", domain_mask: np.ndarray | None = None,
               level: float = 0.95, max_iter: int = 50) -> dict:
    """Survey-weighted GLM (gaussian identity / binomial logit) with linearized sandwich SEs.

    Domain handled by zeroing weights outside the domain (design kept intact)."""
    d = design_from_frame(df, weight)
    cols = [outcome] + predictors
    X_df = df[predictors].astype(float)
    y = df[outcome].to_numpy(dtype=float)
    complete = ~df[cols].isna().any(axis=1).to_numpy() & (d.weight > 0)
    if domain_mask is not None:
        complete &= domain_mask.astype(bool)
    w = np.where(complete, d.weight, 0.0)
    X = np.column_stack([np.ones(len(df)), np.nan_to_num(X_df.to_numpy())])
    y0 = np.nan_to_num(y)
    names = ["(Intercept)"] + predictors
    beta = np.zeros(X.shape[1])
    if family == "binomial":
        if not set(np.unique(y0[complete])).issubset({0.0, 1.0}):
            raise ValueError("Binomial outcome must be 0/1 in the analytic domain.")
        for _ in range(max_iter):
            eta = X @ beta
            mu = 1 / (1 + np.exp(-eta))
            Wt = w * mu * (1 - mu)
            H = X.T @ (Wt[:, None] * X)
            g = X.T @ (w * (y0 - mu))
            step = np.linalg.solve(H, g)
            beta += step
            if np.max(np.abs(step)) < 1e-10:
                break
        mu = 1 / (1 + np.exp(-(X @ beta)))
        bread = X.T @ ((w * mu * (1 - mu))[:, None] * X)
        U = (w * (y0 - mu))[:, None] * X
    elif family == "gaussian":
        bread = X.T @ (w[:, None] * X)
        beta = np.linalg.solve(bread, X.T @ (w * y0))
        U = (w * (y0 - X @ beta))[:, None] * X
    else:
        raise ValueError("family must be 'gaussian' or 'binomial'")
    Binv = np.linalg.inv(bread)
    meat = linearized_cov(U, d)
    V = Binv @ meat @ Binv
    se = np.sqrt(np.diag(V))
    dfree = max(d.df - (len(beta) - 1), 1)  # residual design df, as in R survey
    tcrit = stats.t.ppf(1 - (1 - level) / 2, dfree)
    rows = []
    for n_, b, s in zip(names, beta, se):
        r = {"term": n_, "coef": float(b), "se": float(s), "t": float(b / s) if s else None,
             "p_value": float(2 * stats.t.sf(abs(b / s), dfree)) if s else None,
             "ci_low": float(b - tcrit * s), "ci_high": float(b + tcrit * s)}
        if family == "binomial":
            r.update(odds_ratio=float(np.exp(b)), or_ci_low=float(np.exp(b - tcrit * s)),
                     or_ci_high=float(np.exp(b + tcrit * s)))
        rows.append(r)
    return {"family": family, "weight": weight, "n_unweighted": int(complete.sum()),
            "design_df": d.df, "residual_df": dfree, "coefficients": rows}


def coxph(df: pd.DataFrame, time: str, event: str, predictors: list[str], weight: str,
          domain_mask: np.ndarray | None = None, level: float = 0.95, max_iter: int = 50) -> dict:
    """Survey-weighted Cox proportional hazards (Breslow ties) with Binder (1992) linearized
    variance -- the estimator used by SUDAAN SURVIVAL and R survey::svycoxph."""
    d = design_from_frame(df, weight)
    cols = [time, event] + predictors
    ok = ~df[cols].isna().any(axis=1).to_numpy() & (d.weight > 0)
    if domain_mask is not None:
        ok &= domain_mask.astype(bool)
    idx = np.flatnonzero(ok)
    t = df[time].to_numpy(float)[idx]
    ev = df[event].to_numpy(float)[idx]
    if not set(np.unique(ev)).issubset({0.0, 1.0}):
        raise ValueError("event must be coded 0/1")
    X = df[predictors].to_numpy(float)[idx]
    X = X - X.mean(axis=0)                     # centering: HRs unchanged, better numerics
    w = d.weight[idx]
    p = X.shape[1]
    ut, inv = np.unique(t, return_inverse=True)
    nT = len(ut)
    D = np.bincount(inv, weights=w * ev, minlength=nT)

    def pieces(b):
        eta = X @ b
        r = w * np.exp(eta)
        g0 = np.bincount(inv, weights=r, minlength=nT)
        g1 = np.column_stack([np.bincount(inv, weights=r * X[:, k], minlength=nT) for k in range(p)])
        S0 = np.cumsum(g0[::-1])[::-1]
        S1 = np.cumsum(g1[::-1], axis=0)[::-1]
        return eta, r, S0, S1

    def loglik(b):
        eta, _, S0, _ = pieces(b)
        m = D > 0
        return float((w * ev * eta).sum() - (D[m] * np.log(S0[m])).sum())

    beta = np.zeros(p)
    ll = loglik(beta)
    for _ in range(max_iter):
        eta, r, S0, S1 = pieces(beta)
        m = D > 0
        xbar = np.zeros((nT, p))
        xbar[m] = S1[m] / S0[m, None]
        U = (w * ev) @ X - (D[:, None] * xbar).sum(axis=0)
        info = np.zeros((p, p))
        for k in range(p):
            for l in range(k, p):
                g2 = np.bincount(inv, weights=r * X[:, k] * X[:, l], minlength=nT)
                S2 = np.cumsum(g2[::-1])[::-1]
                v = (D[m] * (S2[m] / S0[m] - xbar[m, k] * xbar[m, l])).sum()
                info[k, l] = info[l, k] = v
        step = np.linalg.solve(info, U)
        new, stepf = beta + step, 1.0
        while loglik(new) < ll - 1e-10 and stepf > 1e-4:
            stepf /= 2
            new = beta + stepf * step
        ll_new = loglik(new)
        converged = abs(ll_new - ll) < 1e-10 * (1 + abs(ll)) or np.max(np.abs(new - beta)) < 1e-10
        beta, ll = new, ll_new
        if converged:
            break
    eta, r, S0, S1 = pieces(beta)
    m = D > 0
    xbar = np.zeros((nT, p))
    xbar[m] = S1[m] / S0[m, None]
    A = np.cumsum(np.where(m, D / np.where(m, S0, 1), 0.0))
    B = np.cumsum(np.where(m[:, None], D[:, None] * xbar / np.where(m, S0, 1)[:, None], 0.0), axis=0)
    info = np.zeros((p, p))
    for k in range(p):
        for l in range(k, p):
            g2 = np.bincount(inv, weights=r * X[:, k] * X[:, l], minlength=nT)
            S2 = np.cumsum(g2[::-1])[::-1]
            info[k, l] = info[l, k] = (D[m] * (S2[m] / S0[m] - xbar[m, k] * xbar[m, l])).sum()
    Iinv = np.linalg.inv(info)
    ex = np.exp(eta)
    resid = w[:, None] * (ev[:, None] * (X - xbar[inv]) - ex[:, None] * (X * A[inv][:, None] - B[inv]))
    full = np.zeros((len(df), p))
    full[idx] = resid
    V = Iinv @ linearized_cov(full, d) @ Iinv
    se = np.sqrt(np.diag(V))
    dfree = max(d.df, 1)
    tcrit = stats.t.ppf(1 - (1 - level) / 2, dfree)
    rows = []
    for n_, b, s in zip(predictors, beta, se):
        rows.append({"term": n_, "coef": float(b), "se": float(s), "hazard_ratio": float(np.exp(b)),
                     "hr_ci_low": float(np.exp(b - tcrit * s)), "hr_ci_high": float(np.exp(b + tcrit * s)),
                     "p_value": float(2 * stats.t.sf(abs(b / s), dfree))})
    return {"model": "Cox PH (Breslow ties), Binder linearized variance", "weight": weight,
            "n_unweighted": int(len(idx)), "events_unweighted": int(ev.sum()), "design_df": d.df,
            "log_partial_likelihood": ll, "coefficients": rows, "_score_residuals": full, "_beta": beta,
            "_info_inv": Iinv}
