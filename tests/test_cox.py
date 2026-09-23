"""Validate the survey Cox estimator: point estimates vs lifelines; score residuals vs lifelines;
Binder variance vs delete-one-PSU jackknife; ties handled (Breslow)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nhanes_mcp import survey as sv  # noqa: E402

rng = np.random.default_rng(3)
rows = []
for h in range(20):
    for psu in (1, 2):
        u = rng.normal(0, 0.3)
        for _ in range(rng.integers(150, 250)):
            age = rng.uniform(25, 85); male = rng.integers(0, 2); x = rng.normal()
            lam = np.exp(-9 + 0.08 * age + 0.5 * male + 0.3 * x + u)
            T = rng.exponential(1 / lam) * 12; C = rng.uniform(60, 240)
            rows.append(dict(SDMVSTRA_U=f"s{h}", SDMVPSU=psu, W=rng.uniform(1e3, 5e4), AGE=age, MALE=male, X=x,
                             TIME=min(T, C), EVENT=float(T <= C)))
df = pd.DataFrame(rows)
P = ["AGE", "MALE", "X"]


def _sm(d):
    from statsmodels.duration.hazard_regression import PHReg
    m = PHReg(d["TIME"].values, d[P].values, status=d["EVENT"].values, ties="breslow").fit()
    return m


def test_matches_statsmodels_unweighted_with_ties():
    d1 = df.assign(W=1.0, TIME=np.ceil(df.TIME))
    r = sv.coxph(d1, "TIME", "EVENT", P, "W")
    m = _sm(d1)
    assert np.allclose([c["coef"] for c in r["coefficients"]], m.params, atol=1e-6)
    assert np.allclose(r["_score_residuals"], m.score_residuals, atol=1e-6)
    print("unweighted Breslow coef & score residuals == statsmodels PHReg")


def test_integer_weights_equal_replicated_rows():
    d2 = df.assign(W=rng.integers(1, 4, len(df)).astype(float), TIME=np.ceil(df.TIME))
    r = sv.coxph(d2, "TIME", "EVENT", P, "W")
    rep = d2.loc[d2.index.repeat(d2.W.astype(int))]
    m = _sm(rep)
    assert np.allclose([c["coef"] for c in r["coefficients"]], m.params, atol=1e-6)
    print("weighted coef == replicated-row statsmodels fit")


def test_binder_vs_jackknife():
    r = sv.coxph(df, "TIME", "EVENT", P, "W")
    b = r["_beta"]
    reps = []
    for h in df.SDMVSTRA_U.unique():
        for p in (1, 2):
            w = df.W.copy()
            drop = (df.SDMVSTRA_U == h) & (df.SDMVPSU == p)
            keep = (df.SDMVSTRA_U == h) & ~drop
            w[drop] = 0; w[keep] *= 2
            rr = sv.coxph(df.assign(W=w), "TIME", "EVENT", P, "W")
            reps.append(rr["_beta"])
    reps = np.array(reps)
    jk_se = np.sqrt(((reps - b) ** 2).sum(axis=0) / 2)
    lin_se = np.array([c["se"] for c in r["coefficients"]])
    print("SE linearized", np.round(lin_se, 5), "jackknife", np.round(jk_se, 5))
    assert np.all(np.abs(lin_se / jk_se - 1) < 0.08)


def test_ties_breslow_domain():
    d2 = df.assign(TIME=np.ceil(df.TIME))          # heavy ties (monthly), like PERMTH_INT
    r = sv.coxph(d2, "TIME", "EVENT", P, "W", domain_mask=(d2.AGE >= 40).to_numpy())
    assert r["n_unweighted"] == int((d2.AGE >= 40).sum())
    assert abs(r["_score_residuals"].sum(axis=0)).max() < 1e-6 * r["_score_residuals"].__abs__().sum()
    print("ties+domain OK", [round(c["hazard_ratio"], 3) for c in r["coefficients"]])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("PASS", k)
