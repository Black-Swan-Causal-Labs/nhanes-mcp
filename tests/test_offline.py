"""Offline tests with synthetic NHANES-shaped XPT files (no network).

Run: NHANES_MCP_DATA_DIR=<tmp> python -m pytest tests/  (the fixture sets this up itself)
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

TMP = Path(tempfile.mkdtemp())
os.environ["NHANES_MCP_DATA_DIR"] = str(TMP / "data")
os.environ["NHANES_MCP_CACHE"] = str(TMP / "cache")
(TMP / "data").mkdir()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rng = np.random.default_rng(7)


def make_cycle(suffix, seq0, stratum0):
    rows = []
    seqn = seq0
    for h in range(15):
        for psu in (1, 2):
            n = rng.integers(250, 350)
            shift = rng.normal(0, 1.5)  # PSU-level clustering
            for _ in range(n):
                seqn += 1
                age = rng.integers(0, 81)
                sex = rng.integers(1, 3)
                examined = rng.random() > 0.06
                w = rng.uniform(2000, 40000)
                bmi = 22 + 0.12 * min(age, 60) + shift + rng.normal(0, 5) if (examined and age >= 2) else np.nan
                if examined and rng.random() < 0.03:
                    bmi = np.nan
                preg = 1.0 if (sex == 2 and 20 <= age <= 44 and rng.random() < 0.05) else (2.0 if sex == 2 and 20 <= age <= 44 else np.nan)
                rows.append(dict(SEQN=seqn, RIDSTATR=2 if examined else 1, RIAGENDR=sex, RIDAGEYR=age,
                                 RIDEXPRG=preg, SDMVSTRA=stratum0 + h, SDMVPSU=psu,
                                 WTINT2YR=w, WTMEC2YR=w * 1.07 if examined else 0.0, BMXBMI=bmi,
                                 BPQ020=rng.choice([1, 2, 9], p=[.3, .69, .01]) if age >= 16 else np.nan))
    df = pd.DataFrame(rows)
    demo = df[["SEQN", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDEXPRG", "SDMVSTRA", "SDMVPSU", "WTINT2YR", "WTMEC2YR"]]
    bmx = df.loc[df.RIDSTATR == 2, ["SEQN", "BMXBMI"]]
    bpq = df.loc[df.RIDAGEYR >= 16, ["SEQN", "BPQ020"]]
    ex = df[df.RIDSTATR == 2]
    fast = ex.sample(frac=0.45, random_state=1)
    glu = pd.DataFrame({"SEQN": ex.SEQN, "LBXGLU": np.where(ex.SEQN.isin(fast.SEQN), rng.normal(100, 15, len(ex)), np.nan),
                        "WTSAF2YR": np.where(ex.SEQN.isin(fast.SEQN), ex.WTMEC2YR * 2.2, 0.0)})
    for name, d in [("DEMO", demo), ("BMX", bmx), ("BPQ", bpq), ("GLU", glu)]:
        pyreadstat.write_xport(d.astype({c: float for c in d.columns}), str(TMP / "data" / f"{name}{suffix}.xpt"),
                               file_format_version=5, table_name=name[:8])
    return df


TRUTH_J = make_cycle("_J", 93700, 134)
TRUTH_I = make_cycle("_I", 83700, 119)

from nhanes_mcp import server as S  # noqa: E402
from nhanes_mcp import survey as sv  # noqa: E402


def naive_linearized_var(y, dom, w, h, i):
    """Independent textbook implementation (loops) of the domain-mean variance."""
    wd = w * dom
    R = np.sum(wd * y) / np.sum(wd)
    z = wd * (y - R) / np.sum(wd)
    var = 0.0
    for hh in np.unique(h):
        psus = np.unique(i[h == hh])
        tots = np.array([z[(h == hh) & (i == p)].sum() for p in psus])
        var += len(psus) / (len(psus) - 1) * np.sum((tots - tots.mean()) ** 2)
    return R, var


def jackknife_var(y, dom, w, h, i):
    """JKn delete-one-PSU jackknife -- should approximate linearization."""
    def est(wt):
        wd = wt * dom
        return np.sum(wd * y) / np.sum(wd)
    full = est(w)
    var = 0.0
    for hh in np.unique(h):
        psus = np.unique(i[h == hh])
        nh = len(psus)
        for p in psus:
            wr = w.copy()
            drop = (h == hh) & (i == p)
            keep = (h == hh) & ~drop
            wr[drop] = 0
            wr[keep] *= nh / (nh - 1)
            var += (nh - 1) / nh * (est(wr) - full) ** 2
    return var


def test_weight_selection():
    r = S.build_dataset(["2017-2018"], ["BMX"])
    assert r["weight"]["kind"] == "WTMEC", r["weight"]
    r = S.build_dataset(["2017-2018"], ["BPQ"])
    assert r["weight"]["kind"] == "WTINT", r["weight"]
    r = S.build_dataset(["2017-2018"], ["BMX", "GLU"])
    assert r["weight"]["kind"] == "WTSAF", r["weight"]
    print("weight selection OK:", r["weight"])


def test_point_and_variance_match_independent_implementation():
    r = S.build_dataset(["2017-2018"], ["BMX"])
    ds = S.DATASETS[r["dataset_id"]]["df"]
    out = S.survey_estimate(r["dataset_id"], "BMXBMI", domain="RIDAGEYR >= 20 & RIDEXPRG != 1")["results"][0]
    y = ds.BMXBMI.to_numpy(float)
    dom = ((ds.RIDAGEYR >= 20) & (ds.RIDEXPRG != 1) & ds.BMXBMI.notna() & (ds.WT_ANALYSIS > 0)).to_numpy()
    R, var = naive_linearized_var(np.nan_to_num(y), dom, ds.WT_ANALYSIS.to_numpy(), ds.SDMVSTRA_U.to_numpy(), ds.SDMVPSU.to_numpy())
    jk = jackknife_var(np.nan_to_num(y), dom, ds.WT_ANALYSIS.to_numpy(), ds.SDMVSTRA_U.to_numpy(), ds.SDMVPSU.to_numpy())
    assert abs(out["estimate"] - R) < 1e-10
    assert abs(out["se"] ** 2 - var) / var < 1e-8
    assert abs(np.sqrt(jk) - out["se"]) / out["se"] < 0.05
    assert out["design_df"] == 15
    print(f"mean BMI {out['estimate']:.3f}, SE lin {out['se']:.4f}, SE jk {np.sqrt(jk):.4f}")


def test_domain_not_subset():
    """Subsetting then estimating drops empty PSUs; domain estimation must keep the full design."""
    r = S.build_dataset(["2017-2018"], ["BMX"])
    out = S.survey_estimate(r["dataset_id"], "BMXBMI", domain="RIDAGEYR >= 75")["results"][0]
    assert out["design_df"] == 15


def test_proportion_age_adjusted_and_missing_propagation():
    r = S.build_dataset(["2017-2018"], ["BMX"])
    d = S.derive_variable(r["dataset_id"], "OBESE", "BMXBMI >= 30")
    ds = S.DATASETS[r["dataset_id"]]["df"]
    assert ds.loc[ds.BMXBMI.isna(), "OBESE"].isna().all(), "missing BMI must yield missing OBESE"
    res = S.survey_estimate(r["dataset_id"], "OBESE", "proportion", domain="RIDAGEYR >= 20 & RIDEXPRG != 1",
                            age_adjust="nchs_adults_20plus")["results"][0]
    manual = sum(c["std_proportion"] * c["crude_estimate"] for c in res["age_adjustment"]["cells"])
    assert abs(manual - res["estimate"]) < 1e-12
    assert res["ci_low"] < res["estimate"] < res["ci_high"]
    by = S.survey_estimate(r["dataset_id"], "OBESE", "proportion", domain="RIDAGEYR >= 20 & RIDEXPRG != 1",
                           by=["RIAGENDR"])["results"]
    assert len(by) == 2
    print("age-adjusted obesity", round(res["estimate"], 4), res["nchs_reliability"], d)


def test_pooling_weights():
    r = S.build_dataset(["2015-2016", "2017-2018"], ["BMX"])
    ds = S.DATASETS[r["dataset_id"]]["df"]
    assert np.allclose(ds.loc[ds.CYCLE == "2017-2018", "WT_ANALYSIS"], ds.loc[ds.CYCLE == "2017-2018", "WTMEC2YR"] * 0.5)
    assert r["weight"]["per_cycle"]["2015-2016"].endswith("x 0.5000")
    out = S.survey_estimate(r["dataset_id"], "BMXBMI")["results"][0]
    assert out["design_df"] == 30


def test_guards():
    try:
        S.build_dataset(["2017-2018", "2017-2020"], ["BMX"])
        raise AssertionError("should refuse overlapping cycles")
    except ValueError:
        pass
    r = S.build_dataset(["2017-2018"], ["BMX"])
    try:
        S.derive_variable(r["dataset_id"], "X", "__import__('os')")
        raise AssertionError("unsafe expression accepted")
    except ValueError:
        pass
    try:
        S.survey_estimate(r["dataset_id"], "BMXBMI", "proportion")
        raise AssertionError("non-binary proportion accepted")
    except ValueError:
        pass


def test_regression():
    r = S.build_dataset(["2017-2018"], ["BMX"])
    S.derive_variable(r["dataset_id"], "FEMALE", "RIAGENDR == 2")
    S.derive_variable(r["dataset_id"], "OBESE", "BMXBMI >= 30")
    lin = S.survey_regression(r["dataset_id"], "BMXBMI", ["RIDAGEYR", "FEMALE"], "gaussian", domain="RIDAGEYR >= 20")
    ds = S.DATASETS[r["dataset_id"]]["df"]
    m = (ds.RIDAGEYR >= 20) & ds.BMXBMI.notna() & (ds.WT_ANALYSIS > 0)
    X = np.column_stack([np.ones(m.sum()), ds.loc[m, "RIDAGEYR"], ds.loc[m, "FEMALE"]])
    W = ds.loc[m, "WT_ANALYSIS"].to_numpy()
    b = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * ds.loc[m, "BMXBMI"].to_numpy()))
    assert np.allclose([c["coef"] for c in lin["coefficients"]], b)
    lg = S.survey_regression(r["dataset_id"], "OBESE", ["RIDAGEYR", "FEMALE"], "binomial", domain="RIDAGEYR >= 20")
    assert all(c["se"] > 0 for c in lg["coefficients"])
    print("logistic OR age", round(lg["coefficients"][1]["odds_ratio"], 4))


def test_expression_semantics():
    r = S.build_dataset(["2017-2018"], ["BMX"])
    ds = S.DATASETS[r["dataset_id"]]["df"]
    S.derive_variable(r["dataset_id"], "AGEGRP", "(RIDAGEYR >= 40) + (RIDAGEYR >= 60)")
    assert set(ds.AGEGRP.dropna().unique()) == {0.0, 1.0, 2.0}, "bool + bool must count, not OR"
    m, _ = S._eval(ds, "RIDAGEYR >= 20 & RIDEXPRG != 1")
    assert (m == ((ds.RIDAGEYR >= 20) & (ds.RIDEXPRG != 1)).astype(float)).all(), "pandas-style precedence"
    m2, _ = S._eval(ds, "RIAGENDR == 1 | RIDAGEYR < 5")
    assert (m2 == ((ds.RIAGENDR == 1) | (ds.RIDAGEYR < 5)).astype(float)).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
