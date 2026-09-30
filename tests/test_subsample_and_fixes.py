"""Offline tests for v0.4 fixes: subsample (surplus-serum) weights with 2Y/4Y suffixes, 1999-2000
'_A' file names, weight overrides recorded in provenance, missing-aware expression functions and
custom age standards. Synthetic data, no network."""
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
rng = np.random.default_rng(11)


def write(name, d):
    pyreadstat.write_xport(d.astype({c: float for c in d.columns}), str(TMP / "data" / f"{name}.xpt"),
                           file_format_version=5, table_name=name[:8])


def make(cycle_file_suffix, seq0, s0, four_year):
    rows, seqn = [], seq0
    for h in range(15):
        for psu in (1, 2):
            for _ in range(rng.integers(150, 220)):
                seqn += 1
                age = int(rng.integers(0, 81))
                w = rng.uniform(2000, 40000)
                ex = rng.random() > 0.05
                tested = ex and 6 <= age <= 49 and rng.random() < 0.7
                rows.append(dict(SEQN=seqn, RIDSTATR=2 if ex else 1, RIAGENDR=int(rng.integers(1, 3)), RIDAGEYR=age,
                                 SDMVSTRA=s0 + h, SDMVPSU=psu, WTINT2YR=w, WTMEC2YR=w * 1.05 if ex else 0.0,
                                 WTINT4YR=w / 2, WTMEC4YR=(w * 1.05 / 2) if ex else 0.0,
                                 tested=tested, cmv=float(rng.random() < 0.3 + 0.008 * age) if tested else np.nan,
                                 wss=w * 1.6 if tested else 0.0))
    df = pd.DataFrame(rows)
    demo_cols = ["SEQN", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "SDMVSTRA", "SDMVPSU", "WTINT2YR", "WTMEC2YR"]
    if four_year:
        demo_cols += ["WTINT4YR", "WTMEC4YR"]
    demo_name = "DEMO" + ("" if cycle_file_suffix == "_A" else cycle_file_suffix)
    write(demo_name, df[demo_cols])
    t = df[df.tested]
    ss = pd.DataFrame({"SEQN": t.SEQN, "SSCMV": np.where(t.cmv == 1, 1.0, 2.0), "WTSSMC2Y": t.wss})
    if four_year:
        ss["WTSSMC4Y"] = t.wss / 2
    write(f"SSCMV{cycle_file_suffix}", ss)
    return df


A = make("_A", 1, 1, True)      # 1999-2000: file named SSCMV_A (the real-world exception)
B = make("_B", 20000, 16, True)  # 2001-2002
C = make("_C", 40000, 31, False)  # 2003-2004: 2-year weights only

from nhanes_mcp import server as S  # noqa: E402
from nhanes_mcp import survey as sv  # noqa: E402


def build():
    return S.build_dataset(["1999-2000", "2001-2002", "2003-2004"], ["SSCMV"], ["SSCMV"])


def test_underscore_a_file_found_and_subsample_weight_selected():
    r = build()
    assert not r["warnings"], r["warnings"]
    assert {f["file"] for f in r["files"]} >= {"SSCMV_A", "SSCMV_B", "SSCMV_C"}
    assert r["weight"]["kind"] == "WTSSMC", r["weight"]
    pc = r["weight"]["per_cycle"]
    assert pc["1999-2000"] == "WTSSMC4Y x 0.6667" and pc["2001-2002"] == "WTSSMC4Y x 0.6667"
    assert pc["2003-2004"] == "WTSSMC2Y x 0.3333"
    ds = S.DATASETS[r["dataset_id"]]["df"]
    c = ds[ds.CYCLE == "2003-2004"]
    assert np.allclose(c.WT_ANALYSIS, c.WTSSMC2Y.fillna(0) / 3)


def test_forced_weight_is_reported():
    r = S.build_dataset(["2003-2004"], ["SSCMV"], ["SSCMV"], weight="WTMEC")
    assert r["weight"]["kind"] == "WTMEC"
    assert any("automatic choice was WTSSMC" in w for w in r["warnings"])


def test_design_columns_protected_and_set_weight_recorded():
    r = build()
    dsid = r["dataset_id"]
    try:
        S.derive_variable(dsid, "WT_ANALYSIS", "WTMEC2YR")
        raise AssertionError("overwriting WT_ANALYSIS via derive_variable must be refused")
    except ValueError:
        pass
    S.derive_variable(dsid, "CMV", "SSCMV == 1")
    S.set_weight(dsid, "coalesce(WTSSMC4Y * 4/6, WTSSMC2Y * 2/6)", "test override")
    out = S.survey_estimate(dsid, "CMV", "proportion", "RIDAGEYR >= 6 & RIDAGEYR <= 49")
    assert out["weight"] == "CUSTOM"
    assert any("set_weight" in w for w in out["warnings"])
    assert all(v.startswith("custom:") for v in out["weight_per_cycle"].values())


def test_coalesce_equals_automatic_pooled_weight():
    r = build()
    dsid = r["dataset_id"]
    ds = S.DATASETS[dsid]["df"]
    auto = ds.WT_ANALYSIS.copy()
    ds["CYCLE_NUM"] = ds.CYCLE.map({"1999-2000": 1, "2001-2002": 2, "2003-2004": 3}).astype(float)
    val, refs = S._eval(ds, "where(CYCLE_NUM == 3, fillna(WTSSMC2Y, 0) * 2/6, fillna(WTSSMC4Y, 0) * 4/6)")
    assert np.allclose(val, auto)
    d = S.derive_variable(dsid, "W2", "coalesce(WTSSMC4Y, WTSSMC2Y)")
    assert d["n_non_missing"] == int(ds[["WTSSMC4Y", "WTSSMC2Y"]].notna().any(axis=1).sum())
    S.derive_variable(dsid, "TESTED", "notna(SSCMV)")
    assert set(ds.TESTED.unique()) == {0.0, 1.0}


def test_custom_age_standard():
    r = build()
    dsid = r["dataset_id"]
    S.derive_variable(dsid, "CMV", "SSCMV == 1")
    spec = {"groups": [[6, 11], [12, 19], [20, 29], [30, 39], [40, 49]],
            "population": [24.65, 32.54, 38.34, 43.22, 42.53], "note": "test standard"}
    out = S.survey_estimate(dsid, "CMV", "proportion", "RIDAGEYR >= 6 & RIDAGEYR <= 49", age_adjust=spec)
    res = out["results"][0]
    manual = sum(c["std_proportion"] * c["crude_estimate"] for c in res["age_adjustment"]["cells"])
    assert abs(manual - res["estimate"]) < 1e-12
    assert abs(sum(c["std_proportion"] for c in res["age_adjustment"]["cells"]) - 1) < 1e-12
    assert "test standard" in out["age_adjusted"] and out["age_adjusted"].count("groups") == 1
    out2 = S.survey_estimate(dsid, "CMV", "proportion", "RIDAGEYR >= 6",
                             age_adjust={"groups": [[6, 19], [20, 49]], "population": [1, 1]})
    for bad in ({"groups": [[6, 20], [20, 49]], "population": [1, 1]}, {"groups": [[6, 19]], "population": [1, 2]}):
        try:
            sv.resolve_age_standard(bad)
            raise AssertionError("bad standard accepted")
        except ValueError:
            pass
    assert out2["results"][0]["estimate"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
