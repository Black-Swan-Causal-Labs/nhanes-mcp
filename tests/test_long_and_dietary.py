"""v0.3 tests: long-format guard, flag_from_long_table, dietary day-1 weights, survey_frequency."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_offline as T  # builds synthetic DEMO/BMX/... files and sets env
S = T.S

rng = np.random.default_rng(11)
demo = T.TRUTH_J
rows = []
for seqn, age in zip(demo.SEQN, demo.RIDAGEYR):
    k = rng.integers(0, 4)
    if k == 0:
        rows.append(dict(SEQN=seqn, RXDUSE=2.0, RXDRSC1="", RXDRSC2=""))
    for _ in range(k):
        code = "G40.9" if rng.random() < 0.02 else rng.choice(["I10", "E11.9", "E78.5"])
        rows.append(dict(SEQN=seqn, RXDUSE=1.0, RXDRSC1=code, RXDRSC2=""))
rx = pd.DataFrame(rows)
pyreadstat.write_xport(rx, str(T.TMP / "data" / "RXQ_RX_J.xpt"), file_format_version=5, table_name="RXQ_RX")
ex = demo[demo.RIDSTATR == 2]
dr = pd.DataFrame({"SEQN": ex.SEQN.astype(float), "DR1DRSTZ": np.where(rng.random(len(ex)) < 0.9, 1.0, 5.0),
                   "DR1TKCAL": rng.normal(2000, 500, len(ex))})
dr["WTDRD1"] = np.where(dr.DR1DRSTZ == 1, ex.WTMEC2YR.values * 1.1, 0.0)
dr["WTDR2D"] = np.where((dr.DR1DRSTZ == 1) & (rng.random(len(ex)) < 0.8), ex.WTMEC2YR.values * 1.3, 0.0)
pyreadstat.write_xport(dr, str(T.TMP / "data" / "DR1TOT_J.xpt"), file_format_version=5, table_name="DR1TOT")


def test_long_table_refused():
    try:
        S.build_dataset(["2017-2018"], ["RXQ_RX"])
        raise AssertionError("long-format table should be refused")
    except ValueError as e:
        assert "flag_from_long_table" in str(e)


def test_dietary_day1_weight_and_flag():
    r = S.build_dataset(["2017-2018"], ["DR1TOT"], ["DR1DRSTZ", "DR1TKCAL"])
    assert r["weight"]["kind"] == "WTDRD1", r["weight"]
    f = S.flag_from_long_table(r["dataset_id"], "RXQ_RX", ["RXDRSC1", "RXDRSC2"], "^G40", "EPILEPSY")
    ds = S.DATASETS[r["dataset_id"]]["df"]
    truth = rx.assign(h=rx.RXDRSC1.str.startswith("G40")).groupby("SEQN").h.any()
    assert f["n_flagged"] == int(truth.sum())
    assert len(ds) == len(demo), "flagging must not duplicate participants"
    fr = S.survey_frequency(r["dataset_id"], "RIAGENDR", domain="RIDAGEYR >= 40 & DR1DRSTZ == 1",
                            by=["EPILEPSY"], labels={"1": "Male", "2": "Female"})
    for grp in fr["results"]:
        assert abs(sum(l["pct"] for l in grp["levels"]) - 100) < 0.05
    print("flagged", f["n_flagged"], "groups", [g["n_total"] for g in fr["results"]])


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn(); print("PASS", k)
