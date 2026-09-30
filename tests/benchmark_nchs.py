"""Benchmark: reproduce published NCHS estimates through the MCP tool functions.

Targets (published NCHS Data Briefs):
  DB360 (2017-2018): adult (20+) obesity, age-adjusted 42.4%; severe obesity 9.2%. Pregnant excluded.
  DB508 (Aug 2021-Aug 2023): obesity 40.3% (men 39.2, women 41.3); severe 9.4%. Crude. Pregnant excluded.
  Bate et al. 2010 (NHANES 1999-2004): CMV seroprevalence 6-49y, age-adjusted 50.4%; surplus-serum weights.
  DB515 (Aug 2021-Aug 2023): high total cholesterol (>=240) 11.3% (men 10.6, women 11.9);
                             low HDL (<40) 13.8% (men 21.5, women 6.6). Crude. Phlebotomy weights.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nhanes_mcp import server as S  # noqa: E402

ADULT_NP = "RIDAGEYR >= 20 & RIDEXPRG != 1"
rows = []


def check(label, published, res, group=None):
    r = next(x for x in res["results"] if (group is None or x["group"] == group))
    est = round(100 * r["estimate"], 1)
    rows.append({"benchmark": label, "published": published, "server": est,
                 "diff_pp": round(est - published, 1),
                 "ci": f"{100*r['ci_low']:.1f}-{100*r['ci_high']:.1f}", "n": r["n_unweighted"],
                 "weight": res["weight_per_cycle"], "reliability": r.get("nchs_reliability")})


# --- DB360: 2017-2018, age-adjusted ---
b = S.build_dataset(["2017-2018"], ["BMX"], ["BMXBMI"])
S.derive_variable(b["dataset_id"], "OBESE", "BMXBMI >= 30")
S.derive_variable(b["dataset_id"], "SEVOB", "BMXBMI >= 40")
check("DB360 obesity 2017-18 (age-adj)", 42.4,
      S.survey_estimate(b["dataset_id"], "OBESE", "proportion", ADULT_NP, age_adjust="nchs_adults_20plus"))
check("DB360 severe obesity 2017-18 (age-adj)", 9.2,
      S.survey_estimate(b["dataset_id"], "SEVOB", "proportion", ADULT_NP, age_adjust="nchs_adults_20plus"))

# --- DB508: 2021-2023, crude ---
b = S.build_dataset(["2021-2023"], ["BMX"], ["BMXBMI"])
S.derive_variable(b["dataset_id"], "OBESE", "BMXBMI >= 30")
S.derive_variable(b["dataset_id"], "SEVOB", "BMXBMI >= 40")
check("DB508 obesity 2021-23", 40.3, S.survey_estimate(b["dataset_id"], "OBESE", "proportion", ADULT_NP))
sx = S.survey_estimate(b["dataset_id"], "OBESE", "proportion", ADULT_NP, by=["RIAGENDR"])
check("DB508 obesity 2021-23 men", 39.2, sx, {"RIAGENDR": 1.0})
check("DB508 obesity 2021-23 women", 41.3, sx, {"RIAGENDR": 2.0})
check("DB508 severe obesity 2021-23", 9.4, S.survey_estimate(b["dataset_id"], "SEVOB", "proportion", ADULT_NP))

# --- DB515: 2021-2023 lipids, crude ---
b = S.build_dataset(["2021-2023"], ["TCHOL", "HDL"], ["LBXTC", "LBDHDD"])
S.derive_variable(b["dataset_id"], "HIGHTC", "LBXTC >= 240")
S.derive_variable(b["dataset_id"], "LOWHDL", "LBDHDD < 40")
tc = S.survey_estimate(b["dataset_id"], "HIGHTC", "proportion", "RIDAGEYR >= 20", by=["RIAGENDR"])
check("DB515 high TC 2021-23", 11.3, S.survey_estimate(b["dataset_id"], "HIGHTC", "proportion", "RIDAGEYR >= 20"))
check("DB515 high TC 2021-23 men", 10.6, tc, {"RIAGENDR": 1.0})
check("DB515 high TC 2021-23 women", 11.9, tc, {"RIAGENDR": 2.0})
hd = S.survey_estimate(b["dataset_id"], "LOWHDL", "proportion", "RIDAGEYR >= 20", by=["RIAGENDR"])
check("DB515 low HDL 2021-23", 13.8, S.survey_estimate(b["dataset_id"], "LOWHDL", "proportion", "RIDAGEYR >= 20"))
check("DB515 low HDL 2021-23 men", 21.5, hd, {"RIAGENDR": 1.0})
check("DB515 low HDL 2021-23 women", 6.6, hd, {"RIAGENDR": 2.0})
lip_weight = b["weight"]

# --- Bate et al., Clin Infect Dis 2010 (doi:10.1086/652438): CMV IgG seroprevalence, ages 6-49,
#     NHANES 1999-2004 pooled, surplus-serum weights (auto-selected: WTSSMC 4-year 1999-2002 + 2-year 2003-04).
#     Published age-adjusted: overall 50.4% (abstract); men 45.2%, women 55.5% (as quoted by Lehrer 2012).
#     Age standard: 2000 Census counts for 6-11/12-19/20-29/30-39/40-49 (millions), derived from
#     the Census 5-year age groups -- an approximation of the paper's standard.
b = S.build_dataset(["1999-2000", "2001-2002", "2003-2004"], ["SSCMV"], ["SSCMV"])
cmv_weight = b["weight"]
S.set_missing(b["dataset_id"], "SSCMV", [3])  # equivocal -> missing
S.derive_variable(b["dataset_id"], "CMV", "SSCMV == 1")
STD_6_49 = {"groups": [[6, 11], [12, 19], [20, 29], [30, 39], [40, 49]],
            "population": [24.65, 32.54, 38.34, 43.22, 42.53],
            "note": "2000 US Census, ages 6-49 (approximation of Bate et al. standard)"}
CMV_DOM = "RIDAGEYR >= 6 & RIDAGEYR <= 49"
check("Bate 2010 CMV 1999-2004 (age-adj)", 50.4,
      S.survey_estimate(b["dataset_id"], "CMV", "proportion", CMV_DOM, age_adjust=STD_6_49))
cs = S.survey_estimate(b["dataset_id"], "CMV", "proportion", CMV_DOM, by=["RIAGENDR"], age_adjust=STD_6_49)
check("Bate 2010 CMV men (age-adj)", 45.2, cs, {"RIAGENDR": 1.0})
check("Bate 2010 CMV women (age-adj)", 55.5, cs, {"RIAGENDR": 2.0})

print(json.dumps({"lipid_weight_choice": lip_weight, "cmv_weight_choice": cmv_weight, "results": rows},
                 indent=1, default=str))
