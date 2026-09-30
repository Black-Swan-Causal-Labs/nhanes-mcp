"""Benchmark: reproduce published NCHS estimates through the MCP tool functions.

Targets (published NCHS Data Briefs):
  DB360 (2017-2018): adult (20+) obesity, age-adjusted 42.4%; severe obesity 9.2%. Pregnant excluded.
  DB508 (Aug 2021-Aug 2023): obesity 40.3% (men 39.2, women 41.3); severe 9.4%. Crude. Pregnant excluded.
  DB511 (Aug 2021-Aug 2023): hypertension 47.7% (age-adj 44.5%), awareness 59.2%, treatment 51.2%, control 20.7%.
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

# --- DB511 (Aug 2021-Aug 2023): hypertension (SBP>=130 or DBP>=80 or on medication), adults 18+,
#     pregnant excluded; mean of up to 3 oscillometric readings. Crude unless noted; age-adjusted to the
#     2000 US standard 18-39/40-59/60+. Awareness/treatment/control among adults with hypertension.
b = S.build_dataset(["2021-2023"], ["BPXO", "BPQ"],
                    ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3", "BPQ020", "BPQ150"])
d = b["dataset_id"]
S.set_missing(d, "BPQ020", [7, 9]); S.set_missing(d, "BPQ150", [7, 9])
S.derive_variable(d, "SBP", "(fillna(BPXOSY1,0)+fillna(BPXOSY2,0)+fillna(BPXOSY3,0))/(notna(BPXOSY1)+notna(BPXOSY2)+notna(BPXOSY3))")
S.derive_variable(d, "DBP", "(fillna(BPXODI1,0)+fillna(BPXODI2,0)+fillna(BPXODI3,0))/(notna(BPXODI1)+notna(BPXODI2)+notna(BPXODI3))")
S.derive_variable(d, "MEDS", "where(BPQ020 == 1, fillna(BPQ150, 2) == 1, 0)")
S.derive_variable(d, "HTN", "(SBP >= 130) | (DBP >= 80) | (MEDS == 1)")
S.derive_variable(d, "AWARE", "BPQ020 == 1")
S.derive_variable(d, "CONTROL", "(SBP < 130) & (DBP < 80)")
S.derive_variable(d, "AGE3", "(RIDAGEYR >= 40) + (RIDAGEYR >= 60) + 1")
AD18 = "RIDAGEYR >= 18 & RIDEXPRG != 1"
HT = AD18 + " & HTN == 1"
STD18 = {"groups": [[18, 39], [40, 59], [60, 200]], "proportions": [0.420263, 0.357202, 0.222535],
         "note": "NCHS 2000 US standard, adults 18+"}
check("DB511 hypertension 2021-23", 47.7, S.survey_estimate(d, "HTN", "proportion", AD18))
check("DB511 hypertension 2021-23 (age-adj)", 44.5, S.survey_estimate(d, "HTN", "proportion", AD18, age_adjust=STD18))
hs = S.survey_estimate(d, "HTN", "proportion", AD18, by=["RIAGENDR"])
check("DB511 hypertension men", 50.8, hs, {"RIAGENDR": 1.0})
check("DB511 hypertension women", 44.6, hs, {"RIAGENDR": 2.0})
ha = S.survey_estimate(d, "HTN", "proportion", AD18, by=["AGE3"])
for g, v in ((1.0, 23.4), (2.0, 52.5), (3.0, 71.6)):
    check(f"DB511 hypertension age group {int(g)}", v, ha, {"AGE3": g})
check("DB511 awareness", 59.2, S.survey_estimate(d, "AWARE", "proportion", HT))
aw = S.survey_estimate(d, "AWARE", "proportion", HT, by=["RIAGENDR"])
check("DB511 awareness men", 55.2, aw, {"RIAGENDR": 1.0})
check("DB511 awareness women", 63.6, aw, {"RIAGENDR": 2.0})
check("DB511 treatment", 51.2, S.survey_estimate(d, "MEDS", "proportion", HT))
check("DB511 control", 20.7, S.survey_estimate(d, "CONTROL", "proportion", HT))
ca = S.survey_estimate(d, "CONTROL", "proportion", HT, by=["AGE3"])
for g, v in ((1.0, 4.5), (2.0, 18.1), (3.0, 29.2)):
    check(f"DB511 control age group {int(g)}", v, ca, {"AGE3": g})

print(json.dumps({"lipid_weight_choice": lip_weight, "cmv_weight_choice": cmv_weight, "results": rows},
                 indent=1, default=str))
