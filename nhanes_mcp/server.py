"""NHANES MCP server — conversational, design-correct access to NHANES public-use data.

Built by Black Swan Causal Labs. The guiding principle: an agent should not be able to
produce an unweighted, wrongly-subset, or wrongly-combined NHANES estimate by accident.
"""
from __future__ import annotations

import itertools
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from . import catalog as cat
from . import survey as sv

mcp = FastMCP("nhanes")
DATASETS: dict[str, dict] = {}
EXPORT_DIR = Path(cat.os.environ.get("NHANES_MCP_EXPORT_DIR", cat.CACHE / "exports"))

DEMO_CORE = ["SEQN", "SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3",
             "RIDEXPRG", "INDFMPIR", "DMDEDUC2", "SDMVSTRA", "SDMVPSU"]
MISSING_WORDS = re.compile(r"refused|don.?t know|missing|not ascertained|cannot be assessed", re.I)

GUIDANCE = """NHANES analysis rules this server enforces (and why):
1. WEIGHTS. Use the weight of the most restrictive component contributing a variable:
   interview-only -> WTINT; any MEC exam/lab/MEC-administered questionnaire -> WTMEC;
   fasting / phlebotomy / other subsample files -> that file's own weight (e.g. WTSAF2YR, WTPH2YR).
   build_dataset selects this automatically and explains its choice.
2. DESIGN. Variance uses Taylor linearization with strata SDMVSTRA and PSUs SDMVPSU
   (with-replacement). Design df = #PSU - #strata.
3. SUBPOPULATIONS. Never subset the data before estimation; pass a `domain` expression.
   The full design is retained and records outside the domain get zero weight.
4. COMBINING CYCLES. Weights are rescaled by (cycle years / total years); 1999-2002 uses the
   4-year weights. 2017-2020 (pre-pandemic P_ files, 3.2 years) overlaps 2017-2018 and cannot
   be combined with it. 2021-2023 followed a redesigned sample; NCHS cautions against pooling it
   with earlier cycles.
5. MISSING CODES. Questionnaire items code refused/don't know as 7/9, 77/99, 777/999 etc.
   Check describe_variable and use set_missing before analysis.
6. DERIVED VARIABLES. derive_variable propagates missingness (a 0/1 indicator built from a
   missing BMI stays missing, it does not silently become 0).
7. AGE. RIDAGEYR is top-coded at 80 (85 in early cycles). Age-adjusted adult prevalence uses the
   2000 projected census standard with groups 20-39, 40-59, 60+ (age_adjust='nchs_adults_20plus').
8. RELIABILITY. Proportions report Korn-Graubard CIs and NCHS 2017 presentation-standard flags;
   do not report estimates flagged 'suppress'.
9. PREGNANCY. NCHS body-measure estimates exclude pregnant participants (RIDEXPRG == 1).
10. MORTALITY. build_dataset(include_mortality=True) joins the public-use Linked Mortality File
   (follow-up through 2019; cycles 1999-2000..2017-2018). Restrict to ELIGSTAT == 1; time =
   PERMTH_INT (from interview) or PERMTH_EXM (from exam); event = MORTSTAT. Use survey_cox.
"""


def _ds(dataset_id: str) -> dict:
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset_id '{dataset_id}'. Active: {list(DATASETS)}")
    return DATASETS[dataset_id]


DIETARY_WEIGHTS = ("WTDRD1", "WTDR2D")  # 24-hour recall day-1 / two-day weights (no 2YR suffix)


def _weight_kind(col: str) -> str | None:
    for k in DIETARY_WEIGHTS:
        if col in (k, f"{k}PP"):
            return k
    m = re.match(r"^(WT[A-Z0-9]+?)(2YR|4YR|PRP)$", col)
    return m.group(1) if m else None


def _weight_col(frame_cols, kind: str, cycle: str, four_year: bool) -> str | None:
    if kind in DIETARY_WEIGHTS:
        opts = [f"{kind}PP"] if cycle == "2017-2020" else [kind]
    else:
        opts = [f"{kind}PRP"] if cycle == "2017-2020" else ([f"{kind}4YR", f"{kind}2YR"] if four_year else [f"{kind}2YR"])
    return next((c for c in opts if c in frame_cols), None)


import ast as _ast

_FUNCS = {"abs": np.abs, "log": np.log, "exp": np.exp, "sqrt": np.sqrt}


def _eval(df: pd.DataFrame, expr: str):
    """Evaluate a restricted expression over columns. Comparisons yield 0/1 floats (so they can be
    summed); & | ~ are logical; missing values propagate through arithmetic."""
    # pandas-style operators: & | ~ bind LOOSER than comparisons (unlike Python), so map them
    # to and / or / not, which have the intended precedence.
    src = re.sub(r"&&?", " and ", expr)
    src = re.sub(r"\|\|?", " or ", src)
    src = src.replace("~", " not ")
    try:
        tree = _ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}")
    refs = []

    def ev(n):
        if isinstance(n, _ast.Expression):
            return ev(n.body)
        if isinstance(n, _ast.Name):
            if n.id not in df.columns:
                raise ValueError(f"Unknown column '{n.id}'")
            refs.append(n.id)
            return df[n.id].astype(float)
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, _ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, _ast.USub):
                return -v
            if isinstance(n.op, _ast.UAdd):
                return v
            if isinstance(n.op, (_ast.Invert, _ast.Not)):
                return (~(pd.Series(v, index=df.index) != 0)).astype(float)
        if isinstance(n, _ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {_ast.Add: lambda: a + b, _ast.Sub: lambda: a - b, _ast.Mult: lambda: a * b,
                   _ast.Div: lambda: a / b, _ast.Mod: lambda: a % b, _ast.Pow: lambda: a ** b,
                   _ast.BitAnd: lambda: ((a != 0) & (b != 0)).astype(float),
                   _ast.BitOr: lambda: ((a != 0) | (b != 0)).astype(float)}
            if type(n.op) in ops:
                return ops[type(n.op)]()
        if isinstance(n, _ast.BoolOp):
            vals = [pd.Series(ev(v), index=df.index) != 0 for v in n.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(n.op, _ast.And) else (out | v)
            return out.astype(float)
        if isinstance(n, _ast.Compare):
            left = ev(n.left)
            res = None
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                f = {_ast.Gt: lambda x, y: x > y, _ast.GtE: lambda x, y: x >= y, _ast.Lt: lambda x, y: x < y,
                     _ast.LtE: lambda x, y: x <= y, _ast.Eq: lambda x, y: x == y, _ast.NotEq: lambda x, y: x != y}
                if type(op) not in f:
                    raise ValueError("Unsupported comparison")
                r = pd.Series(f[type(op)](left, right), index=df.index)
                res = r if res is None else (res & r)
                left = right
            return res.astype(float)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            return _FUNCS[n.func.id](ev(n.args[0]))
        raise ValueError(f"Unsupported syntax in expression: {_ast.dump(n)[:80]}")

    out = ev(tree)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index, dtype=float)
    return out, sorted(set(refs))


@mcp.tool()
def list_cycles() -> dict:
    """List NHANES cycles this server supports, their file-name conventions and weight rules."""
    rows = []
    for k, v in cat.CYCLES.items():
        rows.append({"cycle": k, "label": v.get("label", k), "file_example": cat.file_name("DEMO", k),
                     "years_for_weight_pooling": v["years"], "linked_mortality_available": bool(v["mort"])})
    return {"cycles": rows, "note": "2019-2020 alone is not released for standalone use; use 2017-2020 (P_ files)."}


@mcp.tool()
def analysis_guidance() -> str:
    """Rules for valid NHANES estimation that this server enforces. Read before analyzing."""
    return GUIDANCE


@mcp.tool()
def list_files(cycle: str, component: str) -> dict:
    """List data files for a cycle and component (Demographics, Dietary, Examination, Laboratory, Questionnaire)."""
    files = cat.list_files(cycle, component)
    return {"cycle": cycle, "component": component, "n_files": len(files), "files": files}


@mcp.tool()
def search_variables(query: str, cycles: list[str] | None = None, components: list[str] | None = None,
                     limit: int = 40) -> dict:
    """Search NHANES variable names/descriptions (regex, case-insensitive) across cycles/components.
    Defaults to the 2017-2018 cycle and all components."""
    cycles = cycles or ["2017-2018"]
    components = components or cat.COMPONENTS
    hits, errors = [], []
    pat = re.compile(query, re.I)
    for cy, comp in itertools.product(cycles, components):
        try:
            t = cat.variable_list(cy, comp)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cy}/{comp}: {e}")
            continue
        m = t[t["variable"].astype(str).str.contains(pat) | t["description"].astype(str).str.contains(pat)]
        hits.extend(m.to_dict("records"))
    return {"query": query, "n_hits": len(hits), "results": hits[:limit], "truncated": len(hits) > limit,
            "errors": errors}


@mcp.tool()
def describe_variable(variable: str, table: str, cycle: str) -> dict:
    """Codebook entry (label, question text, target population, value codes) plus observed distribution.
    Flags codes that look like refused/don't-know so they can be set to missing."""
    variable = variable.upper()
    df, meta = cat.load_xpt(table, cycle)
    if variable not in df.columns:
        raise ValueError(f"{variable} not in {meta['file']}. Columns: {list(df.columns)[:60]}")
    try:
        cb = cat.codebook(table, cycle).get(variable, {})
    except Exception as e:  # noqa: BLE001
        cb = {"codebook_error": str(e)}
    s = df[variable]
    out = {"variable": variable, "file": meta["file"], "cycle": cycle,
           "label": cb.get("label") or meta["labels"].get(variable), "question": cb.get("question"),
           "target": cb.get("target"), "value_codes": cb.get("values", [])[:40],
           "n_rows": len(s), "n_missing": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s):
        out["summary"] = {k: float(v) for k, v in s.describe().items()}
        if s.nunique() <= 25:
            out["observed_values"] = {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}
    sentinel = []
    for v in cb.get("values", []):
        if MISSING_WORDS.search(v["description"]):
            try:
                sentinel.append(float(v["code"]))
            except ValueError:
                pass
    out["suggested_missing_codes"] = sentinel
    return out


@mcp.tool()
def build_dataset(cycles: list[str], tables: list[str], variables: list[str] | None = None,
                  include_mortality: bool = False) -> dict:
    """Build an analysis-ready dataset: DEMO universe (all participants, needed for valid domain
    estimation) left-joined to the requested tables on SEQN, for one or more cycles.

    tables: base names without cycle suffix, e.g. ["BMX", "TCHOL", "BPQ"] (DEMO is always included).
    variables: columns to keep from those tables (default: all). Weight/design variables are always kept.
    The analysis weight is chosen and rescaled automatically -> column WT_ANALYSIS.
    """
    if not cycles:
        raise ValueError("Give at least one cycle.")
    for c in cycles:
        cat.check_cycle(c)
    warnings = []
    if "2017-2020" in cycles and "2017-2018" in cycles:
        raise ValueError("2017-2020 (P_ files) already contains 2017-2018; do not combine them.")
    if "2021-2023" in cycles and len(cycles) > 1:
        warnings.append("2021-2023 used a redesigned sample and methods; NCHS cautions against pooling it "
                        "with earlier cycles. Prefer separate estimates per cycle.")
    four_year = "1999-2000" in cycles and "2001-2002" in cycles
    total_years = sum(cat.CYCLES[c]["years"] for c in cycles)
    wanted = {v.upper() for v in variables} if variables else None
    tables = [t.upper() for t in tables if t.upper() != "DEMO"]

    frames, provenance = [], []
    levels, subsample_kinds = set(), {}
    for cy in cycles:
        demo, dmeta = cat.load_xpt("DEMO", cy)
        provenance.append({"file": dmeta["file"], "rows": dmeta["n_rows"]})
        wcols = [c for c in demo.columns if _weight_kind(c)]
        keep = [c for c in DEMO_CORE if c in demo.columns] + wcols
        if wanted:
            keep += [c for c in demo.columns if c in wanted and c not in keep]
        else:
            keep = list(demo.columns)
        merged = demo[keep].copy()
        mec_col = _weight_col(demo.columns, "WTMEC", cy, False)
        mec_pos = set(demo.loc[demo[mec_col] > 0, "SEQN"]) if mec_col else set()
        for t in tables:
            try:
                tdf, tmeta = cat.load_xpt(t, cy)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{cy}: could not load {cat.file_name(t, cy)} ({e}).")
                continue
            if tdf["SEQN"].duplicated().any():
                raise ValueError(
                    f"{tmeta['file']} has several rows per participant (long format, e.g. prescriptions or "
                    "individual foods). Merging it would duplicate people and corrupt the weights. Build the "
                    "dataset without it, then use flag_from_long_table to add a per-person indicator.")
            provenance.append({"file": tmeta["file"], "rows": tmeta["n_rows"]})
            t_w = [c for c in tdf.columns if _weight_kind(c)]
            if cat.base_name(t).startswith("DR1") or cat.base_name(t).startswith("DS1"):
                t_w = [c for c in t_w if _weight_kind(c) != "WTDR2D"]  # day-1 variables -> day-1 weight
            cols = [c for c in tdf.columns if c != "SEQN" and (wanted is None or c in wanted)]
            cols = [c for c in cols if c not in merged.columns]
            if cols:
                share_int_only = 1 - (tdf["SEQN"].isin(mec_pos).mean() if mec_pos else 1)
                lvl = "interview" if share_int_only > 0.005 else "mec"
                levels.add(lvl)
                for c in t_w:
                    subsample_kinds.setdefault(_weight_kind(c), set()).add(t)
            add = list(dict.fromkeys(cols + [c for c in t_w if c not in merged.columns]))
            merged = merged.merge(tdf[["SEQN"] + add], on="SEQN", how="left")
        if include_mortality:
            try:
                mort = cat.mortality(cy)
                merged = merged.merge(mort, on="SEQN", how="left")
                provenance.append({"file": f"LMF {cy} (through 2019)", "rows": len(mort)})
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{cy}: mortality linkage unavailable ({e}).")
        merged["CYCLE"] = cy
        frames.append(merged)

    # ---- choose the weight kind (most restrictive) ----
    if subsample_kinds:
        if len(subsample_kinds) > 1:
            counts = {}
            for k in subsample_kinds:
                n = sum(int((f[_weight_col(f.columns, k, f["CYCLE"].iat[0], four_year)] > 0).sum())
                        for f in frames if _weight_col(f.columns, k, f["CYCLE"].iat[0], four_year))
                counts[k] = n
            kind = min(counts, key=counts.get)
            warnings.append(f"Multiple subsample weights present {counts}; using the most restrictive ({kind}). "
                            "Consider analyzing subsample variables separately.")
        else:
            kind = next(iter(subsample_kinds))
        reason = f"subsample weight from {sorted(subsample_kinds[kind])}"
    elif "mec" in levels:
        kind, reason = "WTMEC", "at least one variable comes from an MEC-examined component"
    else:
        kind, reason = "WTINT", "all variables come from the household interview"

    out_frames = []
    for f in frames:
        cy = f["CYCLE"].iat[0]
        wc = _weight_col(f.columns, kind, cy, four_year and cy in ("1999-2000", "2001-2002"))
        if wc is None:
            raise ValueError(f"Weight {kind} not found for cycle {cy}.")
        span = 4.0 if wc.endswith("4YR") else cat.CYCLES[cy]["years"]
        factor = (span / total_years) if wc.endswith("4YR") else (cat.CYCLES[cy]["years"] / total_years)
        f = f.copy()
        f["WT_ANALYSIS"] = f[wc].fillna(0) * factor
        f["WT_SOURCE"] = f"{wc} x {factor:.4f}"
        out_frames.append(f)
    df = pd.concat(out_frames, ignore_index=True)
    df["SDMVSTRA_U"] = df["CYCLE"] + "|" + df["SDMVSTRA"].astype("Int64").astype(str)
    df["SDMVPSU"] = df["SDMVPSU"].astype("Int64")

    dsid = uuid.uuid4().hex[:8]
    labels = {}
    DATASETS[dsid] = {"df": df, "cycles": cycles, "tables": ["DEMO"] + tables, "weight_kind": kind,
                      "weight_reason": reason, "warnings": warnings, "provenance": provenance,
                      "derived": {}, "labels": labels}
    return {"dataset_id": dsid, "n_rows": len(df), "n_positive_weight": int((df["WT_ANALYSIS"] > 0).sum()),
            "cycles": cycles, "weight": {"kind": kind, "reason": reason,
                                         "per_cycle": df.groupby("CYCLE")["WT_SOURCE"].first().to_dict()},
            "columns": [c for c in df.columns if c not in ("WT_SOURCE",)],
            "files": provenance, "warnings": warnings}


@mcp.tool()
def describe_dataset(dataset_id: str) -> dict:
    """Columns, non-missing counts, weight choice, derived-variable definitions and warnings."""
    ds = _ds(dataset_id)
    df = ds["df"]
    return {"dataset_id": dataset_id, "n_rows": len(df), "cycles": ds["cycles"], "tables": ds["tables"],
            "weight_kind": ds["weight_kind"], "weight_reason": ds["weight_reason"],
            "non_missing": {c: int(df[c].notna().sum()) for c in df.columns if c not in ("WT_SOURCE",)},
            "derived": ds["derived"], "warnings": ds["warnings"]}


@mcp.tool()
def set_missing(dataset_id: str, variable: str, codes: list[float]) -> dict:
    """Recode sentinel values (e.g. 7, 9, 77, 99 for refused/don't know) to missing."""
    ds = _ds(dataset_id)
    v = variable.upper()
    before = int(ds["df"][v].isin(codes).sum())
    ds["df"].loc[ds["df"][v].isin(codes), v] = np.nan
    ds["derived"].setdefault("_missing_recodes", {})[v] = codes
    return {"variable": v, "set_to_missing": before}


@mcp.tool()
def derive_variable(dataset_id: str, name: str, expression: str, missing: str = "any") -> dict:
    """Create a variable from an expression over existing columns, e.g.
    name='OBESE', expression='BMXBMI >= 30'  (booleans become 0/1).
    missing: 'any' -> result missing if ANY referenced column is missing (default, conservative);
             'all' -> missing only if ALL referenced columns are missing (for OR-type definitions);
             'none' -> no propagation."""
    ds = _ds(dataset_id)
    df = ds["df"]
    name = name.upper()
    res, refs = _eval(df, expression)
    res = res.astype(float).copy()
    if refs and missing != "none":
        na = df[refs].isna()
        mask = na.any(axis=1) if missing == "any" else na.all(axis=1)
        res[mask] = np.nan
    df[name] = res
    ds["derived"][name] = {"expression": expression, "missing_rule": missing}
    nn = res.dropna()
    return {"name": name, "n_non_missing": int(nn.size),
            "values": {str(k): int(v) for k, v in nn.value_counts().head(10).items()} if nn.nunique() <= 10 else None,
            "mean_unweighted": float(nn.mean()) if nn.size else None}


def _groups(df: pd.DataFrame, by: list[str] | None):
    if not by:
        yield {}, np.ones(len(df), dtype=bool)
        return
    by = [b.upper() for b in by]
    levels = [sorted(df[b].dropna().unique()) for b in by]
    for combo in itertools.product(*levels):
        m = np.ones(len(df), dtype=bool)
        for b, v in zip(by, combo):
            m &= (df[b] == v).to_numpy()
        yield {b: (float(v) if isinstance(v, (int, float, np.number)) else v) for b, v in zip(by, combo)}, m


@mcp.tool()
def survey_estimate(dataset_id: str, variable: str, statistic: str = "mean", domain: str | None = None,
                    by: list[str] | None = None, age_adjust: str | None = None) -> dict:
    """Design-based estimate (Taylor linearization) of a mean, proportion (0/1 variable) or total.
    domain: expression defining the subpopulation, e.g. 'RIDAGEYR >= 20 & RIDEXPRG != 1'
            (the design is NOT subset; out-of-domain records get zero weight).
    by: grouping variables, e.g. ['RIAGENDR'].
    age_adjust: 'nchs_adults_20plus' for NCHS direct age standardization (2000 census; 20-39/40-59/60+).
    Proportions come with Korn-Graubard CIs and NCHS reliability flags."""
    ds = _ds(dataset_id)
    df = ds["df"]
    variable = variable.upper()
    dmask = np.ones(len(df), dtype=bool)
    if domain:
        res, _ = _eval(df, domain)
        dmask = (res.fillna(0) != 0).to_numpy()
    results = []
    for g, gm in _groups(df, by):
        r = sv.estimate(df, variable, "WT_ANALYSIS", statistic, dmask & gm, age_adjust)
        r["group"] = g
        results.append(r)
    return {"dataset_id": dataset_id, "variable": variable, "statistic": statistic, "domain": domain,
            "age_adjusted": age_adjust, "weight": ds["weight_kind"],
            "weight_per_cycle": df.groupby("CYCLE")["WT_SOURCE"].first().to_dict(),
            "results": results, "warnings": ds["warnings"]}


@mcp.tool()
def survey_regression(dataset_id: str, outcome: str, predictors: list[str], family: str = "gaussian",
                      domain: str | None = None) -> dict:
    """Survey-weighted linear ('gaussian') or logistic ('binomial') regression with design-based SEs.
    Categorical predictors must be dummy-coded first with derive_variable."""
    ds = _ds(dataset_id)
    df = ds["df"]
    dmask = None
    if domain:
        res, _ = _eval(df, domain)
        dmask = (res.fillna(0) != 0).to_numpy()
    r = sv.regression(df, outcome.upper(), [p.upper() for p in predictors], "WT_ANALYSIS", family, dmask)
    r.update(dataset_id=dataset_id, domain=domain, weight=ds["weight_kind"], warnings=ds["warnings"])
    return r


@mcp.tool()
def flag_from_long_table(dataset_id: str, table: str, columns: list[str], pattern: str, name: str) -> dict:
    """Add a per-person 0/1 indicator from a file with several rows per participant (e.g. RXQ_RX
    prescriptions, DR1IFF foods). Flag = 1 if ANY row's value in ANY of `columns` matches the regex
    `pattern` (case-insensitive), e.g. table='RXQ_RX', columns=['RXDRSC1','RXDRSC2','RXDRSC3'],
    pattern='^G40' for ICD-10 epilepsy as the reason for use. People absent from the file are missing."""
    ds = _ds(dataset_id)
    df = ds["df"]
    name = name.upper()
    cols = [c.upper() for c in columns]
    pat = re.compile(pattern, re.I)
    flags, per_cycle = [], {}
    for cy in ds["cycles"]:
        tdf, meta = cat.load_xpt(table, cy)
        missing = [c for c in cols if c not in tdf.columns]
        if missing:
            raise ValueError(f"{meta['file']} lacks columns {missing}. Columns: {list(tdf.columns)[:40]}")
        hit = np.zeros(len(tdf), dtype=bool)
        for c in cols:
            hit |= tdf[c].astype(str).str.contains(pat, na=False).to_numpy()
        g = pd.DataFrame({"SEQN": tdf["SEQN"], "h": hit}).groupby("SEQN")["h"].any().astype(float)
        f = g.rename(name).reset_index()
        f["CYCLE"] = cy
        flags.append(f)
        per_cycle[cy] = {"file": meta["file"], "people_in_file": int(len(g)), "flagged": int(g.sum())}
        ds["provenance"].append({"file": meta["file"], "rows": meta["n_rows"], "used_for": name})
    allf = pd.concat(flags, ignore_index=True)
    if name in df.columns:
        df.drop(columns=[name], inplace=True)
    merged = df.merge(allf, on=["SEQN", "CYCLE"], how="left")
    ds["df"] = merged
    ds["derived"][name] = {"from_table": table, "columns": cols, "pattern": pattern}
    return {"name": name, "per_cycle": per_cycle, "n_flagged": int((merged[name] == 1).sum()),
            "n_not_flagged": int((merged[name] == 0).sum()), "n_missing": int(merged[name].isna().sum())}


@mcp.tool()
def survey_frequency(dataset_id: str, variable: str, domain: str | None = None, by: list[str] | None = None,
                     labels: dict | None = None) -> dict:
    """Weighted distribution of a categorical variable (Table 1 style): for each level, unweighted n,
    weighted percent, SE and Korn-Graubard CI, within the domain and optionally by group.
    labels: optional {code: label} map, e.g. {"1": "Male", "2": "Female"}."""
    ds = _ds(dataset_id)
    df = ds["df"]
    v = variable.upper()
    dmask = np.ones(len(df), dtype=bool)
    if domain:
        res, _ = _eval(df, domain)
        dmask = (res.fillna(0) != 0).to_numpy()
    labels = {str(k): val for k, val in (labels or {}).items()}
    out = []
    for g, gm in _groups(df, by):
        m = dmask & gm & df[v].notna().to_numpy()
        levels = sorted(pd.unique(df.loc[m, v]))
        rows = []
        for lv in levels:
            tmp = "__LEVEL__"
            df[tmp] = np.where(df[v].isna(), np.nan, (df[v] == lv).astype(float))
            r = sv.estimate(df, tmp, "WT_ANALYSIS", "proportion", m)
            key = str(int(lv)) if isinstance(lv, (int, float, np.number)) and float(lv).is_integer() else str(lv)
            rows.append({"level": key, "label": labels.get(key), "n": int(((df[v] == lv).to_numpy() & m).sum()),
                         "pct": round(100 * r["estimate"], 2), "se": round(100 * r["se"], 2),
                         "ci": [round(100 * r["ci_low"], 1), round(100 * r["ci_high"], 1)],
                         "reliability": r["nchs_reliability"]})
            df.drop(columns=[tmp], inplace=True)
        out.append({"group": g, "n_total": int(m.sum()), "n_missing_in_domain": int((dmask & gm & df[v].isna().to_numpy()).sum()),
                    "levels": rows})
    return {"dataset_id": dataset_id, "variable": v, "domain": domain, "weight": ds["weight_kind"],
            "weight_per_cycle": df.groupby("CYCLE")["WT_SOURCE"].first().to_dict(), "results": out}


@mcp.tool()
def survey_cox(dataset_id: str, time: str, event: str, predictors: list[str], domain: str | None = None) -> dict:
    """Survey-weighted Cox proportional hazards model (Breslow ties, Binder linearized variance;
    same estimator as SUDAAN SURVIVAL / R svycoxph). Typical use with include_mortality=True:
    time='PERMTH_INT' (or PERMTH_EXM), event='MORTSTAT', domain including 'ELIGSTAT == 1'.
    Categorical predictors must be dummy-coded first with derive_variable."""
    ds = _ds(dataset_id)
    df = ds["df"]
    dmask = None
    if domain:
        res, _ = _eval(df, domain)
        dmask = (res.fillna(0) != 0).to_numpy()
    r = sv.coxph(df, time.upper(), event.upper(), [p.upper() for p in predictors], "WT_ANALYSIS", dmask)
    for k in ("_score_residuals", "_beta", "_info_inv"):
        r.pop(k)
    r.update(dataset_id=dataset_id, domain=domain, weight=ds["weight_kind"],
             weight_per_cycle=df.groupby("CYCLE")["WT_SOURCE"].first().to_dict(), warnings=ds["warnings"])
    return r


@mcp.tool()
def export_dataset(dataset_id: str, filename: str | None = None) -> dict:
    """Write the analytic dataset (with WT_ANALYSIS, SDMVSTRA_U, SDMVPSU) to CSV plus a provenance sidecar."""
    ds = _ds(dataset_id)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    fn = EXPORT_DIR / (filename or f"nhanes_{dataset_id}.csv")
    ds["df"].to_csv(fn, index=False)
    side = fn.with_suffix(".provenance.json")
    pd.Series({k: ds[k] for k in ("cycles", "tables", "weight_kind", "weight_reason", "warnings",
                                   "provenance", "derived")}).to_json(side, indent=2)
    return {"csv": str(fn), "provenance": str(side), "n_rows": len(ds["df"])}


def main():
    mcp.run()


if __name__ == "__main__":
    main()
