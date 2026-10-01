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
   fasting / phlebotomy / surplus-serum / other subsample files -> that file's own weight
   (e.g. WTSAF2YR, WTPH2YR, WTSSMC2Y/WTSSMC4Y). build_dataset selects this automatically and
   explains its choice; build_dataset(weight=...) or set_weight() override it, and any override
   is reported with every result.
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
   Other age ranges: pass a custom standard {"groups": [[a0,a1],...], "population": [...]} and name
   its source in "note".
8. RELIABILITY. Proportions report Korn-Graubard CIs and NCHS 2017 presentation-standard flags;
   do not report estimates flagged 'suppress'.
9. PREGNANCY. NCHS body-measure estimates exclude pregnant participants (RIDEXPRG == 1).
10. MORTALITY. build_dataset(include_mortality=True) joins the public-use Linked Mortality File
   (follow-up through 2019; cycles 1999-2000..2017-2018). Restrict to ELIGSTAT == 1; time =
   PERMTH_INT (from interview) or PERMTH_EXM (from exam); event = MORTSTAT. Use survey_cox.
11. PRESENTING RESULTS. show_results returns a text summary plus structured results (and the
   interactive Results Explorer in MCP Apps hosts when that add-on is installed). Always fill in the analysis plan (outcome_label,
   population_label, rationale) so the user can see how the question was interpreted, and pass
   published comparators as benchmarks when they exist.
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
    m = WEIGHT_RE.match(col)
    return m.group(1) if m else None


WEIGHT_RE = re.compile(r"^(WT[A-Z0-9]+?)(2YR|4YR|2Y|4Y|PRP)$")
FOUR_YEAR_RE = re.compile(r"4(YR|Y)$")
PROTECTED_COLS = {"WT_ANALYSIS", "WT_SOURCE", "SDMVSTRA", "SDMVSTRA_U", "SDMVPSU", "CYCLE", "SEQN"}


def _weight_col(frame_cols, kind: str, cycle: str, four_year: bool) -> str | None:
    if kind in DIETARY_WEIGHTS:
        opts = [f"{kind}PP"] if cycle == "2017-2020" else [kind]
    else:
        two = [f"{kind}2YR", f"{kind}2Y"]
        opts = [f"{kind}PRP"] if cycle == "2017-2020" else ([f"{kind}4YR", f"{kind}4Y"] + two if four_year else two)
    return next((c for c in opts if c in frame_cols), None)


import ast as _ast

_FUNCS = {"abs": np.abs, "log": np.log, "exp": np.exp, "sqrt": np.sqrt}
# Functions that deliberately look at missingness. Columns referenced only inside them are exempt
# from derive_variable's missing-propagation rule (otherwise coalesce() could never fill anything).
_NA_FUNCS = {"coalesce", "fillna", "isna", "notna", "where"}


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
    refs, na_refs = [], []
    depth = [0]

    def ev(n):
        if isinstance(n, _ast.Expression):
            return ev(n.body)
        if isinstance(n, _ast.Name):
            if n.id not in df.columns:
                raise ValueError(f"Unknown column '{n.id}'")
            (na_refs if depth[0] else refs).append(n.id)
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
                r = pd.Series(f[type(op)](left, right), index=df.index).astype(float)
                if depth[0]:
                    # Inside missing-aware functions a comparison with a missing operand is itself
                    # missing (not False), so where(cond, ...) and coalesce() see an unknown condition
                    # as unknown. Outside them, the derive_variable missing rule / domain handling apply.
                    na = pd.Series(left, index=df.index).isna() | pd.Series(right, index=df.index).isna()
                    r[na] = np.nan
                if res is None:
                    res = r
                else:
                    both = ((res != 0) & (r != 0)).astype(float)
                    both[res.isna() | r.isna()] = np.nan
                    res = both
                left = right
            return res.astype(float)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id in _FUNCS and len(n.args) == 1:
            return _FUNCS[n.func.id](ev(n.args[0]))
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id in _NA_FUNCS and not n.keywords:
            fn = n.func.id
            depth[0] += 1
            try:
                args = [pd.Series(ev(a), index=df.index, dtype=float) for a in n.args]
            finally:
                depth[0] -= 1
            if fn == "coalesce" and len(args) >= 2:
                out = args[0]
                for a in args[1:]:
                    out = out.fillna(a)
                return out
            if fn == "fillna" and len(args) == 2:
                return args[0].fillna(args[1])
            if fn in ("isna", "notna") and len(args) == 1:
                return (args[0].isna() if fn == "isna" else args[0].notna()).astype(float)
            if fn == "where" and len(args) == 3:
                cond = args[0]
                out = pd.Series(np.where(cond.fillna(0) != 0, args[1], args[2]), index=df.index, dtype=float)
                out[cond.isna()] = np.nan
                return out
            raise ValueError(f"Wrong number of arguments for {fn}()")
        raise ValueError(f"Unsupported syntax in expression: {_ast.dump(n)[:80]}")

    out = ev(tree)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index, dtype=float)
    return out, sorted(set(refs))  # na_refs deliberately excluded from missing propagation


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
                  include_mortality: bool = False, weight: str | None = None) -> dict:
    """Build an analysis-ready dataset: DEMO universe (all participants, needed for valid domain
    estimation) left-joined to the requested tables on SEQN, for one or more cycles.

    tables: base names without cycle suffix, e.g. ["BMX", "TCHOL", "BPQ"] (DEMO is always included).
    variables: columns to keep from those tables (default: all). Weight/design variables are always kept.
    The analysis weight is chosen and rescaled automatically -> column WT_ANALYSIS. Subsample weights
    carried by a file (fasting WTSAF2YR, phlebotomy WTPH2YR, surplus serum WTSSMC2Y/4Y, ...) are detected.
    weight: optional weight kind to force, e.g. 'WTSSMC' or 'WTMEC' (without the 2YR/4YR suffix). The server
    warns if it is less restrictive than the automatic choice.
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

    auto_kind = kind
    if weight:
        forced = _weight_kind(weight.upper()) or weight.upper()
        if not all(_weight_col(f.columns, forced, f["CYCLE"].iat[0],
                               four_year and f["CYCLE"].iat[0] in ("1999-2000", "2001-2002")) for f in frames):
            raise ValueError(f"Weight '{forced}' not found in every cycle. Weight columns present: "
                             f"{sorted({c for f in frames for c in f.columns if _weight_kind(c)})}")
        if forced != auto_kind:
            warnings.append(f"Weight set by caller to {forced}; automatic choice was {auto_kind} ({reason}). "
                            "Check that the forced weight is appropriate for every variable analyzed.")
        kind, reason = forced, f"set by caller (automatic choice: {auto_kind})"

    out_frames = []
    for f in frames:
        cy = f["CYCLE"].iat[0]
        wc = _weight_col(f.columns, kind, cy, four_year and cy in ("1999-2000", "2001-2002"))
        if wc is None:
            raise ValueError(f"Weight {kind} not found for cycle {cy}.")
        factor = (4.0 / total_years) if FOUR_YEAR_RE.search(wc) else (cat.CYCLES[cy]["years"] / total_years)
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
def set_weight(dataset_id: str, expression: str, reason: str) -> dict:
    """Replace the analysis weight with a custom expression over existing columns, e.g.
    'coalesce(WTSSMC4Y * 4/6, WTSSMC2Y * 2/6)'. Use only when build_dataset cannot select the right
    weight. The change is recorded and reported (weight kind 'CUSTOM' + a warning) in every later result.
    Negative or missing weights become 0 (out of sample)."""
    ds = _ds(dataset_id)
    df = ds["df"]
    res, _ = _eval(df, expression)
    res = res.astype(float).fillna(0.0)
    if (res < 0).any():
        raise ValueError("Weight expression produced negative values.")
    old = ds["weight_kind"]
    df["WT_ANALYSIS"] = res
    df["WT_SOURCE"] = f"custom: {expression}"
    ds["weight_kind"] = "CUSTOM"
    ds["weight_reason"] = f"set_weight: {reason} (replaced {old})"
    ds["warnings"].append(f"Analysis weight overridden by set_weight: '{expression}' ({reason}). "
                          f"Previous automatic weight: {old}.")
    ds["derived"]["WT_ANALYSIS"] = {"expression": expression, "reason": reason, "replaced": old}
    return {"weight": "CUSTOM", "expression": expression, "n_positive_weight": int((res > 0).sum()),
            "sum_weights_by_cycle": {k: float(v) for k, v in df.groupby("CYCLE")["WT_ANALYSIS"].sum().items()}}


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
             'none' -> no propagation.
    Missing-aware functions: coalesce(a, b, ...), fillna(x, value), isna(x), notna(x),
    where(cond, a, b). Columns used only inside them are exempt from the missing rule.
    Design columns (WT_ANALYSIS, strata, PSU, SEQN, CYCLE) cannot be overwritten here; use set_weight."""
    ds = _ds(dataset_id)
    df = ds["df"]
    name = name.upper()
    if name in PROTECTED_COLS:
        raise ValueError(f"'{name}' is a design column and cannot be overwritten with derive_variable. "
                         "To change the analysis weight use build_dataset(weight=...) or set_weight(), "
                         "which record the change in every result.")
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
                    by: list[str] | None = None, age_adjust: str | dict | None = None) -> dict:
    """Design-based estimate (Taylor linearization) of a mean, proportion (0/1 variable) or total.
    domain: expression defining the subpopulation, e.g. 'RIDAGEYR >= 20 & RIDEXPRG != 1'
            (the design is NOT subset; out-of-domain records get zero weight).
    by: grouping variables, e.g. ['RIAGENDR'].
    age_adjust: a preset name ('nchs_adults_20plus': 2000 census, 20-39/40-59/60+) or a custom standard
        {"groups": [[6,11],[12,19],[20,29]], "population": [24.6, 32.5, 38.3]} (or "proportions"),
        optionally "age_var" (default RIDAGEYR) and "note" naming the source of the standard.
    Proportions come with Korn-Graubard CIs and NCHS reliability flags."""
    ds = _ds(dataset_id)
    df = ds["df"]
    variable = variable.upper()
    dmask = np.ones(len(df), dtype=bool)
    if domain:
        res, _ = _eval(df, domain)
        dmask = (res.fillna(0) != 0).to_numpy()
    std = sv.resolve_age_standard(age_adjust) if age_adjust else None
    extra_warn = []
    if std is not None:
        age = df[std["age_var"]].to_numpy(dtype=float)
        covered = np.zeros(len(df), dtype=bool)
        for a0, a1 in std["groups"]:
            covered |= (age >= a0) & (age <= a1)
        n_out = int((dmask & ~covered & df[variable].notna().to_numpy()).sum())
        if n_out:
            extra_warn.append(f"{n_out} in-domain records fall outside the age-standard groups and are "
                              "excluded from the age-adjusted estimate.")
    results = []
    for g, gm in _groups(df, by):
        r = sv.estimate(df, variable, "WT_ANALYSIS", statistic, dmask & gm, std)
        r["group"] = g
        results.append(r)
    return {"dataset_id": dataset_id, "variable": variable, "statistic": statistic, "domain": domain,
            "age_adjusted": (std or {}).get("note"), "weight": ds["weight_kind"],
            "weight_per_cycle": df.groupby("CYCLE")["WT_SOURCE"].first().to_dict(),
            "results": results, "warnings": ds["warnings"] + extra_warn}


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


# ----------------------------------------------------------------------------------------------
# MCP App: Results Explorer (interactive view rendered by MCP Apps hosts such as Claude Desktop).
# The view is a self-contained HTML resource; hosts without MCP Apps support get the text summary.
# ----------------------------------------------------------------------------------------------
from mcp import types as _mt  # noqa: E402

VIEW_URI = "ui://nhanes-mcp/results-explorer.html"

KNOWN_NAMES = {"RIAGENDR": "Sex", "RIDRETH1": "Race and Hispanic origin", "RIDRETH3": "Race and Hispanic origin",
               "CYCLE": "Survey cycle", "DMDEDUC2": "Education (adults 20+)"}
KNOWN_LABELS = {
    "RIAGENDR": {"1": "Men", "2": "Women"},
    "RIDRETH1": {"1": "Mexican American", "2": "Other Hispanic", "3": "Non-Hispanic White",
                 "4": "Non-Hispanic Black", "5": "Other / multiracial"},
    "RIDRETH3": {"1": "Mexican American", "2": "Other Hispanic", "3": "Non-Hispanic White",
                 "4": "Non-Hispanic Black", "6": "Non-Hispanic Asian", "7": "Other / multiracial"},
    "DMDEDUC2": {"1": "Less than 9th grade", "2": "9-11th grade", "3": "High school / GED",
                 "4": "Some college / AA", "5": "College graduate or above"},
}


def _load_explorer():
    """Optional add-on: the NHANES Results Explorer view (separate package, PolyForm Noncommercial,
    Black Swan Causal Labs). Found if installed, via NHANES_MCP_EXPLORER_PATH, or as a sibling folder
    named nhanes-mcp-explorer. Without it, show_results returns text only."""
    import importlib
    import sys
    try:
        return importlib.import_module("nhanes_mcp_explorer")
    except ImportError:
        pass
    cands = [cat.os.environ.get("NHANES_MCP_EXPLORER_PATH"),
             str(Path(__file__).resolve().parents[2] / "nhanes-mcp-explorer")]
    for c in cands:
        if c and (Path(c) / "nhanes_mcp_explorer" / "__init__.py").exists():
            sys.path.insert(0, c)
            try:
                return importlib.import_module("nhanes_mcp_explorer")
            except ImportError:
                sys.path.remove(c)
    return None


EXPLORER = _load_explorer()
_SHOW_META = {"ui": {"resourceUri": VIEW_URI}, "ui/resourceUri": VIEW_URI} if EXPLORER else None

if EXPLORER:
    @mcp.resource(VIEW_URI, name="results_explorer", title="NHANES Results Explorer",
                  description="Interactive view for show_results (MCP Apps). NHANES Results Explorer add-on, "
                              "Black Swan Causal Labs, PolyForm Noncommercial 1.0.0.",
                  mime_type="text/html;profile=mcp-app", meta={"ui": {"prefersBorder": False}})
    def results_explorer_view() -> str:
        return EXPLORER.view_html()


def _key(v) -> str:
    if isinstance(v, (int, float, np.number)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def _row(r: dict, level=None, label=None) -> dict:
    return {"level": level, "label": label, "estimate": r.get("estimate"), "se": r.get("se"),
            "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"), "n": r.get("n_unweighted"),
            "design_df": r.get("design_df"), "reliability": r.get("nchs_reliability"),
            "flags": r.get("reliability_flags", []), "ci_method": r.get("ci_method")}


@mcp.tool(meta=_SHOW_META)
def show_results(dataset_id: str, variable: str, statistic: str = "proportion", domain: str | None = None,
                 by: list[str] | None = None, age_adjust: str | dict | None = None,
                 title: str | None = None, question: str | None = None,
                 outcome_label: str | None = None, population_label: str | None = None,
                 group_names: dict | None = None, group_labels: dict | None = None,
                 rationale: str | None = None, benchmarks: list[dict] | None = None) -> _mt.CallToolResult:
    """Present design-based results: headline estimate, subgroup panels, the analysis plan (so the user
    can check how the question was interpreted), server warnings, NCHS benchmarks and provenance.
    Returns a text summary plus structured data; when the optional NHANES Results Explorer add-on is
    installed, MCP Apps hosts also render it as an interactive view.

    Same estimation arguments as survey_estimate (dataset_id, variable, statistic, domain, age_adjust);
    by: one panel per grouping variable, e.g. ['RIAGENDR', 'RIDRETH3', 'AGEGRP'] (age-adjustment is
    skipped for age groupings -- names starting with AGE or RIDAGE, or containing _AGE -- whose panels
    are age-specific).
    title / question / outcome_label / population_label / rationale: plain-language analysis plan.
    group_names: {"AGEGRP": "Age group"}; group_labels: {"AGEGRP": {"1": "20-39", ...}} (common NHANES
    demographics are labeled automatically).
    benchmarks: published comparators, e.g. [{"label": "Obesity 2021-23", "published": 40.3,
    "source_label": "NCHS Data Brief 508", "source_url": "https://...", "by": "RIAGENDR", "level": "1"}]
    (published in percent for proportions; omit by/level for the overall estimate)."""
    ds = _ds(dataset_id)
    variable = variable.upper()
    by = [b.upper() for b in (by or [])]
    names = {**KNOWN_NAMES, **{k.upper(): v for k, v in (group_names or {}).items()}}
    labels = {k: dict(v) for k, v in KNOWN_LABELS.items()}
    for k, v in (group_labels or {}).items():
        labels[k.upper()] = {str(kk): vv for kk, vv in v.items()}

    base = survey_estimate(dataset_id, variable, statistic, domain, None, age_adjust)
    overall = _row(base["results"][0])
    warnings = list(base["warnings"])
    age_note = base.get("age_adjusted")
    if age_adjust and base["results"][0].get("age_adjustment"):
        overall["age_cells"] = base["results"][0]["age_adjustment"]["cells"]

    panels = []
    for b in by:
        is_age = b.startswith(("AGE", "RIDAGE")) or "_AGE" in b  # not RIAGENDR (sex)
        res = survey_estimate(dataset_id, variable, statistic, domain, [b], None if is_age else age_adjust)
        rows = []
        for r in res["results"]:
            lv = _key(r["group"][b])
            if not r.get("n_unweighted"):
                continue
            rows.append(_row(r, lv, labels.get(b, {}).get(lv, lv)))
        for w in res["warnings"]:
            if w not in warnings:
                warnings.append(w)
        panels.append({"var": b, "name": names.get(b, b), "age_adjusted": bool(age_adjust) and not is_age,
                       "age_specific": is_age,
                       "rows": rows})

    bench = []
    for bm in benchmarks or []:
        tgt = None
        if bm.get("by"):
            p = next((p for p in panels if p["var"] == str(bm["by"]).upper()), None)
            if p:
                tgt = next((r for r in p["rows"] if r["level"] == _key(bm.get("level"))), None)
        else:
            tgt = overall
        server = None if tgt is None or tgt["estimate"] is None else (
            round(100 * tgt["estimate"], 1) if statistic == "proportion" else round(tgt["estimate"], 2))
        pub = bm.get("published")
        diff = None if server is None or pub is None else round(server - float(pub), 1)
        bench.append({"label": bm.get("label"), "published": pub, "server": server, "difference": diff,
                      "source_label": bm.get("source_label"), "source_url": bm.get("source_url")})

    df = ds["df"]
    data = {
        "kind": "nhanes-results", "version": 1, "tool_args": {
            "dataset_id": dataset_id, "variable": variable, "statistic": statistic, "domain": domain,
            "by": by, "age_adjust": age_adjust},
        "title": title or f"{outcome_label or variable}", "question": question,
        "plan": {"outcome": outcome_label or variable, "variable": variable, "statistic": statistic,
                 "population": population_label or domain or "All participants", "domain": domain,
                 "groups": [p["name"] for p in panels], "age_standard": age_note, "rationale": rationale},
        "design": {"cycles": ds["cycles"], "tables": ds["tables"], "weight_kind": ds["weight_kind"],
                   "weight_reason": ds["weight_reason"],
                   "weight_per_cycle": df.groupby("CYCLE")["WT_SOURCE"].first().to_dict(),
                   "variance": "Taylor linearization; strata x PSU, with replacement",
                   "design_df": overall["design_df"], "ci_method": overall["ci_method"],
                   "derived": ds["derived"], "files": ds["provenance"]},
        "overall": overall, "panels": panels, "warnings": warnings, "benchmarks": bench,
    }
    pct = statistic == "proportion"
    fmt = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.2f}")
    lines = [f"{data['title']}: {fmt(overall['estimate'])} (95% CI {fmt(overall['ci_low'])}-{fmt(overall['ci_high'])}, "
             f"n={overall['n']}, {overall['reliability']})" + (f"; {age_note}" if age_note else "")]
    for p in panels:
        lines.append(p["name"] + ": " + "; ".join(f"{r['label']} {fmt(r['estimate'])}" for r in p["rows"]))
    for b_ in bench:
        lines.append(f"Benchmark {b_['label']}: server {b_['server']} vs published {b_['published']} ({b_['source_label']})")
    if warnings:
        lines.append("Warnings: " + " | ".join(warnings))
    if EXPLORER:
        lines.append("Shown in the interactive Results Explorer where the client supports MCP Apps.")
    return _mt.CallToolResult(content=[_mt.TextContent(type="text", text="\n".join(lines))],
                              structuredContent=data)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
