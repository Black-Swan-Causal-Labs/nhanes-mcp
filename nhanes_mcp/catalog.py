"""NHANES cycle catalog, file download/caching, codebook + variable-list parsing."""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import httpx
import pandas as pd

CYCLES = {
    "1999-2000": {"prefix": "", "suffix": "", "start": 1999, "years": 2.0, "mort": "1999_2000"},
    "2001-2002": {"prefix": "", "suffix": "_B", "start": 2001, "years": 2.0, "mort": "2001_2002"},
    "2003-2004": {"prefix": "", "suffix": "_C", "start": 2003, "years": 2.0, "mort": "2003_2004"},
    "2005-2006": {"prefix": "", "suffix": "_D", "start": 2005, "years": 2.0, "mort": "2005_2006"},
    "2007-2008": {"prefix": "", "suffix": "_E", "start": 2007, "years": 2.0, "mort": "2007_2008"},
    "2009-2010": {"prefix": "", "suffix": "_F", "start": 2009, "years": 2.0, "mort": "2009_2010"},
    "2011-2012": {"prefix": "", "suffix": "_G", "start": 2011, "years": 2.0, "mort": "2011_2012"},
    "2013-2014": {"prefix": "", "suffix": "_H", "start": 2013, "years": 2.0, "mort": "2013_2014"},
    "2015-2016": {"prefix": "", "suffix": "_I", "start": 2015, "years": 2.0, "mort": "2015_2016"},
    "2017-2018": {"prefix": "", "suffix": "_J", "start": 2017, "years": 2.0, "mort": "2017_2018"},
    "2017-2020": {"prefix": "P_", "suffix": "", "start": 2017, "years": 3.2, "mort": None,
                  "label": "2017-March 2020 pre-pandemic (combined 2017-2018 + partial 2019-2020)"},
    "2021-2023": {"prefix": "", "suffix": "_L", "start": 2021, "years": 2.0, "mort": None,
                  "label": "August 2021-August 2023 (post-pandemic redesign)"},
}
COMPONENTS = ["Demographics", "Dietary", "Examination", "Laboratory", "Questionnaire"]
BASE = "https://wwwn.cdc.gov"
MORT_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality"

CACHE = Path(os.environ.get("NHANES_MCP_CACHE", Path.home() / ".cache" / "nhanes-mcp"))
LOCAL_DATA = os.environ.get("NHANES_MCP_DATA_DIR")  # optional folder of manually downloaded .xpt files
CACHE.mkdir(parents=True, exist_ok=True)

_client = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=120, follow_redirects=True,
                               headers={"User-Agent": "nhanes-mcp/0.1 (research; Black Swan Causal Labs)"})
    return _client


def check_cycle(cycle: str) -> dict:
    if cycle not in CYCLES:
        raise ValueError(f"Unknown cycle '{cycle}'. Valid: {list(CYCLES)}")
    return CYCLES[cycle]


def file_name(base: str, cycle: str) -> str:
    """Map a base table name (e.g. 'BMX') to the cycle-specific file (BMX_J, P_BMX, BMX_L).
    If `base` already looks cycle-specific it is returned upper-cased unchanged."""
    c = check_cycle(cycle)
    b = base.upper()
    if b.startswith("P_") or re.search(r"_[B-L]$", b):
        return b
    return f"{c['prefix']}{b}{c['suffix']}"


def base_name(fname: str) -> str:
    b = fname.upper()
    if b.startswith("P_"):
        b = b[2:]
    return re.sub(r"_[B-L]$", "", b)


def _urls(fname: str, cycle: str, ext: str) -> list[str]:
    c = check_cycle(cycle)
    old_folder = "2017-2018" if cycle == "2017-2020" else cycle
    return [
        f"{BASE}/Nchs/Data/Nhanes/Public/{c['start']}/DataFiles/{fname}.{ext.lower()}",
        f"{BASE}/Nchs/Nhanes/{old_folder}/{fname}.{ext.upper()}",
        f"{BASE}/Nchs/Nhanes/{old_folder}/{fname}.{ext.lower()}",
    ]


def _get(urls: list[str]) -> tuple[bytes, str]:
    errs = []
    for u in urls:
        try:
            r = client().get(u)
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and len(r.content) > 0 and not (
                    u.lower().endswith(".xpt") and "html" in ctype):
                return r.content, u
            errs.append(f"{u} -> HTTP {r.status_code} ({ctype})")
        except httpx.HTTPError as e:
            errs.append(f"{u} -> {type(e).__name__}: {e}")
    raise RuntimeError("Download failed:\n" + "\n".join(errs))


def load_xpt(base: str, cycle: str) -> tuple[pd.DataFrame, dict]:
    """Return (data, meta). Uses local data dir, then cache, then CDC."""
    fname = file_name(base, cycle)
    meta = {"file": fname, "cycle": cycle}
    candidates = []
    if LOCAL_DATA:
        candidates += [Path(LOCAL_DATA) / f"{fname}.xpt", Path(LOCAL_DATA) / f"{fname}.XPT"]
    cpath = CACHE / cycle / f"{fname}.xpt"
    candidates.append(cpath)
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        content, url = _get(_urls(fname, cycle, "xpt"))
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(content)
        path, meta["source_url"] = cpath, url
    meta["path"] = str(path)
    labels = {}
    try:
        import pyreadstat
        df, m = pyreadstat.read_xport(str(path))
        labels = dict(zip(m.column_names, m.column_labels))
    except Exception:
        df = pd.read_sas(str(path), format="xport")
    df.columns = [c.upper() for c in df.columns]
    # SAS transport stores exact zeros as ~5.4e-79 when read by some readers; restore true zeros
    # (otherwise e.g. WTMEC2YR 'not examined' looks like a positive weight).
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    if "SEQN" in df.columns:
        df["SEQN"] = df["SEQN"].astype("int64")
    meta["labels"] = {k.upper(): v for k, v in labels.items()}
    meta["n_rows"] = len(df)
    return df, meta


# ---------- listings & codebooks ----------

def _read_html_tables(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(io.StringIO(html))
    except ValueError:
        return []


def list_files(cycle: str, component: str) -> list[dict]:
    check_cycle(cycle)
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    cache = CACHE / "listings" / f"files_{cycle}_{component}.json"
    if cache.exists():
        return pd.read_json(cache).to_dict("records")
    url = f"{BASE}/nchs/nhanes/search/datapage.aspx?Component={component}&Cycle={cycle}"
    html, _ = _get([url])
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        links = [a.get("href", "") for a in tr.find_all("a")]
        xpt = next((l for l in links if l.lower().endswith(".xpt")), None)
        if not xpt:
            continue
        fname = Path(xpt).stem.upper()
        rows.append({"file": fname, "base": base_name(fname),
                     "description": tds[0].get_text(" ", strip=True),
                     "published": tds[-1].get_text(" ", strip=True)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(cache)
    return rows


def variable_list(cycle: str, component: str) -> pd.DataFrame:
    check_cycle(cycle)
    cache = CACHE / "listings" / f"vars_{cycle}_{component}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    url = f"{BASE}/nchs/nhanes/search/variablelist.aspx?Component={component}&Cycle={cycle}"
    html, _ = _get([url])
    tables = [t for t in _read_html_tables(html.decode("utf-8", "ignore") if isinstance(html, bytes) else html)
              if any("Variable Name" in str(c) for c in t.columns)]
    if not tables:
        raise RuntimeError(f"No variable table parsed from {url}")
    t = tables[0]
    t.columns = [str(c).strip() for c in t.columns]
    rename = {"Variable Name": "variable", "Variable Description": "description",
              "Data File Name": "file", "Data File Description": "file_description"}
    t = t.rename(columns=rename)
    keep = [c for c in ["variable", "description", "file", "file_description", "Use Constraints"] if c in t.columns]
    t = t[keep].copy()
    t["component"] = component
    t["cycle"] = cycle
    cache.parent.mkdir(parents=True, exist_ok=True)
    t.to_pickle(cache)
    return t


def codebook(base: str, cycle: str) -> dict:
    """Parse the CDC .htm documentation for a file: per-variable label, target, value table."""
    fname = file_name(base, cycle)
    cpath = CACHE / cycle / f"{fname}.htm"
    if cpath.exists():
        html = cpath.read_bytes()
    else:
        html, _ = _get(_urls(fname, cycle, "htm"))
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(html)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out = {}
    for dl in soup.find_all("dl"):
        info = {}
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd is not None:
                info[dt.get_text(" ", strip=True).rstrip(":").strip()] = dd.get_text(" ", strip=True)
        name = info.get("Variable Name")
        if not name:
            continue
        values = []
        tbl = dl.find_next("table")
        if tbl is not None and (tbl.find_previous("dl") is dl):
            for tr in tbl.find_all("tr")[1:]:
                cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if len(cells) >= 3:
                    values.append({"code": cells[0], "description": cells[1], "count": cells[2]})
        out[name.upper()] = {"label": info.get("SAS Label"), "question": info.get("English Text"),
                             "target": info.get("Target"), "values": values}
    return out


def mortality(cycle: str) -> pd.DataFrame:
    """NCHS public-use Linked Mortality File (follow-up through 31 Dec 2019)."""
    c = check_cycle(cycle)
    if not c["mort"]:
        raise ValueError(f"No public-use linked mortality file for {cycle} (available 1999-2000 .. 2017-2018).")
    fname = f"NHANES_{c['mort']}_MORT_2019_PUBLIC.dat"
    cpath = CACHE / "mortality" / fname
    if not cpath.exists():
        content, _ = _get([f"{MORT_BASE}/{fname}"])
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(content)
    colspecs = [(0, 6), (14, 15), (15, 16), (16, 19), (19, 20), (20, 21), (42, 45), (45, 48)]
    names = ["SEQN", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING", "DIABETES", "HYPERTEN", "PERMTH_INT", "PERMTH_EXM"]
    df = pd.read_fwf(cpath, colspecs=colspecs, names=names, na_values=[".", ""])
    df["SEQN"] = df["SEQN"].astype("int64")
    return df
