# nhanes-mcp

An MCP server for **design-correct**, conversational access to NHANES public-use data.
Black Swan Causal Labs · MIT license · v0.5

Most "chat with a dataset" layers let an agent compute an unweighted mean. With NHANES that
answer is wrong. This server makes the defensible analysis the default: the agent asks a question
in plain language, and the server finds the files, merges them on `SEQN`, picks the right weight,
keeps the full survey design, and reports design-based estimates with NCHS reliability flags.

> **Not affiliated with, or endorsed by, NCHS or CDC.** Data are the public-use NHANES files
> published by the National Center for Health Statistics and downloaded directly from cdc.gov.
> Users are responsible for following the [NCHS data use agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm).

## What it handles for you

| Pitfall | What the server does |
|---|---|
| Wrong / no weight | `build_dataset` picks the most restrictive weight (interview → MEC → fasting / phlebotomy / surplus-serum subsample → dietary day-1/day-2), explains why, and stores it in `WT_ANALYSIS`. Overrides (`build_dataset(weight=...)`, `set_weight`) are reported with every result; `WT_ANALYSIS` cannot be overwritten silently |
| Subsetting before estimation | `domain=` expressions keep the full design (zero weight outside the domain) |
| Pooling cycles | Weights rescaled by cycle years / total years (2017–March 2020 counts as 3.2 years); 1999–2002 uses 4-year weights; strata made cycle-unique; refuses 2017–2018 + 2017–2020 overlap |
| Long-format tables (e.g. prescriptions) | Refuses to join tables with repeated `SEQN` (which would silently duplicate weights); `flag_from_long_table` collapses them to one row per person |
| Refused / don't-know codes | `describe_variable` reads the CDC codebook and suggests sentinel codes; `set_missing` recodes them |
| Silent 0 for missing | `derive_variable` propagates missingness (`any` / `all` / `none`); `coalesce`, `fillna`, `isna`, `notna`, `where` handle missingness deliberately |
| Irregular file names | Tries known variants (e.g. 1999–2000 surplus-serum files `SSCMV_A`, `SSMUMP_A`) |
| Variance | Taylor linearization, strata × PSU, design df; Korn–Graubard CIs and NCHS 2017 reliability flags for proportions |
| Age adjustment | Direct adjustment to the 2000 US standard (20–39 / 40–59 / 60+) with linearized SE, or to any caller-supplied standard (age groups + population), with a warning for in-domain records outside the groups |
| Mortality | Optional join of the public-use Linked Mortality File (follow-up through 2019) and a design-based Cox model |

## Tools (17)

| Step | Tools |
|---|---|
| Orient | `list_cycles`, `analysis_guidance` |
| Find | `list_files`, `search_variables`, `describe_variable` |
| Build | `build_dataset` (optional mortality join), `describe_dataset` |
| Clean / derive | `set_missing`, `derive_variable`, `flag_from_long_table`, `set_weight` |
| Analyze | `survey_frequency`, `survey_estimate`, `survey_regression` (linear / logistic), `survey_cox` (Cox PH, Binder variance) |
| Present | `show_results` — text + structured results; interactive view with the optional Results Explorer add-on |
| Export | `export_dataset` |

Cycles: 1999–2000 through 2017–2018, 2017–March 2020 (pre-pandemic, `P_` files) and August 2021–August 2023.

## Results Explorer (optional add-on)

`show_results` returns design-based results as text plus structured data in every client. With the
optional **NHANES Results Explorer** add-on installed, MCP Apps hosts (Claude Desktop/web, ChatGPT,
VS Code, Goose) also render an interactive view: headline estimate with CI and NCHS reliability badge,
a crude / age-adjusted toggle that re-runs the estimate on the server, the analysis plan, subgroup
panels, server warnings, benchmarks against published estimates, and design provenance.

The add-on is a separate package from Black Swan Causal Labs under the **PolyForm Noncommercial
License 1.0.0** (free for academic, public-health and other noncommercial use; commercial use needs a
license — https://blackswancausallabs.com). It is not part of this MIT repository. nhanes-mcp finds it if
it is installed in the same Python environment, if `NHANES_MCP_EXPLORER_PATH` points to it, or if a
folder named `nhanes-mcp-explorer` sits next to the `nhanes-mcp` folder.

## Validation

Estimates were checked against published NCHS results (`validation/`):

- **Prevalence:** 57 of 57 published NCHS estimates reproduced (Data Briefs 360, 363, 508, 515;
  pooled 2015–2018; 2017–March 2020 pre-pandemic).
- **Standard errors:** 20 of 20 match; 11 of 12 published 95% CI bounds identical (the 12th differs by 0.1 at a rounding edge).
- **Mortality:** a design-based Cox model on NHANES 1999–2006 (adults 25+) reproduces 6 of 7 published
  hazard ratios within their CIs (NHSR 155). The Mexican American contrast does not reproduce
  (0.71 vs 1.12 published); this is under investigation and the linked file here has longer follow-up (2019 vs 2015).
- **Hypertension** (NCHS Data Brief 511, 2021–2023, adults 18+): prevalence (crude and age-adjusted, by sex and
  age), awareness, treatment and control — 17 of 17 published estimates reproduced exactly.
- **CMV seroprevalence** (Bate et al., *Clin Infect Dis* 2010; NHANES 1999–2004, ages 6–49, surplus-serum
  weights): see `tests/benchmark_nchs.py`. Before v0.4 the server silently used MEC weights here and could not
  load the 1999–2000 file.
- **Unit tests** (`tests/`): variance checked against an independent loop implementation and a
  delete-one-PSU jackknife; Cox model checked against statsmodels PHReg and a jackknife; weight
  selection, pooling, guards, expression semantics, long-table and dietary-weight handling.

## Install

### Easiest: let your AI assistant do it

Paste this into Claude (Cowork or Claude Code), or any agent that can run commands on your computer:

> Install the nhanes-mcp MCP server from https://github.com/Black-Swan-Causal-Labs/nhanes-mcp for Claude
> Desktop. Install `uv` if it is missing, then add this entry to my Claude Desktop config
> (`claude_desktop_config.json`), using the full path to `uvx`:
> `"nhanes": {"command": "uvx", "args": ["--from", "git+https://github.com/Black-Swan-Causal-Labs/nhanes-mcp", "nhanes-mcp"]}`.
> Keep my existing servers. Then tell me to restart Claude Desktop.

### One line in the config (uvx)

With [uv](https://docs.astral.sh/uv/) installed, add to `claude_desktop_config.json` and restart Claude Desktop
(on macOS use the full path from `which uvx`, e.g. `/Users/<you>/.local/bin/uvx`):

```json
"nhanes": {
  "command": "uvx",
  "args": ["--from", "git+https://github.com/Black-Swan-Causal-Labs/nhanes-mcp", "nhanes-mcp"]
}
```

`uvx` fetches the server and its dependencies into an isolated environment on first launch; no clone or
virtual environment to manage.

### From source (for development)

```bash
python3 -m venv ~/.nhanes-mcp-venv
~/.nhanes-mcp-venv/bin/pip install "mcp>=1.2,<2" pandas numpy scipy pyreadstat httpx beautifulsoup4 lxml
git clone https://github.com/Black-Swan-Causal-Labs/nhanes-mcp.git ~/nhanes-mcp
```

```json
"nhanes": {
  "command": "/Users/<you>/.nhanes-mcp-venv/bin/python",
  "args": ["-m", "nhanes_mcp"],
  "env": {"PYTHONPATH": "/Users/<you>/nhanes-mcp"}
}
```

Any MCP client that runs local stdio servers works the same way. Data are downloaded from
cdc.gov on first use and cached in `~/.cache/nhanes-mcp` (override with `NHANES_MCP_CACHE`).
Set `NHANES_MCP_DATA_DIR` to a folder of manually downloaded `.xpt` files to work offline.

Need help setting it up for your team, or adapting it to another survey or dataset?
Contact [Black Swan Causal Labs](https://blackswancausallabs.com).

## Tests

```bash
python tests/test_offline.py            # synthetic NHANES-shaped data, no network
python tests/test_cox.py
python tests/test_long_and_dietary.py
python tests/benchmark_nchs.py          # reproduces published NCHS estimates (needs network)
```

## Known open issues

- Age-adjusted adult obesity for 2009–2010 and earlier runs 0.1–0.7 points below NCHS Health E-Stat 111
  (2011–2012 onward matches exactly). Pooling and pregnancy-code handling were ruled out; cause under investigation.
- The CMV analysis finds 14,198 tested participants aged 6–49 in the public surplus-serum files versus 15,310
  reported by Bate et al.; unexplained.
- NHANES III (1988–1994) is not supported.

## Changelog

- **0.5.0** — `show_results` tool (text + structured results; interactive view via the optional
  Results Explorer add-on); one-command install with `uvx`; analysis guidance rule 11.
- **0.4.1** — Comparisons inside missing-aware functions now return missing when an operand is missing, so
  skip-pattern definitions such as `where(BPQ020 == 1, fillna(BPQ150, 2) == 1, 0)` are missing (not 0) for people
  never asked the screener. Hypertension benchmark (NCHS Data Brief 511) added.
- **0.4.0** — Surplus-serum and other file-specific subsample weights with `2Y`/`4Y` suffixes are detected
  and pooled; 1999–2000 `_A` file names resolved; `build_dataset(weight=...)` and new `set_weight` tool, both
  recorded in every result; design columns protected from `derive_variable`; missing-aware expression functions;
  custom age standards; CMV benchmark added.
- **0.3.0** — Initial public release.

## Limitations

- Public-use files only. Restricted-use data (including the NHANES–CMS Medicare/Medicaid linkage) require an NCHS Research Data Center.
- Variance uses Taylor linearization with PSUs treated as sampled with replacement, as NCHS recommends; replicate weights are not used.
- The server reports what the data support; it does not choose a study design for you.
