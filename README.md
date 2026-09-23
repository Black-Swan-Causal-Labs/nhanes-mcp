# nhanes-mcp

An MCP server for **design-correct**, conversational access to NHANES public-use data.
Black Swan Causal Labs · MIT license · v0.3

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
| Wrong / no weight | `build_dataset` picks the most restrictive weight (interview → MEC → fasting/phlebotomy subsample → dietary day-1/day-2), explains why, and stores it in `WT_ANALYSIS` |
| Subsetting before estimation | `domain=` expressions keep the full design (zero weight outside the domain) |
| Pooling cycles | Weights rescaled by cycle years / total years (2017–March 2020 counts as 3.2 years); 1999–2002 uses 4-year weights; strata made cycle-unique; refuses 2017–2018 + 2017–2020 overlap |
| Long-format tables (e.g. prescriptions) | Refuses to join tables with repeated `SEQN` (which would silently duplicate weights); `flag_from_long_table` collapses them to one row per person |
| Refused / don't-know codes | `describe_variable` reads the CDC codebook and suggests sentinel codes; `set_missing` recodes them |
| Silent 0 for missing | `derive_variable` propagates missingness (`any` / `all` / `none`) |
| Variance | Taylor linearization, strata × PSU, design df; Korn–Graubard CIs and NCHS 2017 reliability flags for proportions |
| Age adjustment | Direct adjustment to the 2000 US standard (20–39 / 40–59 / 60+) with linearized SE |
| Mortality | Optional join of the public-use Linked Mortality File (follow-up through 2019) and a design-based Cox model |

## Tools (15)

| Step | Tools |
|---|---|
| Orient | `list_cycles`, `analysis_guidance` |
| Find | `list_files`, `search_variables`, `describe_variable` |
| Build | `build_dataset` (optional mortality join), `describe_dataset` |
| Clean / derive | `set_missing`, `derive_variable`, `flag_from_long_table` |
| Analyze | `survey_frequency`, `survey_estimate`, `survey_regression` (linear / logistic), `survey_cox` (Cox PH, Binder variance) |
| Export | `export_dataset` |

Cycles: 1999–2000 through 2017–2018, 2017–March 2020 (pre-pandemic, `P_` files) and August 2021–August 2023.

## Validation

Estimates were checked against published NCHS results (`validation/`):

- **Prevalence:** 57 of 57 published NCHS estimates reproduced (Data Briefs 360, 363, 508, 515;
  pooled 2015–2018; 2017–March 2020 pre-pandemic).
- **Standard errors:** 20 of 20 match; 11 of 12 published 95% CI bounds identical (the 12th differs by 0.1 at a rounding edge).
- **Mortality:** a design-based Cox model on NHANES 1999–2006 (adults 25+) reproduces 6 of 7 published
  hazard ratios within their CIs (NHSR 155). The Mexican American contrast does not reproduce
  (0.71 vs 1.12 published); this is under investigation and the linked file here has longer follow-up (2019 vs 2015).
- **Unit tests** (`tests/`): variance checked against an independent loop implementation and a
  delete-one-PSU jackknife; Cox model checked against statsmodels PHReg and a jackknife; weight
  selection, pooling, guards, expression semantics, long-table and dietary-weight handling.

## Install (Claude Desktop, macOS)

```bash
python3 -m venv ~/.nhanes-mcp-venv
~/.nhanes-mcp-venv/bin/pip install "mcp>=1.2,<2" pandas numpy scipy pyreadstat httpx beautifulsoup4 lxml
git clone https://github.com/Black-Swan-Causal-Labs/nhanes-mcp.git ~/nhanes-mcp
```

Then add to `~/Library/Application Support/Claude/claude_desktop_config.json` (use absolute paths)
and restart Claude Desktop:

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

## Limitations

- Public-use files only. Restricted-use data (including the NHANES–CMS Medicare/Medicaid linkage) require an NCHS Research Data Center.
- Variance uses Taylor linearization with PSUs treated as sampled with replacement, as NCHS recommends; replicate weights are not used.
- The server reports what the data support; it does not choose a study design for you.
