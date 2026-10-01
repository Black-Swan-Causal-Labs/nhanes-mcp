"""Offline tests for the MCP App (Results Explorer): tool metadata, UI resource, structured output."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_offline  # noqa: E402,F401  (builds synthetic NHANES-shaped files)
from nhanes_mcp import server as S  # noqa: E402


def test_tool_and_resource_registered():
    tools = asyncio.run(S.mcp.list_tools())
    t = next(t for t in tools if t.name == "show_results")
    assert t.meta["ui"]["resourceUri"] == S.VIEW_URI
    res = asyncio.run(S.mcp.list_resources())
    r = next(r for r in res if str(r.uri) == S.VIEW_URI)
    assert r.mimeType == "text/html;profile=mcp-app"
    html = S._view_html()
    assert "__FEATHER_B64__" not in html and "/*__MCP_APPS_SDK__*/" not in html
    assert "globalThis.McpApps" in html and "data:image/png;base64," in html
    assert html.count("</script>") == 2  # SDK script + app script; none leaked from the bundle


def test_show_results_structured_output():
    b = S.build_dataset(["2017-2018"], ["BMX"])
    d = b["dataset_id"]
    S.derive_variable(d, "OBESE", "BMXBMI >= 30")
    S.derive_variable(d, "AGEGRP", "(RIDAGEYR >= 40) + (RIDAGEYR >= 60) + 1")
    out = S.show_results(d, "OBESE", "proportion", "RIDAGEYR >= 20 & RIDEXPRG != 1", ["RIAGENDR", "AGEGRP"],
                         "nchs_adults_20plus", title="Adult obesity", question="How common is obesity?",
                         outcome_label="Obesity (BMI >= 30)", population_label="Adults 20+, not pregnant",
                         group_labels={"AGEGRP": {"1": "20-39", "2": "40-59", "3": "60+"}},
                         benchmarks=[{"label": "overall", "published": 30.0, "source_label": "x"},
                                     {"label": "men", "published": 29.0, "by": "RIAGENDR", "level": 1}])
    sc = out.structuredContent
    json.dumps(sc)  # must be JSON-serializable
    assert sc["kind"] == "nhanes-results"
    ref = S.survey_estimate(d, "OBESE", "proportion", "RIDAGEYR >= 20 & RIDEXPRG != 1",
                            age_adjust="nchs_adults_20plus")["results"][0]["estimate"]
    assert abs(sc["overall"]["estimate"] - ref) < 1e-12
    sex, age = sc["panels"]
    assert [r["label"] for r in sex["rows"]] == ["Men", "Women"] and sex["age_adjusted"]
    assert [r["label"] for r in age["rows"]] == ["20-39", "40-59", "60+"] and not age["age_adjusted"] and age["age_specific"] and not sex["age_specific"]
    assert sc["benchmarks"][0]["server"] == round(100 * ref, 1)
    assert sc["benchmarks"][1]["server"] == round(100 * sex["rows"][0]["estimate"], 1)
    assert "Adult obesity:" in out.content[0].text
    Path(__file__).with_name("_preview.json").write_text(json.dumps(sc))


if __name__ == "__main__":
    test_tool_and_resource_registered(); print("PASS test_tool_and_resource_registered")
    test_show_results_structured_output(); print("PASS test_show_results_structured_output")
