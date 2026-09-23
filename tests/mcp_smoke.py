"""Smoke test over real MCP stdio transport."""
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main(calls):
    params = StdioServerParameters(command=sys.executable, args=["-m", "nhanes_mcp"], env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            ctx = {}
            for name, args in calls:
                args = {k: (ctx.get(v[1:]) if isinstance(v, str) and v.startswith("$") else v) for k, v in args.items()}
                res = await s.call_tool(name, args)
                txt = res.content[0].text if res.content else ""
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and "dataset_id" in obj: ctx["ds"] = obj["dataset_id"]
                except Exception:
                    obj = txt
                print(f"\n=== {name} {args} (isError={res.isError})\n", txt[:1500])

if __name__ == "__main__":
    asyncio.run(main(json.loads(sys.argv[1])))
