`mcp-apps-sdk.js` is `@modelcontextprotocol/ext-apps` v2.0.3 (`dist/src/app-with-deps.js`, MIT license,
https://github.com/modelcontextprotocol/ext-apps), with its final ES `export{}` statement replaced by
`globalThis.McpApps = { App }` and wrapped in an IIFE so it can be inlined as a classic script. The view is
therefore fully self-contained: no CDN, no network access, nothing to declare in the CSP.
