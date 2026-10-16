---
trigger: always_on
description: Local Obsidian Knowledge Base routing pointers for colab-ollama-mcp
---

# Knowledge Base Routing Reference

When you need technical specifications, architecture decisions, Colab GPU setup workflows, or MCP schemas for `colab-ollama-mcp`, look up the local Obsidian Knowledge Base:

- **Local Path:** `/Users/marchhygge/MarchHygge/Google Drive Backup/knowledge-base/colab-ollama-mcp`
- **CLI Symlink Path:** `.agents/knowledge-base/` (relative to `/Users/marchhygge/MarchHygge/project/colab-ollama-mcp`)
- **Entry Point (Index):** `00 - Home & Master MOC.md` (`00 - KB Index.base`, `00 - KB Canvas.canvas`)
  - `01-architecture/`: System architecture, sequence diagrams, MCP protocol compliance, and security/tunneling design.
  - `02-colab-runtime/`: Colab GPU provisioning, Ollama installation & daemon management, tunnel providers (Cloudflared, ngrok), and session keep-alive.
  - `03-mcp-server/`: FastMCP / Python MCP server implementation, tool definitions (`generate`, `chat`, `pull`, `list`), streaming, and error handling.

> **Rule:** Read `00 - Home & Master MOC.md` first to locate specific notes on demand rather than scanning the entire vault.
