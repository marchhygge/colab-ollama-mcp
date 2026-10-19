---
trigger: always_on
description: Local Obsidian Knowledge Base routing pointers for colab-ollama-mcp
---

# Knowledge Base Routing Reference

When you need technical specifications, architecture decisions, Colab GPU setup workflows, or MCP schemas for `colab-ollama-mcp`, look up the local Obsidian Knowledge Base:

- **Local Path:** `/Users/marchhygge/MarchHygge/Google Drive Backup/knowledge-base/colab-ollama-mcp`
- **CLI Symlink Path:** `.agents/knowledge-base/` (relative to `/Users/marchhygge/MarchHygge/project/colab-ollama-mcp`)
- **Entry Point (Index):** `00 - Home & Master MOC.md`
  - `01-architecture/`: System architecture (`ARC-01`), tunnel security (`ARC-02`), zero-token fallback (`ARC-03`), Conductor-Worker protocol (`ARC-04`), and interactive HTML explainers (`docs/explainer.html`, `docs/evaluation-report.html`, `report.html`).
  - `02-colab-runtime/`: Colab GPU provisioning (`COLAB-01`), session keep-alive (`COLAB-02`), and Cloudflare Quick Tunnel automation (`COLAB-03`).
  - `03-mcp-server/`: Zero-dependency Python MCP server implementation (`MCP-01`–`MCP-03`), tool definitions (`ollama_chat`, `ollama_list_models`, `ollama_health_check`, `ollama_set_endpoint`), NDJSON streaming, SSRF URL validation, and error handling.

> **Rule:** Read `00 - Home & Master MOC.md` first to locate specific notes on demand rather than scanning the entire vault.
