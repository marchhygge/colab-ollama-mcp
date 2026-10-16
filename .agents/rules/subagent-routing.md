---
trigger: always_on
description: Project-specific subagent & model mapping for colab-ollama-mcp (references Global GEMINI.md S0-S8)
---

# Subagent & Model Mapping — `colab-ollama-mcp`

> **Inherits from Global Rule:** [`~/.gemini/config/GEMINI.md`](file:///Users/marchhygge/.gemini/config/GEMINI.md) (`S0`–`S8`).
> Do not duplicate global rules here; use this file strictly to map `colab-ollama-mcp` modules and workflows to Global Scope IDs.

## 1. Domain-to-Scope Mapping (`S0`–`S8`)

| Global Scope | `TypeName` | `Model` | Concrete Mapping in `colab-ollama-mcp` |
| :--- | :---: | :---: | :--- |
| **`S0` Direct** | *(Main)* | *(Main)* | Editing `<= 3` known files (e.g., `server.py`, `pyproject.toml`, `README.md`, `mcp_config.json`), single CLI commands, or direct status checks. |
| **`S1` Quick Lookup** | `research` | `flash_lite` | Counting files, checking endpoint responsiveness (`/api/tags`), listing supported models, checking local/remote tunnel status. |
| **`S2` Research / KB** | `research` | `flash` | Reading multiple Obsidian KB notes in `.agents/knowledge-base/` (architecture, Colab keep-alive strategies, Cloudflare/ngrok tunneling, MCP specs). **Return only synthesis + exact `file:line` pointers.** |
| **`S3` Routine Review** | `research` | `flash` | Routine linter checks (Ruff, Flake8), type checks (Mypy), formatting, git branch promotion, and MCP tool schema validation. |
| **`S4` Deep Review** | `research` | `pro` | Auditing tunnel security (token leaks), Colab GPU VRAM limits, SSE/stdio protocol resilience, streaming backpressure, and connection failure handling. |
| **`S5` Code (Parallel)** | `self` | `inherit` | Implementing independent modules in parallel: e.g., Subagent A (Colab bootstrap notebook & Ollama daemon script), Subagent B (Python MCP server with FastMCP / MCP SDK), Subagent C (tunnel client & keep-alive monitor). Batch in a single `invoke_subagent` call. |
| **`S6` Standard Debug** | `self` | `inherit` | Fixing localized Python exceptions in the MCP server, HTTP request parsing errors, or payload serialization bugs. |
| **`S7` Complex RCA** | `research` / `self` | `pro` | Multi-hop root-cause analysis: Client (Claude / Antigravity) -> MCP Server -> Tunnel (Cloudflare / ngrok) -> Colab GPU Ollama daemon (SSE disconnects, token stream timeouts, Colab kernel death). |
| **`S8` Infra / Validation** | `self` | `flash` | Testing tunnel health, dry-running MCP tools against mock or live Ollama APIs, validating JSON-RPC schemas, and measuring token generation latency (tps). |

## 2. Bottom-Up Promotion Rule (Project -> Global)
- If a task in `colab-ollama-mcp` introduces a new scope not covered by `S0`–`S8` (such as specialized GPU quantization pipelines or multi-node Colab cluster coordination), propose adding the specific mapping here **and** ask the user whether to promote its generalized pattern (`S9+`) into [`~/.gemini/config/GEMINI.md`](file:///Users/marchhygge/.gemini/config/GEMINI.md).
