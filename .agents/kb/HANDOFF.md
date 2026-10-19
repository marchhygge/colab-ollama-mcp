# [STATE HANDOFF] Parallel Remediation Scopes (Excluding Scope 3 Notebooks)

## Stream 1 — Core MCP Server Hardening (`colab_ollama_mcp/server.py`, `pyproject.toml`)
- **Target Files**: `colab_ollama_mcp/server.py`, `pyproject.toml` (DO NOT touch `configs/`, `tests/`, `notebooks/`, or `docs/`).
- **Tasks**:
  1. Add `validate_endpoint_url(url: str) -> tuple[bool, str]` using `urllib.parse.urlparse`. Allow `https://` with valid public hostnames or explicit `http://localhost[:port]` / `http://127.0.0.1[:port]`. Block `file://`, `ftp://`, `javascript:`, and private LAN/metadata IPs (`169.254.*`, `10.*`, `192.168.*`). Enforce in `set_endpoint_in_configs()` and `call_ollama_api()`.
  2. Add 60-second TTL cache `_MODEL_CACHE: dict[str, tuple[float, list]] = {}` in `fetch_available_models(endpoint: str, use_cache: bool = True)` so `resolve_model()` avoids redundant `GET /api/tags` calls on every `ollama_chat`. Clear `_MODEL_CACHE` when `set_endpoint_in_configs()` runs.
  3. In `call_ollama_api()`, use `"stream": True` in the `/api/chat` payload and iterate over NDJSON response lines (with single-JSON fallback) to prevent Cloudflare HTTP 524 timeouts on long generations.
  4. Guard `DEFAULT_TIMEOUT` env parsing and `float(args.get("temperature", 0.2))`. Wrap `handle_request()` tool dispatch in `try/except Exception` returning JSON-RPC `-32603` on unhandled errors, and set `"isError": True` in MCP tool results whenever an `❌` error string is returned.
  5. In `set_endpoint_in_configs()`, also update `configs/claude_desktop_config.json` and `configs/cursor_mcp.json` if present while preserving existing `env` keys.

## Stream 2 — Automated Unit Test Suite & Client Configs (`tests/test_server.py`, `configs/`)
- **Target Files**: `tests/test_server.py`, `configs/.env.example`, `configs/antigravity_mcp_config.json`, `configs/claude_desktop_config.json`, `configs/cursor_mcp.json` (DO NOT modify `colab_ollama_mcp/server.py` or `notebooks/`).
- **Tasks**:
  1. Update `configs/.env.example` to document `OLLAMA_ENDPOINT`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT`.
  2. Create `tests/test_server.py` using Python stdlib `unittest` + `unittest.mock` testing `get_tool_definitions`, `resolve_model`, `fetch_available_models`, `call_ollama_api`, `format_server_status`, and `handle_request` (`initialize`, `ping`, `tools/list`, `tools/call`). Run `python3 -m unittest discover -s tests -v`.

## Stream 4 — Skills, Obsidian KB & Interactive Evaluation Report (`docs/evaluation-report.html`, `.agents/`, `README.md`, `report.html`)
- **Target Files**: `docs/evaluation-report.html`, `.agents/skills/colab-ollama-ops/SKILL.md`, `README.md`, `report.html`, `.agents/kb/TASKS.md`, `.agents/rules/kb-reference.md`, `.agents/knowledge-base/00 - Home & Master MOC.md`, `.agents/knowledge-base/03-mcp-server/MCP-01 FastMCP Server Design.md`, `.agents/knowledge-base/03-mcp-server/MCP-02 Client Configuration & Tool Schemas.md` (DO NOT touch `notebooks/*.ipynb` or `colab_ollama_mcp/server.py`).
- **Tasks**:
  1. Generate self-contained interactive browser artifact `docs/evaluation-report.html` (with tabs for Scope 1 MCP Server, Scope 2 Tests & Configs, Scope 3 Notebooks [On Hold - Pending Model Selection], Scope 4 KB & Docs, interactive P0-P3 filter, scorecard, and architecture diagram) and open it via `open docs/evaluation-report.html`.
  2. Fix `.agents/skills/colab-ollama-ops/SKILL.md` (`os.environ.copy()` + Flash Attention/KV cache, non-blocking cloudflared with pipe drain, 4 actual MCP tools & `--set-endpoint` / `--status`).
  3. Sync `README.md` (`ollama_set_endpoint` in sequence diagram, `report.html` & `docs/evaluation-report.html` in repo tree, fix `CONDUCUTOR` typo), `MCP-01`, `MCP-02`, `00 - Home & Master MOC.md`, `.agents/rules/kb-reference.md`, and `.agents/kb/TASKS.md`.
