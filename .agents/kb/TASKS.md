# Active Project Tasks & Roadmap

> **Tracking:** Conductor & Worker Task Execution Progress

---

## 🎯 Completed Milestones
- [x] **M1: Core MCP Server (`colab_ollama_mcp/server.py`):** Pure Python stdlib JSON-RPC 2.0 server with hot-reload endpoint capability.
- [x] **M2: Colab Notebooks Suite:**
  - `notebooks/colab_ollama_mcp.ipynb` (Option 2: Performance Multi-Model on T4/A100).
  - `notebooks/colab_ollama_t4_saver.ipynb` (Option 1: Saver Multi-Vendor on T4).
  - `notebooks/colab_ollama_a100_max.ipynb` (Combo 2: Max Power on A100 40GB).
- [x] **M3: Hardware & Memory Safeguards:**
  - Integrated 16GB NVMe swapfile creation in all notebooks.
  - Enabled Flash Attention (`OLLAMA_FLASH_ATTENTION=1`) and 8-bit quantized KV Cache (`OLLAMA_KV_CACHE_TYPE=q8_0`).
- [x] **M4: Conductor & Worker Subagent Protocol:**
  - Designed State Handoff specification (`.agents/kb/HANDOFF_TEMPLATE.md` & `.agents/kb/HANDOFF.md`).
  - Added Knowledge Base documentation (`ARC-04 Conductor-Worker Protocol & State Handoff.md`).
- [x] **M5: MCP Server Hardening & Unit Test Suite (Streams 1 & 2):**
  - Added SSRF endpoint URL validation (`validate_endpoint_url`), 60s `/api/tags` model cache (`_MODEL_CACHE`), internal NDJSON streaming (`stream: True`) to prevent Cloudflare HTTP 524 timeouts, and MCP `"isError": True` compliance.
  - Added pure-stdlib unit test suite (`tests/test_server.py`) and synchronized `configs/*.json` & `configs/.env.example`.
- [x] **M6: Interactive Evaluation Report & KB Synchronization (Stream 4):**
  - Generated interactive HTML evaluation dashboard (`docs/evaluation-report.html`) and updated `report.html`, `README.md`, `SKILL.md`, `MCP-01`, `MCP-02`, and `00 - Home & Master MOC.md`.

---

## ⏸️ On Hold / Upcoming Work
- [ ] **Scope 3 (On Hold — Pending User Model Selection): Colab Notebooks Remediation (`notebooks/*.ipynb`):**
  - Fix 64KB OS pipe buffer deadlock on `tunnel_proc.stdout` across `colab_ollama_mcp.ipynb`, `colab_ollama_t4_saver.ipynb`, and `colab_ollama_a100_max.ipynb`.
  - Add `OLLAMA_FLASH_ATTENTION="1"` and `OLLAMA_KV_CACHE_TYPE="q8_0"` to `colab_ollama_a100_max.ipynb` and unify HTML5 audio keep-alive cells once target model roster is finalized.
- [ ] **T1: Live Operational Handoff Benchmark:** Benchmark end-to-end Conductor $\rightarrow$ Worker latency on live Colab T4/A100 tunnel.
