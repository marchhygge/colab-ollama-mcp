---
name: colab-ollama-ops
description: Operational workflows and troubleshooting for running Ollama on Google Colab GPUs and exposing it via MCP tools.
---

# Colab Ollama MCP Operations & Workflow Guide

This skill guides the design, setup, operation, and troubleshooting of `colab-ollama-mcp`, enabling local LLM tools (e.g., Claude Desktop, Antigravity IDE, Cursor) to leverage free/pro Google Colab GPU runtimes without local VRAM.

---

## 1. Core Architecture Overview

```
+-------------------------------------------------------------+
| Local Environment (Zero GPU / Zero RAM)                     |
|                                                             |
| [Antigravity / Claude / Cursor]                             |
|          | (stdio / SSE)                                    |
|          v                                                  |
|   [Python MCP Server (colab-ollama-mcp)]                    |
+--------------------------|----------------------------------+
                           | HTTPS Tunnel (Cloudflare / ngrok)
                           v
+-------------------------------------------------------------+
| Google Colab Runtime (NVIDIA T4 / V100 / A100 GPU)          |
|                                                             |
|   [Tunnel Daemon (cloudflared / ngrok)]                     |
|          | HTTP localhost:11434                             |
|          v                                                  |
|   [Ollama Daemon (OLLAMA_HOST=0.0.0.0:11434)]               |
|          |                                                  |
|   [GPU Models (e.g., llama3.1:8b, qwen2.5:14b, deepseek)]  |
+-------------------------------------------------------------+
```

---

## 2. Standard Operating Procedures (SOPs)

### SOP-01: Colab Setup & Daemon Initialization
1. Verify GPU availability: `!nvidia-smi`.
2. Install Ollama:
   ```bash
   !curl -fsSL https://ollama.com/install.sh | sh
   ```
3. Start Ollama daemon in background with port binding:
   ```python
   import subprocess
   process = subprocess.Popen(["ollama", "serve"], env={"OLLAMA_HOST": "0.0.0.0:11434"})
   ```
4. Pull target models:
   ```bash
   !ollama pull qwen2.5-coder:7b
   ```

### SOP-02: Tunnel Setup & Endpoint Exposure
Prefer **Cloudflare Quick Tunnels** (no account or token required) or **ngrok**:
- **Cloudflare Tunnel:**
  ```bash
  !curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  !dpkg -i cloudflared.deb
  !cloudflared tunnel --url http://localhost:11434
  ```
- Parse the resulting `https://*.trycloudflare.com` URL.

### SOP-03: MCP Server Verification & Tool Testing
1. Configure local MCP client (`mcp_config.json`):
   ```json
   {
     "mcpServers": {
       "colab-ollama": {
         "command": "python3",
         "args": ["-m", "colab_ollama_mcp.server", "--endpoint", "<TUNNEL_URL>"]
       }
     }
   }
   ```
2. Verify connectivity via endpoint tags: `GET <TUNNEL_URL>/api/tags`.
3. Test tool executions: `ollama_list_models`, `ollama_chat`, `ollama_generate`.

---

## 3. Disconnection & Keep-Alive Strategies
- Colab disconnects on idle (~60-90 mins without active DOM interaction).
- Use audio background playback or heartbeat WebSocket ping to maintain active session.
- Gracefully handle tunnel resets by allowing dynamic endpoint updates in the MCP server.
