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
2. Provision 16GB NVMe Swapfile (prevents Linux OOM killer on 12.7GB System RAM):
   ```bash
   !test -f /swapfile || (fallocate -l 16G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile)
   ```
3. Install Ollama:
   ```bash
   !curl -fsSL https://ollama.com/install.sh | sh
   ```
4. Start Ollama daemon in background while preserving system environment (`PATH`, `LD_LIBRARY_PATH`) and enabling Flash Attention + 8-bit KV cache:
   ```python
   import os, subprocess

   env = os.environ.copy()
   env.update({
       "OLLAMA_HOST": "0.0.0.0:11434",
       "OLLAMA_KEEP_ALIVE": "24h",
       "OLLAMA_FLASH_ATTENTION": "1",
       "OLLAMA_KV_CACHE_TYPE": "q8_0",
   })
   process = subprocess.Popen(
       ["ollama", "serve"],
       env=env,
       stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL,
   )
   ```
5. Pull target models:
   ```bash
   !ollama pull qwen2.5-coder:14b
   ```

### SOP-02: Tunnel Setup & Endpoint Exposure
Prefer **Cloudflare Quick Tunnels** (no account or token required) with non-blocking log file or background pipe draining to prevent 64KB OS pipe buffer deadlocks:
- **Cloudflare Tunnel (Non-Blocking & Pipe-Safe):**
  ```python
  import re, subprocess, threading, time

  !curl -sL --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  !dpkg -i cloudflared.deb > /dev/null 2>&1
  !pkill -f cloudflared || true

  tunnel_proc = subprocess.Popen(
      ["cloudflared", "tunnel", "--url", "http://localhost:11434"],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
  )

  tunnel_url = None
  for _ in range(60):
      line = tunnel_proc.stdout.readline()
      match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
      if match:
          tunnel_url = match.group(0)
          break
      time.sleep(0.2)

  # Drain stdout continuously in background daemon thread to prevent 64KB OS pipe deadlock
  threading.Thread(
      target=lambda: [None for _ in iter(tunnel_proc.stdout.readline, "")],
      daemon=True,
  ).start()
  print("Tunnel URL:", tunnel_url)
  ```

### SOP-03: MCP Server Verification & Tool Testing
1. Configure local MCP client (`mcp_config.json`) or hot-reload via CLI:
   ```bash
   # Hot-reload endpoint across MCP config files & verify status:
   python3 -m colab_ollama_mcp.server --set-endpoint https://xxxx.trycloudflare.com

   # Check current endpoint health & active GPU VRAM models (/api/ps):
   python3 -m colab_ollama_mcp.server --status
   ```
   Standard `mcp_config.json` entry:
   ```json
   {
     "mcpServers": {
       "colab-ollama": {
         "command": "python3",
         "args": ["-m", "colab_ollama_mcp.server", "--endpoint", "<TUNNEL_URL>", "--model", "qwen2.5-coder:14b"]
       }
     }
   }
   ```
2. Verify connectivity via endpoint tags: `GET <TUNNEL_URL>/api/tags` and VRAM residency via `GET <TUNNEL_URL>/api/ps`.
3. Test the 4 exposed MCP tools:
   - `ollama_chat` (`prompt`, optional `system`, `model`, `temperature`)
   - `ollama_list_models` (lists downloaded models via `/api/tags`)
   - `ollama_health_check` (checks tunnel RTT latency, disk models, and live GPU VRAM offload via `/api/ps`)
   - `ollama_set_endpoint` (`endpoint` — hot-swaps the active Cloudflare tunnel URL without restarting the IDE)

---

## 3. Disconnection & Keep-Alive Strategies
- Colab disconnects on idle (~60-90 mins without active DOM interaction).
- Use HTML5 silent audio background playback in Colab cells to keep the browser tab marked as *Media Active*.
- Gracefully handle tunnel resets by invoking `ollama_set_endpoint` or `--set-endpoint <NEW_URL>` for zero-downtime hot-reloading.
