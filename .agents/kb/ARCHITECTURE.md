# Project Architecture & System State

> **Repository:** `colab-ollama-mcp`  
> **Host Client:** Antigravity IDE / Claude Desktop / Cursor  
> **Remote Runtime:** Google Colab GPU (Tesla T4 16GB / A100 40GB) via Cloudflare Tunnel

---

## 1. High-Level Topology

```text
[Local Laptop (~30MB RAM)]               [Cloudflare Edge]            [Google Colab GPU VM]
Antigravity IDE <--(JSON-RPC 2.0 stdio)--> server.py <--(HTTPS POST)--> trycloudflare.com <--> Ollama (11434)
```

---

## 2. Notebook Execution Profiles

1. **Option 2 (Performance — Primary Daily Driver):**
   - Notebook: `notebooks/colab_ollama_mcp.ipynb`
   - Hardware: Tesla T4 (16GB VRAM, ~1.53 CU/h)
   - Conductor: `deepseek-r1:14b` (Planning & Architecture CoT)
   - Worker: `qwen2.5-coder:14b` (Coding, Workflow, Diagram)
2. **Option 1 (Saver — Fallback Mode):**
   - Notebook: `notebooks/colab_ollama_t4_saver.ipynb`
   - Hardware: Tesla T4 (16GB VRAM, ~1.53 CU/h)
   - Conductor: `deepseek-r1:8b` (Lightweight CoT)
   - Worker: `qwen2.5-coder:7b` (High-Speed Coding 65 tok/s)
   - Workflow: `llama3.1:8b` (Tool calling)
3. **Combo 2 (A100 Max Power — Milestone Sprints):**
   - Notebook: `notebooks/colab_ollama_a100_max.ipynb`
   - Hardware: NVIDIA A100 (40GB VRAM, ~5.38 CU/h)
   - Suite: `deepseek-r1:32b`, `qwen2.5-coder:32b`, `qwen2.5-coder:14b`, `minicpm-v`

---

## 3. Memory & Safeguard Configurations
- **Linux NVMe Swapfile:** 16GB Swap allocated on Colab NVMe SSD (`/swapfile`), expanding total virtual memory to 28.67 GB.
- **Ollama Runtime Settings:**
  - `OLLAMA_FLASH_ATTENTION=1`
  - `OLLAMA_KV_CACHE_TYPE=q8_0`
  - `OLLAMA_MAX_LOADED_MODELS=1`
  - `OLLAMA_KEEP_ALIVE=24h`
- **Context Limit:** Bounded to 16,384 tokens to maintain peak VRAM usage $\le 12.5$ GB.
