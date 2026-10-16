# Auto-Connect & Switch to Colab GPU Model (`*.trycloudflare.com`)

Whenever the user pastes a Cloudflare Quick Tunnel URL (`https://*.trycloudflare.com`) into the conversation (even if they only paste the raw link without additional instructions):

1. **Auto-Save & Health Check:**
   - Immediately update the endpoint and verify connectivity by running:
     ```bash
     python3 /Users/marchhygge/MarchHygge/project/colab-ollama-mcp/colab_ollama_mcp/server.py --set-endpoint "<URL>"
     ```
     (or calling the MCP tool `ollama_set_endpoint` with `endpoint: "<URL>"`).
2. **Confirm Connection:**
   - Report the connection status (`ONLINE`), round-trip latency (`ms`), and active model on the Colab GPU (e.g., `qwen2.5-coder:14b`).
3. **Auto-Switch to Colab GPU Model (`ollama_chat`):**
   - For all subsequent coding, implementation, refactoring, or debugging requests in the session, automatically delegate the code generation/analysis to the remote Colab GPU by calling `ollama_chat` (via MCP or `server.py`) so the user's tasks run on `qwen2.5-coder:14b` on Google Colab!
