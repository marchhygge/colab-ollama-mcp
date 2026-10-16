#!/usr/bin/env python3
"""
Colab Ollama MCP Server (Zero-Dependency Pure Standard Library)
===============================================================
A lightweight Model Context Protocol (MCP) server that bridges local AI IDE clients
(such as Antigravity, Claude Desktop, and Cursor) to an Ollama LLM daemon running
on a remote Google Colab GPU runtime via a Cloudflare Quick Tunnel.

Architecture & Data Flow:
  [IDE Client] <--(stdio JSON-RPC 2.0)--> [server.py] <--(HTTPS POST)--> [Cloudflare Tunnel] --> [Colab GPU Ollama]

Key Design Principles:
  1. Zero external dependencies: Uses only Python built-in modules (`sys`, `json`, `os`, `urllib`, `time`).
  2. Dynamic configuration: Parses `--endpoint` and `--model` from `sys.argv` or environment variables
     so users never need to modify source code when Google Colab assigns a new ephemeral tunnel URL.
  3. Non-blocking idle wait: Reads `sys.stdin` line-by-line using OS kernel blocking I/O (0% CPU when idle).
  4. Immediate pipe delivery: Explicitly calls `sys.stdout.flush()` after every response to bypass
     OS output buffering and prevent IDE client deadlocks.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# MCP Protocol specification version and server metadata
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "colab-ollama-mcp"
SERVER_VERSION = "1.0.0"

# Default coding model recommended for Google Colab Pro (Tesla T4 16GB / A100 40GB VRAM)
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "180"))


def get_endpoint() -> str:
    """
    Resolve the active Cloudflare Tunnel URL dynamically (supports hot-reload without restarting IDE).

    Why this exists:
      Google Colab generates a new random `*.trycloudflare.com` URL every time a new VM starts.
      By checking the local MCP config files on disk first and falling back to `sys.argv`, you can
      run `--set-endpoint <NEW_URL>` at any time and the already-running MCP subprocess immediately
      switches to the new tunnel URL without needing to reload or restart Antigravity IDE!
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    config_candidates = [
        os.path.expanduser("~/.gemini/config/mcp_config.json"),
        os.path.join(repo_root, "configs", "antigravity_mcp_config.json"),
    ]
    for cfg_path in config_candidates:
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                args = cfg.get("mcpServers", {}).get("colab-ollama", {}).get("args", [])
                for i, arg in enumerate(args):
                    if arg == "--endpoint" and i + 1 < len(args):
                        val = args[i + 1].strip().rstrip("/")
                        if val and "YOUR-TUNNEL-URL" not in val:
                            return val
        except Exception:
            pass

    for i, arg in enumerate(sys.argv):
        if arg == "--endpoint" and i + 1 < len(sys.argv):
            return sys.argv[i + 1].rstrip("/")
    return os.environ.get("OLLAMA_ENDPOINT", "https://YOUR-TUNNEL-URL.trycloudflare.com").rstrip("/")



def get_default_model() -> str:
    """
    Scan command-line arguments (`sys.argv`) for an optional `--model` override.
    Falls back to `DEFAULT_MODEL` (`qwen2.5-coder:14b`).
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return DEFAULT_MODEL


def fetch_available_models(endpoint: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """
    Query `GET /api/tags` on the remote Colab Ollama daemon to list all downloaded models in VRAM/Disk.
    """
    url = f"{endpoint.rstrip('/')}/api/tags"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": f"{SERVER_NAME}/{SERVER_VERSION}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
        return data.get("models", [])


def fetch_running_models(endpoint: str, timeout: int = 10) -> List[Dict[str, Any]]:
    """
    Query `GET /api/ps` on the remote Colab Ollama daemon to inspect models currently loaded
    in GPU VRAM, including exact VRAM consumption (`size_vram`), total weight size (`size`),
    and processor offload ratio (GPU vs. CPU RAM).
    """
    url = f"{endpoint.rstrip('/')}/api/ps"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": f"{SERVER_NAME}/{SERVER_VERSION}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
        return data.get("models", [])



def resolve_model(endpoint: str, requested_model: Optional[str]) -> str:
    """
    Determine which model to run on Colab.
    If the requested or default model (e.g., `qwen2.5-coder:14b`) is not yet pulled on Colab,
    automatically fall back to the first available model on the remote server (e.g., `qwen2.5-coder:7b`).
    """
    target = requested_model or get_default_model()
    try:
        models = fetch_available_models(endpoint, timeout=10)
        installed_names = [m.get("name", "") for m in models if m.get("name")]
        if not installed_names:
            return target
        if target in installed_names:
            return target
        # Check prefix match (e.g., "qwen2.5-coder:14b" vs "qwen2.5-coder:14b-instruct")
        for name in installed_names:
            if name.startswith(target.split(":")[0]):
                return name
        return installed_names[0]
    except Exception:
        # If tag lookup fails, proceed with target and let the main call report any connection error
        return target


def call_ollama_api(
    endpoint: str,
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[str, float, str]:
    """
    Package the user prompt into an HTTP POST request and send it through the Cloudflare Tunnel
    to the remote Ollama daemon (`/api/chat`) running on Google Colab GPU.

    Why this exists:
      The IDE speaks MCP (JSON-RPC 2.0 over stdio pipes), whereas Ollama only understands HTTP REST calls
      at `localhost:11434`. This function acts as the protocol translator across the Reverse Tunnel.

    Technical mechanics:
      - Resolves the active model name on Colab.
      - Uses `"stream": False` so the complete generated code block is returned in a single JSON payload.
      - Enforces an explicit socket `timeout` (default 180s) so the local subprocess never hangs indefinitely
        if the Colab session disconnects or GPU VRAM is exhausted.
    """
    active_model = resolve_model(endpoint, model)
    url = f"{endpoint.rstrip('/')}/api/chat"

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": active_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"{SERVER_NAME}/{SERVER_VERSION}",
        },
        method="POST",
    )

    try:
        start_time = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start_time
            content = res_body.get("message", {}).get("content", "")
            return content, elapsed, active_model
    except urllib.error.HTTPError as e:
        err_Detail = e.read().decode("utf-8", errors="replace")
        return f"HTTP Error {e.code} from Colab Ollama ({active_model}): {err_Detail}", 0.0, active_model
    except urllib.error.URLError as e:
        return (
            f"Connection Error: Unable to reach Colab tunnel at '{endpoint}'. "
            f"Verify that Google Colab is running and the Cloudflare URL is up-to-date. Details: {e.reason}",
            0.0,
            active_model,
        )
    except Exception as e:
        return f"Unexpected error calling Colab Ollama: {str(e)}", 0.0, active_model


def send_response(res: Dict[str, Any]) -> None:
    """
    Serialize a JSON-RPC 2.0 response object to a single newline-delimited string on `sys.stdout`
    and immediately flush the OS I/O buffer.

    Why this exists:
      Operating systems buffer `stdout` data in RAM (typically 4KB-8KB blocks) to minimize system calls.
      Without `sys.stdout.flush()`, small JSON responses remain stuck in the buffer and the IDE client
      waits forever (deadlock).
    """
    sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Return the MCP tool schemas exposed to the IDE client during the `tools/list` discovery phase.
    """
    return [
        {
            "name": "ollama_chat",
            "description": (
                "Send a coding prompt or architectural question to the Ollama LLM running on "
                "Google Colab GPU (default: qwen2.5-coder:14b). Use this to generate, refactor, "
                "review, or debug code without consuming local RAM or cloud API tokens."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The programming task, question, or code snippet to process.",
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system instruction (e.g., 'You are a senior Python architect').",
                    },
                    "model": {
                        "type": "string",
                        "description": f"Optional model tag override (defaults to '{get_default_model()}').",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0.0 for deterministic code, 0.7 for creative). Default: 0.2.",
                        "default": 0.2,
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "ollama_list_models",
            "description": "List all LLM models currently downloaded and available on the remote Google Colab GPU instance.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "ollama_health_check",
            "description": "Check the connectivity status and latency of the Cloudflare Tunnel and remote Google Colab Ollama daemon.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "ollama_set_endpoint",
            "description": (
                "Update the active Cloudflare Tunnel URL (*.trycloudflare.com) for Colab Ollama, "
                "save it to mcp_config.json, verify GPU connectivity, and switch to the remote Colab model."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "The public Cloudflare Quick Tunnel URL (e.g., 'https://xxxx.trycloudflare.com').",
                    },
                },
                "required": ["endpoint"],
            },
        },
    ]


def handle_request(line: str, endpoint: str) -> None:
    """
    Parse and route incoming JSON-RPC 2.0 messages received from the IDE client over `sys.stdin`.

    Supported MCP lifecycle methods:
      1. `initialize`                -> Handshake protocol version and server capabilities.
      2. `notifications/initialized` -> One-way client acknowledgment (no response required).
      3. `ping`                      -> Keep-alive check from the IDE.
      4. `tools/list`                -> Advertise available tools and their JSON input schemas.
      5. `tools/call`                -> Execute the requested tool (`ollama_chat`, `ollama_list_models`, etc.).
    """
    try:
        req = json.loads(line)
    except Exception:
        # Ignore malformed non-JSON lines
        return

    req_id = req.get("id")
    method = req.get("method", "")

    # Ignore JSON-RPC notifications (messages without an 'id' field, such as 'notifications/initialized')
    if req_id is None:
        return

    # 1. MCP Handshake Initialization
    if method == "initialize":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        })

    # 2. Health Ping
    elif method == "ping":
        send_response({"jsonrpc": "2.0", "id": req_id, "result": {}})

    # 3. Advertise Available MCP Tools
    elif method == "tools/list":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": get_tool_definitions()},
        })

    # 4. Execute Tool Call
    elif method == "tools/call":
        # Dynamically hot-reload the latest tunnel URL from config on every call (zero IDE reload needed)
        endpoint = get_endpoint()
        params = req.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {}) or {}

        if tool_name == "ollama_chat":
            prompt = args.get("prompt", "")
            system = args.get("system")
            model = args.get("model")
            temperature = float(args.get("temperature", 0.2))

            output, elapsed, used_model = call_ollama_api(
                endpoint=endpoint,
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
            )
            footer = (
                f"\n\n[⏱️ Completed in {elapsed:.2f}s via Colab GPU ({used_model})]"
                if elapsed > 0
                else ""
            )
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"{output}{footer}"}],
                },
            })

        elif tool_name == "ollama_list_models":
            try:
                models = fetch_available_models(endpoint)
                if not models:
                    text = f"Connected to {endpoint}, but no models are currently installed. Run `!ollama pull qwen2.5-coder:14b` on Colab."
                else:
                    lines = [f"Available models on Colab ({endpoint}):"]
                    for m in models:
                        name = m.get("name", "unknown")
                        size_gb = m.get("size", 0) / (1024 ** 3)
                        lines.append(f"  - {name} ({size_gb:.2f} GB)")
                    text = "\n".join(lines)
            except Exception as e:
                text = f"Failed to list models from '{endpoint}': {str(e)}"

            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            })

        elif tool_name == "ollama_health_check":
            text = format_server_status(endpoint)
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            })

        elif tool_name == "ollama_set_endpoint":
            new_url = args.get("endpoint", "")
            text = set_endpoint_in_configs(new_url, quiet=True)
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": text}]},
            })

        else:
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            })

    # 5. Fallback for unsupported JSON-RPC methods
    else:
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        })


def format_server_status(endpoint: str) -> str:
    """
    Build a detailed telemetry report of the remote Colab Ollama server by querying both
    `GET /api/tags` (installed disk models) and `GET /api/ps` (active GPU VRAM allocation).
    """
    start_t = time.time()
    try:
        models = fetch_available_models(endpoint, timeout=10)
        latency_ms = (time.time() - start_t) * 1000
        installed = (
            ", ".join(f"{m.get('name', '')} ({m.get('size', 0) / (1024**3):.1f} GB)" for m in models)
            or "None"
        )
        lines = [
            f"🟢 ONLINE: Connected to Colab Ollama at {endpoint}",
            f"⚡ Tunnel Latency: {latency_ms:.0f} ms",
            f"📦 Installed Models (Disk): {installed}",
        ]
        try:
            running = fetch_running_models(endpoint, timeout=10)
            if running:
                lines.append("🔥 Active Models Pinned in Colab GPU VRAM (/api/ps):")
                for rm in running:
                    r_name = rm.get("name", "unknown")
                    size_total = rm.get("size", 0)
                    size_vram = rm.get("size_vram", 0)
                    vram_gb = size_vram / (1024 ** 3)
                    total_gb = size_total / (1024 ** 3)
                    gpu_pct = (size_vram / size_total * 100) if size_total > 0 else 100.0
                    quant = rm.get("details", {}).get("quantization_level", "unknown")
                    lines.append(
                        f"   • {r_name} [{quant}] — VRAM Used: {vram_gb:.2f} GB / {total_gb:.2f} GB "
                        f"({gpu_pct:.0f}% GPU offload)"
                    )
            else:
                lines.append(
                    "💤 GPU VRAM Status: Idle (0.0 GB allocated — model will auto-load into VRAM on first prompt)"
                )
        except Exception:
            pass
        return "\n".join(lines)
    except Exception as e:
        return f"🔴 OFFLINE: Cannot reach Colab Ollama at {endpoint}.\nReason: {str(e)}"


def set_endpoint_in_configs(new_url: str, quiet: bool = False) -> str:
    """
    Update the Cloudflare Tunnel URL inside `~/.gemini/config/mcp_config.json` and
    local `configs/antigravity_mcp_config.json`, then verify connectivity against Colab GPU.
    Can be invoked via the `ollama_set_endpoint` MCP tool directly inside an active `agy` chat
    or via `--set-endpoint <URL>` on the command line.
    """
    clean_url = new_url.strip().rstrip("/")
    script_path = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(script_path))
    model = get_default_model()
    logs: List[str] = []

    targets = [
        os.path.expanduser("~/.gemini/config/mcp_config.json"),
        os.path.join(repo_root, "configs", "antigravity_mcp_config.json"),
    ]

    for cfg_path in targets:
        try:
            data: Dict[str, Any] = {"mcpServers": {}}
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            if "mcpServers" not in data:
                data["mcpServers"] = {}
            data["mcpServers"]["colab-ollama"] = {
                "command": "python3",
                "args": [script_path, "--endpoint", clean_url, "--model", model],
            }
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            logs.append(f"✅ Updated MCP config: {cfg_path}")
        except Exception as e:
            logs.append(f"⚠️ Could not update {cfg_path}: {e}")

    logs.append(format_server_status(clean_url))

    result_msg = "\n".join(logs)
    if not quiet:
        print(result_msg)
    return result_msg


def main() -> None:
    """
    Main event loop maintaining the long-lived MCP server subprocess.

    Technical mechanics:
      - If invoked with `--set-endpoint <URL>`, updates MCP config files and exits.
      - If invoked with `--status`, prints live tunnel latency and GPU VRAM usage and exits.
      - Otherwise, resolves the Cloudflare Tunnel endpoint via `get_endpoint()`.
      - Iterates over `sys.stdin` using OS kernel blocking I/O (consumes 0% CPU while waiting).
      - Automatically exits cleanly when the parent IDE closes the pipe (`EOF`).
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--set-endpoint" and i + 1 < len(sys.argv):
            set_endpoint_in_configs(sys.argv[i + 1])
            return
        if arg == "--status":
            print(format_server_status(get_endpoint()))
            return

    endpoint = get_endpoint()
    for line in sys.stdin:
        stripped = line.strip()
        if stripped:
            handle_request(stripped, endpoint)


if __name__ == "__main__":
    main()


