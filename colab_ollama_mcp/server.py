#!/usr/bin/env python3
"""
Colab Ollama MCP Server
-----------------------
A lightweight Model Context Protocol (MCP) server that exposes an Ollama instance
running on a remote Google Colab GPU runtime (or local Ollama) to local AI clients
like Claude Desktop, Antigravity IDE, Cursor, and Cline.

Compatible with Python 3.9+ using zero external dependencies (pure standard library).
"""

import sys
import json
import os
import argparse
import urllib.request
import urllib.error
import time
from typing import Dict, Any, List, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "colab-ollama-mcp"
SERVER_VERSION = "1.0.0"

class OllamaClient:
    """Client for communicating with the remote Ollama HTTP REST API."""
    def __init__(self, endpoint: str, timeout: int = 180):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.endpoint}{path}"
        headers = {"Content-Type": "application/json"}
        req_data = json.dumps(data).encode("utf-8") if data is not None else None
        
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        try:
            start_time = time.time()
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                latency_ms = round((time.time() - start_time) * 1000, 2)
                resp_bytes = response.read()
                resp_text = resp_bytes.decode("utf-8")
                
                # Some Ollama streaming endpoints return multiple JSON lines
                if "\n" in resp_text.strip():
                    lines = resp_text.strip().split("\n")
                    try:
                        return json.loads(lines[-1])
                    except json.JSONDecodeError:
                        pass
                
                result = json.loads(resp_text)
                if isinstance(result, dict):
                    result["_latency_ms"] = latency_ms
                return result
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP error {e.code}: {e.reason} - {err_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.endpoint}. Check Colab tunnel status. Error: {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error communicating with Ollama: {str(e)}")

    def list_models(self) -> Dict[str, Any]:
        return self._request("/api/tags")

    def list_running_models(self) -> Dict[str, Any]:
        return self._request("/api/ps")

    def chat(self, model: str, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if options:
            payload["options"] = options
        return self._request("/api/chat", method="POST", data=payload)

    def generate(self, model: str, prompt: str, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        return self._request("/api/generate", method="POST", data=payload)

    def pull(self, model_name: str) -> Dict[str, Any]:
        payload = {"name": model_name, "stream": False}
        return self._request("/api/pull", method="POST", data=payload)


class MCPServer:
    """Standard JSON-RPC stdio MCP Server."""
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client
        self.tools = self._register_tools()

    def _register_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "ollama_list_models",
                "description": "List all LLM models currently downloaded and available on the Google Colab GPU instance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "ollama_chat",
                "description": "Send a chat prompt or multi-turn conversation to an LLM running on Google Colab GPU (e.g., qwen2.5-coder:7b, llama3.1:8b).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Name of the model to use (e.g., 'qwen2.5-coder:7b', 'llama3.1:8b'). If not specified, uses the first available model."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The message or instruction to send to the model."
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "Optional system prompt to guide model behavior or role."
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature between 0.0 (precise, deterministic) and 1.0 (creative). Default is 0.2."
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "ollama_generate",
                "description": "Generate text or code completion for a raw prompt using the remote Ollama model.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Model tag to use (e.g., 'qwen2.5-coder:7b')."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Raw prompt text."
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature. Default is 0.2."
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "ollama_status",
                "description": "Check the health, tunnel round-trip latency, and VRAM/active model status of the Colab Ollama daemon.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "ollama_pull_model",
                "description": "Pull (download) a new model from the Ollama library directly into the Google Colab environment.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Name of the model to pull (e.g., 'deepseek-r1:8b', 'qwen2.5:14b')."
                        }
                    },
                    "required": ["model_name"]
                }
            }
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "ollama_list_models":
                res = self.client.list_models()
                models = res.get("models", [])
                if not models:
                    return "No models found on remote Colab Ollama. Use 'ollama_pull_model' to download one."
                
                output = [f"Found {len(models)} model(s) on Colab Ollama (Endpoint: {self.client.endpoint}):\n"]
                for m in models:
                    size_gb = round(m.get("size", 0) / (1024**3), 2)
                    modified = m.get("modified_at", "N/A")[:10]
                    output.append(f"- **{m.get('name')}** | Size: {size_gb} GB | Modified: {modified}")
                return "\n".join(output)

            elif name == "ollama_chat":
                prompt = args.get("prompt", "")
                system_prompt = args.get("system_prompt")
                temp = float(args.get("temperature", 0.2))
                
                # Determine model
                model = args.get("model")
                if not model:
                    models_res = self.client.list_models()
                    models = models_res.get("models", [])
                    if models:
                        model = models[0].get("name")
                    else:
                        return "Error: No model available on Colab. Please specify a model or pull one first."

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                res = self.client.chat(model=model, messages=messages, options={"temperature": temp})
                msg = res.get("message", {}).get("content", "")
                eval_count = res.get("eval_count", 0)
                eval_duration_s = res.get("eval_duration", 1) / 1e9
                tps = round(eval_count / eval_duration_s, 1) if eval_duration_s > 0 else 0
                
                footer = f"\n\n---\n*Model: {model} | Speed: {tps} tokens/s | Latency: {res.get('_latency_ms', 'N/A')}ms*"
                return msg + footer

            elif name == "ollama_generate":
                prompt = args.get("prompt", "")
                temp = float(args.get("temperature", 0.2))
                model = args.get("model")
                if not model:
                    models_res = self.client.list_models()
                    models = models_res.get("models", [])
                    if models:
                        model = models[0].get("name")
                    else:
                        return "Error: No model available on Colab."

                res = self.client.generate(model=model, prompt=prompt, options={"temperature": temp})
                response_text = res.get("response", "")
                return response_text

            elif name == "ollama_status":
                start_time = time.time()
                res = self.client.list_models()
                latency_ms = round((time.time() - start_time) * 1000, 2)
                models = res.get("models", [])
                
                # Check running processes
                try:
                    ps_res = self.client.list_running_models()
                    running = ps_res.get("models", [])
                except Exception:
                    running = []

                status_lines = [
                    "🟢 **Colab Ollama Connection Status: ONLINE**",
                    f"- **Endpoint:** `{self.client.endpoint}`",
                    f"- **Network Latency:** ~{latency_ms} ms",
                    f"- **Total Installed Models:** {len(models)}",
                ]
                if running:
                    status_lines.append(f"- **Currently Loaded in VRAM:** {', '.join([r.get('name', 'unknown') for r in running])}")
                else:
                    status_lines.append("- **Currently Loaded in VRAM:** None (idle)")
                
                return "\n".join(status_lines)

            elif name == "ollama_pull_model":
                model_name = args.get("model_name")
                if not model_name:
                    return "Error: 'model_name' parameter is required."
                res = self.client.pull(model_name)
                status = res.get("status", "completed")
                return f"Successfully initiated pull for model '{model_name}'. Status: {status}"

            else:
                return f"Error: Unknown tool '{name}'"

        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION
                    }
                }
            }

        elif method == "notifications/initialized":
            # Client acknowledgment; no response required
            return None

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.tools
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            result_text = self._execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ]
                }
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                }
            }

    def run_stdio(self):
        """Main stdio loop reading JSON-RPC from stdin and writing to stdout."""
        # Ensure UTF-8 I/O
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error or internal exception: {str(e)}"
                    }
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Colab Ollama MCP Server")
    parser.add_argument("--endpoint", default=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"),
                        help="Cloudflare / ngrok tunnel URL to Colab Ollama (e.g. https://xxx.trycloudflare.com)")
    parser.add_argument("--test", action="store_true", help="Test connection to Ollama and list tools without starting stdio server")
    args = parser.parse_args()

    client = OllamaClient(endpoint=args.endpoint)
    server = MCPServer(ollama_client=client)

    if args.test:
        print(f"Testing connection to Ollama at {args.endpoint}...")
        status = server._execute_tool("ollama_status", {})
        print(status)
        print("\nRegistered MCP Tools:")
        for t in server.tools:
            print(f"- {t['name']}: {t['description']}")
        return

    # Start stdio MCP Server
    server.run_stdio()


if __name__ == "__main__":
    main()
