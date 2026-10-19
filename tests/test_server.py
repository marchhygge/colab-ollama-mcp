#!/usr/bin/env python3
"""
Unit test suite for colab_ollama_mcp.server using Python stdlib unittest and unittest.mock.
"""

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from colab_ollama_mcp import server


class _MockHTTPResponse(io.BytesIO):
    """
    Context-manager-compatible HTTP response mock supporting both `.read()` (single JSON)
    and line iteration (`for line in response:` for NDJSON streaming).
    """

    def __init__(self, payload: object) -> None:
        if isinstance(payload, (bytes, bytearray)):
            raw = bytes(payload)
        elif isinstance(payload, str):
            raw = payload.encode("utf-8")
        else:
            raw = (json.dumps(payload) + "\n").encode("utf-8")
        super().__init__(raw)

    def __enter__(self) -> "_MockHTTPResponse":
        self.seek(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class TestColabOllamaMCPServer(unittest.TestCase):
    """Unit tests for MCP tool definitions, model resolution, API transport, and JSON-RPC dispatch."""

    ENDPOINT = "https://test-tunnel.trycloudflare.com"

    def setUp(self) -> None:
        cache = getattr(server, "_MODEL_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()

    def test_get_tool_definitions_exposes_all_four_tools(self) -> None:
        tools = server.get_tool_definitions()
        self.assertEqual(len(tools), 4)
        names = [t["name"] for t in tools]
        self.assertEqual(
            names,
            [
                "ollama_chat",
                "ollama_list_models",
                "ollama_health_check",
                "ollama_set_endpoint",
            ],
        )
        chat_schema = tools[0]["inputSchema"]
        self.assertIn("prompt", chat_schema["required"])
        self.assertIn("temperature", chat_schema["properties"])

    @patch("urllib.request.urlopen")
    def test_fetch_available_models_parses_tags_payload(self, mock_urlopen) -> None:
        mock_models = [
            {"name": "qwen2.5-coder:14b", "size": 9_000_000_000},
            {"name": "deepseek-r1:14b", "size": 9_100_000_000},
        ]
        mock_urlopen.return_value = _MockHTTPResponse({"models": mock_models})

        result = server.fetch_available_models(self.ENDPOINT)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "qwen2.5-coder:14b")

    @patch("colab_ollama_mcp.server.fetch_available_models")
    def test_resolve_model_exact_prefix_and_fallback(self, mock_fetch) -> None:
        mock_fetch.return_value = [
            {"name": "qwen2.5-coder:7b-instruct"},
            {"name": "llama3.1:8b"},
        ]
        # Exact match
        self.assertEqual(
            server.resolve_model(self.ENDPOINT, "llama3.1:8b"),
            "llama3.1:8b",
        )
        # Prefix match ("qwen2.5-coder:14b" -> "qwen2.5-coder:7b-instruct")
        self.assertEqual(
            server.resolve_model(self.ENDPOINT, "qwen2.5-coder:14b"),
            "qwen2.5-coder:7b-instruct",
        )
        # Fallback to first installed model when requested model family is absent
        self.assertEqual(
            server.resolve_model(self.ENDPOINT, "codellama:34b"),
            "qwen2.5-coder:7b-instruct",
        )

    @patch("urllib.request.urlopen")
    @patch("colab_ollama_mcp.server.resolve_model", return_value="qwen2.5-coder:14b")
    def test_call_ollama_api_success_and_url_error(self, _mock_resolve, mock_urlopen) -> None:
        mock_urlopen.return_value = _MockHTTPResponse(
            {"message": {"role": "assistant", "content": "def add(a, b): return a + b"}, "done": True}
        )
        content, elapsed, used_model = server.call_ollama_api(
            endpoint=self.ENDPOINT,
            prompt="Write add function",
            model="qwen2.5-coder:14b",
            system="Return Python only",
            temperature=0.1,
        )
        self.assertEqual(content, "def add(a, b): return a + b")
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(used_model, "qwen2.5-coder:14b")

        # Simulate tunnel disconnect
        mock_urlopen.side_effect = urllib.error.URLError("Tunnel connection refused")
        err_msg, err_elapsed, _ = server.call_ollama_api(
            endpoint=self.ENDPOINT,
            prompt="Ping",
        )
        self.assertEqual(err_elapsed, 0.0)
        self.assertIn("Connection Error", err_msg)

    @patch("colab_ollama_mcp.server.fetch_running_models")
    @patch("colab_ollama_mcp.server.fetch_available_models")
    def test_format_server_status_online_and_offline(self, mock_avail, mock_running) -> None:
        mock_avail.return_value = [{"name": "qwen2.5-coder:14b", "size": 9 * (1024 ** 3)}]
        mock_running.return_value = [
            {
                "name": "qwen2.5-coder:14b",
                "size": 9 * (1024 ** 3),
                "size_vram": 9 * (1024 ** 3),
                "details": {"quantization_level": "Q4_K_M"},
            }
        ]
        status_online = server.format_server_status(self.ENDPOINT)
        self.assertIn("ONLINE", status_online)
        self.assertIn("qwen2.5-coder:14b", status_online)
        self.assertIn("Q4_K_M", status_online)

        mock_avail.side_effect = RuntimeError("Cloudflare 530 Origin DNS Error")
        status_offline = server.format_server_status(self.ENDPOINT)
        self.assertIn("OFFLINE", status_offline)

    @patch("colab_ollama_mcp.server.get_endpoint", return_value="https://test-tunnel.trycloudflare.com")
    @patch("colab_ollama_mcp.server.call_ollama_api", return_value=("print('ok')", 0.42, "qwen2.5-coder:14b"))
    def test_handle_request_initialize_ping_list_and_call(self, _mock_chat, _mock_ep) -> None:
        responses = []
        with patch("colab_ollama_mcp.server.send_response", side_effect=responses.append):
            # 1. initialize
            server.handle_request(
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                self.ENDPOINT,
            )
            self.assertEqual(responses[-1]["id"], 1)
            self.assertEqual(responses[-1]["result"]["protocolVersion"], server.PROTOCOL_VERSION)

            # 2. ping
            server.handle_request(
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}),
                self.ENDPOINT,
            )
            self.assertEqual(responses[-1], {"jsonrpc": "2.0", "id": 2, "result": {}})

            # 3. tools/list
            server.handle_request(
                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
                self.ENDPOINT,
            )
            self.assertEqual(len(responses[-1]["result"]["tools"]), 4)

            # 4. tools/call -> ollama_chat
            server.handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "tools/call",
                        "params": {
                            "name": "ollama_chat",
                            "arguments": {"prompt": "Say ok", "temperature": 0.2},
                        },
                    }
                ),
                self.ENDPOINT,
            )
            chat_text = responses[-1]["result"]["content"][0]["text"]
            self.assertIn("print('ok')", chat_text)
            self.assertIn("qwen2.5-coder:14b", chat_text)

            # 5. tools/call -> unknown tool returns -32601
            server.handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {"name": "nonexistent_tool", "arguments": {}},
                    }
                ),
                self.ENDPOINT,
            )
            self.assertEqual(responses[-1]["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
