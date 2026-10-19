#!/usr/bin/env python3
"""
Unit test suite for colab_ollama_mcp.server using Python stdlib unittest and unittest.mock.
"""

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

# Allow running `python3 tests/test_server.py` directly from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from colab_ollama_mcp import server  # noqa: E402


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
        elif isinstance(payload, list):
            # Multi-chunk NDJSON stream (list of dicts or strings)
            lines = [
                item if isinstance(item, str) else json.dumps(item)
                for item in payload
            ]
            raw = ("\n".join(lines) + "\n").encode("utf-8")
        else:
            raw = (json.dumps(payload) + "\n").encode("utf-8")
        super().__init__(raw)

    def __enter__(self) -> "_MockHTTPResponse":
        self.seek(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


class TestColabOllamaMCPServer(unittest.TestCase):
    """Unit tests for SSRF validation, config persistence, NDJSON streaming, and JSON-RPC dispatch."""

    ENDPOINT = "https://test-tunnel.trycloudflare.com"

    def setUp(self) -> None:
        cache = getattr(server, "_MODEL_CACHE", None)
        if isinstance(cache, dict):
            cache.clear()
        cache_ids = getattr(server, "_CACHE_URLOPEN_IDS", None)
        if isinstance(cache_ids, dict):
            cache_ids.clear()

    def test_validate_endpoint_url_allows_valid_https_and_localhost(self) -> None:
        # Valid Cloudflare & ngrok HTTPS tunnels
        ok, url = server.validate_endpoint_url("https://random-words-123.trycloudflare.com/")
        self.assertTrue(ok)
        self.assertEqual(url, "https://random-words-123.trycloudflare.com")

        ok_ngrok, url_ngrok = server.validate_endpoint_url("https://abc123.ngrok-free.app")
        self.assertTrue(ok_ngrok)
        self.assertEqual(url_ngrok, "https://abc123.ngrok-free.app")

        # Valid local HTTP loopback endpoints
        ok_local, url_local = server.validate_endpoint_url("http://localhost:11434/")
        self.assertTrue(ok_local)
        self.assertEqual(url_local, "http://localhost:11434")

        ok_loop, url_loop = server.validate_endpoint_url("http://127.0.0.1:11434")
        self.assertTrue(ok_loop)
        self.assertEqual(url_loop, "http://127.0.0.1:11434")

    def test_validate_endpoint_url_blocks_ssrf_and_invalid_schemes(self) -> None:
        blocked_urls = [
            "",
            "   ",
            "file:///etc/passwd",
            "ftp://example.com/model",
            "javascript:alert(1)",
            "http://169.254.169.254/latest/meta-data/",
            "https://169.254.169.254",
            "http://10.0.0.1:11434",
            "https://192.168.1.50:11434",
            "https://172.16.0.5:11434",
            "http://0.0.0.0:11434",
            "http://test-tunnel.trycloudflare.com",  # Insecure HTTP on public domain
            "https://singlelabelhost",               # Missing public TLD
            "https://example.com:99999",             # Invalid port
        ]
        for bad_url in blocked_urls:
            with self.subTest(bad_url=bad_url):
                ok, reason = server.validate_endpoint_url(bad_url)
                self.assertFalse(ok, f"Expected {bad_url!r} to be blocked, but got True")
                self.assertTrue(len(reason) > 0)

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

    def test_get_endpoint_and_default_model_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_script = os.path.join(tmpdir, "colab_ollama_mcp", "server.py")
            cfg_dir = os.path.join(tmpdir, "configs")
            os.makedirs(cfg_dir, exist_ok=True)
            cfg_file = os.path.join(cfg_dir, "antigravity_mcp_config.json")

            # 1. Config has live tunnel URL -> returns config URL
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mcpServers": {
                            "colab-ollama": {
                                "command": "python3",
                                "args": [fake_script, "--endpoint", "https://from-config.trycloudflare.com/"],
                            }
                        }
                    },
                    f,
                )

            with patch.object(server, "__file__", fake_script), patch(
                "os.path.expanduser", return_value=os.path.join(tmpdir, "nonexistent.json")
            ):
                self.assertEqual(server.get_endpoint(), "https://from-config.trycloudflare.com")

            # 2. Config has placeholder YOUR-TUNNEL-URL -> falls back to sys.argv / env
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mcpServers": {
                            "colab-ollama": {
                                "command": "python3",
                                "args": [fake_script, "--endpoint", "https://YOUR-TUNNEL-URL.trycloudflare.com"],
                            }
                        }
                    },
                    f,
                )

            with patch.object(server, "__file__", fake_script), patch(
                "os.path.expanduser", return_value=os.path.join(tmpdir, "nonexistent.json")
            ), patch.object(
                sys, "argv", ["server.py", "--endpoint", "https://from-argv.trycloudflare.com/", "--model", "deepseek-r1:14b"]
            ):
                self.assertEqual(server.get_endpoint(), "https://from-argv.trycloudflare.com")
                self.assertEqual(server.get_default_model(), "deepseek-r1:14b")

    @patch("urllib.request.urlopen")
    def test_fetch_available_models_and_ttl_cache(self, mock_urlopen) -> None:
        mock_models = [
            {"name": "qwen2.5-coder:14b", "size": 9_000_000_000},
            {"name": "deepseek-r1:14b", "size": 9_100_000_000},
        ]
        mock_urlopen.return_value = _MockHTTPResponse({"models": mock_models})

        # First call populates cache
        result1 = server.fetch_available_models(self.ENDPOINT, use_cache=True)
        self.assertEqual(len(result1), 2)
        self.assertEqual(result1[0]["name"], "qwen2.5-coder:14b")
        self.assertEqual(mock_urlopen.call_count, 1)

        # Second call with use_cache=True hits cache without extra HTTP request
        result2 = server.fetch_available_models(self.ENDPOINT, use_cache=True)
        self.assertEqual(len(result2), 2)
        self.assertEqual(mock_urlopen.call_count, 1)

        # Call with use_cache=False bypasses cache
        mock_urlopen.return_value = _MockHTTPResponse({"models": mock_models[:1]})
        result3 = server.fetch_available_models(self.ENDPOINT, use_cache=False)
        self.assertEqual(len(result3), 1)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_fetch_running_models_parses_ps_payload(self, mock_urlopen) -> None:
        running_payload = {
            "models": [
                {
                    "name": "qwen2.5-coder:14b",
                    "size": 9 * (1024 ** 3),
                    "size_vram": 9 * (1024 ** 3),
                    "details": {"quantization_level": "Q4_K_M"},
                }
            ]
        }
        mock_urlopen.return_value = _MockHTTPResponse(running_payload)
        running = server.fetch_running_models(self.ENDPOINT)
        self.assertEqual(len(running), 1)
        self.assertEqual(running[0]["name"], "qwen2.5-coder:14b")
        self.assertEqual(running[0]["details"]["quantization_level"], "Q4_K_M")

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
    def test_call_ollama_api_single_json_and_errors(self, _mock_resolve, mock_urlopen) -> None:
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

        # Simulate tunnel disconnect (URLError)
        mock_urlopen.side_effect = urllib.error.URLError("Tunnel connection refused")
        err_msg, err_elapsed, _ = server.call_ollama_api(
            endpoint=self.ENDPOINT,
            prompt="Ping",
        )
        self.assertEqual(err_elapsed, 0.0)
        self.assertIn("Connection Error", err_msg)

        # Simulate HTTP 500 error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url=f"{self.ENDPOINT}/api/chat",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"CUDA out of memory"),
        )
        http_err_msg, http_elapsed, _ = server.call_ollama_api(
            endpoint=self.ENDPOINT,
            prompt="Ping",
        )
        self.assertEqual(http_elapsed, 0.0)
        self.assertIn("HTTP Error 500", http_err_msg)
        self.assertIn("CUDA out of memory", http_err_msg)

        # Invalid SSRF endpoint rejected before network call
        ssrf_msg, ssrf_elapsed, _ = server.call_ollama_api(
            endpoint="http://169.254.169.254/latest",
            prompt="Ping",
        )
        self.assertEqual(ssrf_elapsed, 0.0)
        self.assertIn("Invalid endpoint URL", ssrf_msg)

    @patch("urllib.request.urlopen")
    @patch("colab_ollama_mcp.server.resolve_model", return_value="qwen2.5-coder:14b")
    def test_call_ollama_api_multi_chunk_ndjson_streaming(self, _mock_resolve, mock_urlopen) -> None:
        ndjson_chunks = [
            {"message": {"role": "assistant", "content": "def "}, "done": False},
            {"message": {"role": "assistant", "content": "multiply(x, y):\n"}, "done": False},
            {"message": {"role": "assistant", "content": "    return x * y"}, "done": True},
        ]
        mock_urlopen.return_value = _MockHTTPResponse(ndjson_chunks)

        content, elapsed, used_model = server.call_ollama_api(
            endpoint=self.ENDPOINT,
            prompt="Write multiply function",
        )
        self.assertEqual(content, "def multiply(x, y):\n    return x * y")
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(used_model, "qwen2.5-coder:14b")

    @patch("colab_ollama_mcp.server.format_server_status", return_value="🟢 ONLINE: Mock Status")
    def test_set_endpoint_in_configs_updates_files_and_preserves_env(self, _mock_status) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pkg_dir = os.path.join(tmpdir, "colab_ollama_mcp")
            os.makedirs(fake_pkg_dir, exist_ok=True)
            fake_script = os.path.join(fake_pkg_dir, "server.py")
            fake_home_cfg = os.path.join(tmpdir, "home_gemini", "mcp_config.json")
            cfg_dir = os.path.join(tmpdir, "configs")
            os.makedirs(cfg_dir, exist_ok=True)

            antigravity_cfg = os.path.join(cfg_dir, "antigravity_mcp_config.json")
            claude_cfg = os.path.join(cfg_dir, "claude_desktop_config.json")

            # Seed existing config with custom env key + OLLAMA_ENDPOINT
            with open(antigravity_cfg, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "_comment": "Antigravity config",
                        "mcpServers": {
                            "colab-ollama": {
                                "command": "python3",
                                "args": [fake_script, "--endpoint", "https://old.trycloudflare.com"],
                                "env": {
                                    "OLLAMA_ENDPOINT": "https://old.trycloudflare.com",
                                    "OLLAMA_TIMEOUT": "180",
                                    "CUSTOM_KEY": "keep_me",
                                },
                            }
                        },
                    },
                    f,
                )
            with open(claude_cfg, "w", encoding="utf-8") as f:
                json.dump({"mcpServers": {}}, f)

            new_tunnel = "https://new-gpu-tunnel.trycloudflare.com/"
            with patch.object(server, "__file__", fake_script), patch(
                "os.path.expanduser", return_value=fake_home_cfg
            ):
                res_text = server.set_endpoint_in_configs(new_tunnel, quiet=True)
                self.assertIn("Updated MCP config", res_text)
                self.assertIn("🟢 ONLINE", res_text)

                # Verify primary home config was created
                self.assertTrue(os.path.exists(fake_home_cfg))
                with open(fake_home_cfg, "r", encoding="utf-8") as f:
                    home_data = json.load(f)
                self.assertIn(
                    "https://new-gpu-tunnel.trycloudflare.com",
                    home_data["mcpServers"]["colab-ollama"]["args"],
                )

                # Verify antigravity config preserved CUSTOM_KEY and updated OLLAMA_ENDPOINT
                with open(antigravity_cfg, "r", encoding="utf-8") as f:
                    ag_data = json.load(f)
                ag_entry = ag_data["mcpServers"]["colab-ollama"]
                self.assertEqual(ag_entry["env"]["CUSTOM_KEY"], "keep_me")
                self.assertEqual(
                    ag_entry["env"]["OLLAMA_ENDPOINT"],
                    "https://new-gpu-tunnel.trycloudflare.com",
                )

                # Verify optional existing claude_desktop_config.json was also updated
                with open(claude_cfg, "r", encoding="utf-8") as f:
                    cl_data = json.load(f)
                self.assertIn(
                    "https://new-gpu-tunnel.trycloudflare.com",
                    cl_data["mcpServers"]["colab-ollama"]["args"],
                )

                # Verify invalid SSRF URL is rejected and does not overwrite configs
                bad_res = server.set_endpoint_in_configs("file:///etc/passwd", quiet=True)
                self.assertIn("❌ Invalid endpoint URL", bad_res)

    @patch("colab_ollama_mcp.server.fetch_running_models")
    @patch("colab_ollama_mcp.server.fetch_available_models")
    def test_format_server_status_online_idle_and_offline(self, mock_avail, mock_running) -> None:
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

        # Idle VRAM state (no models in /api/ps)
        mock_running.return_value = []
        status_idle = server.format_server_status(self.ENDPOINT)
        self.assertIn("Idle", status_idle)

        # Offline state
        mock_avail.side_effect = RuntimeError("Cloudflare 530 Origin DNS Error")
        status_offline = server.format_server_status(self.ENDPOINT)
        self.assertIn("OFFLINE", status_offline)

    @patch("colab_ollama_mcp.server.get_endpoint", return_value="https://test-tunnel.trycloudflare.com")
    @patch("colab_ollama_mcp.server.call_ollama_api", return_value=("print('ok')", 0.42, "qwen2.5-coder:14b"))
    @patch("colab_ollama_mcp.server.fetch_available_models")
    @patch("colab_ollama_mcp.server.format_server_status", return_value="🟢 ONLINE: Healthy")
    @patch("colab_ollama_mcp.server.set_endpoint_in_configs", return_value="✅ Updated MCP config\n🟢 ONLINE")
    def test_handle_request_all_methods_and_four_tools(
        self,
        _mock_set_ep,
        _mock_status,
        mock_list_models,
        _mock_chat,
        _mock_ep,
    ) -> None:
        mock_list_models.return_value = [
            {"name": "qwen2.5-coder:14b", "size": 9 * (1024 ** 3)},
            {"name": "deepseek-r1:14b", "size": 9 * (1024 ** 3)},
        ]
        responses = []
        with patch("colab_ollama_mcp.server.send_response", side_effect=responses.append):
            # 0. Malformed JSON & notification (no id) are ignored
            server.handle_request("not valid json {{{", self.ENDPOINT)
            server.handle_request(
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                self.ENDPOINT,
            )
            self.assertEqual(len(responses), 0)

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

            # 4a. tools/call -> ollama_chat
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
            self.assertFalse(responses[-1]["result"].get("isError", False))

            # 4b. tools/call -> ollama_list_models
            server.handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "tools/call",
                        "params": {"name": "ollama_list_models", "arguments": {}},
                    }
                ),
                self.ENDPOINT,
            )
            list_text = responses[-1]["result"]["content"][0]["text"]
            self.assertIn("Available models on Colab", list_text)
            self.assertIn("qwen2.5-coder:14b", list_text)
            self.assertFalse(responses[-1]["result"].get("isError", False))

            # 4c. tools/call -> ollama_health_check
            server.handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 6,
                        "method": "tools/call",
                        "params": {"name": "ollama_health_check", "arguments": {}},
                    }
                ),
                self.ENDPOINT,
            )
            health_text = responses[-1]["result"]["content"][0]["text"]
            self.assertIn("🟢 ONLINE: Healthy", health_text)
            self.assertFalse(responses[-1]["result"].get("isError", False))

            # 4d. tools/call -> ollama_set_endpoint
            server.handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 7,
                        "method": "tools/call",
                        "params": {
                            "name": "ollama_set_endpoint",
                            "arguments": {"endpoint": "https://new-tunnel.trycloudflare.com"},
                        },
                    }
                ),
                self.ENDPOINT,
            )
            set_ep_text = responses[-1]["result"]["content"][0]["text"]
            self.assertIn("Updated MCP config", set_ep_text)
            self.assertFalse(responses[-1]["result"].get("isError", False))

            # 4e. tools/call -> error state sets isError: True
            mock_list_models.side_effect = RuntimeError("Connection reset")
            server.handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 8,
                        "method": "tools/call",
                        "params": {"name": "ollama_list_models", "arguments": {}},
                    }
                ),
                self.ENDPOINT,
            )
            self.assertTrue(responses[-1]["result"].get("isError", False))

            # 5. tools/call -> unknown tool returns -32601
            server.handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 9,
                        "method": "tools/call",
                        "params": {"name": "nonexistent_tool", "arguments": {}},
                    }
                ),
                self.ENDPOINT,
            )
            self.assertEqual(responses[-1]["error"]["code"], -32601)

            # 6. Unknown top-level JSON-RPC method returns -32601
            server.handle_request(
                json.dumps({"jsonrpc": "2.0", "id": 10, "method": "resources/list"}),
                self.ENDPOINT,
            )
            self.assertEqual(responses[-1]["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
