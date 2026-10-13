"""Integración local sin credenciales reales ni llamadas a proveedores externos."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
ENC = tiktoken.get_encoding("o200k_base")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, payload, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        try:
            self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authenticated(self, method, body=None):
        route = "azure" if self.path.startswith("/azure") else "gateway"
        self.server.seen.append({"method": method, "path": self.path, "route": route,
                                 "body": body, "auth": self.headers.get("Authorization"),
                                 "api_key": self.headers.get("api-key")})
        expected = "azure-test-key" if route == "azure" else "gateway-test-key"
        good = self.headers.get("api-key") == expected if route == "azure" else self.headers.get("Authorization") == "Bearer " + expected
        if not good:
            self.send(401, {"error": "sensitive-error-body " + str(self.headers.get("Authorization"))})
        return good

    def do_GET(self):
        if not self.authenticated("GET"):
            return
        if not self.path.endswith("/models"):
            self.send(404, {})
            return
        self.send(self.server.models_status, self.server.models)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.authenticated("POST", body):
            return
        route = "azure" if self.path.startswith("/azure") else "gateway"
        behavior = self.server.behavior.get(route, {})
        if behavior.get("delay"):
            time.sleep(behavior["delay"])
        if behavior.get("status"):
            self.send(behavior["status"], {"error": "sensitive-error-body " + behavior.get("error", "")})
            return
        messages = body["messages"]
        content = messages[0]["content"]
        if "tools" not in body:
            if len(ENC.encode(content)) > behavior.get("context_limit", float("inf")):
                self.send(400, {"error": {"code": "context_length_exceeded"}})
                return
            found = dict(re.findall(r"TARGET_(start|middle|end)=([a-f0-9]+)", content))
            found["combined"] = found["start"] + ":" + found["end"]
            if behavior.get("lose_middle"):
                found["middle"] = "wrong"
            message = {"role": "assistant", "content": json.dumps(found)}
            finish = behavior.get("finish", "stop")
        elif behavior.get("reject_default") and "reasoning_effort" not in body:
            self.send(400, {"error": "tools require reasoning_effort=none"})
            return
        elif messages[-1]["role"] == "tool":
            result = json.loads(messages[-1]["content"])
            if behavior.get("bad_receipt"):
                result["receipt"] = "fabricated"
            message = {"role": "assistant", "content": json.dumps(result)}
            finish = "stop"
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": behavior.get("call_id", "call_test"), "type": behavior.get("call_type", "function"),
                "function": {"name": behavior.get("call_name", "add"),
                             "arguments": behavior.get("call_args", '{"a":19,"b":23}')}}]}
            finish = behavior.get("tool_finish", "tool_calls")
        usage = {"prompt_tokens": len(ENC.encode(content)) + 8, "completion_tokens": 50}
        if not body.get("stream"):
            self.send(200, {"model": "fake-gpt", "choices": [{"message": message, "finish_reason": finish}], "usage": usage})
            return
        if message.get("tool_calls"):
            call = message["tool_calls"][0]
            deltas = [{"role": "assistant", "tool_calls": [{"index": 0, "id": call["id"][:5], "type": call["type"],
                       "function": {"name": call["function"]["name"][:1], "arguments": call["function"]["arguments"][:5]}}]},
                      {"tool_calls": [{"index": 0, "id": call["id"][5:], "function": {
                       "name": call["function"]["name"][1:], "arguments": call["function"]["arguments"][5:]}}]}]
        else:
            text = message["content"]
            deltas = [{"content": text[:10]}, {"content": text[10:]}]
        chunks = [{"model": "fake-gpt", "choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
        chunks += [{"choices": [{"index": 0, "delta": {}, "finish_reason": None if behavior.get("no_finish") else finish}]},
                   {"choices": [], "usage": usage}]
        if behavior.get("stream_error"):
            chunks.insert(0, {"error": "sensitive-error-body"})
        payload = b": keepalive\n\n" + b"".join(("data: " + json.dumps(c) + "\n\n").encode() for c in chunks)
        if not behavior.get("no_done"):
            payload += b"data: [DONE]\n\n"
        self.send(200, payload, "application/json" if behavior.get("wrong_mime") else "text/event-stream")


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        (ROOT / ".cache").mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.seen = []
        self.server.behavior = {}
        self.server.models_status = 200
        self.server.models = {"data": [{"id": "gpt-5.6-sol"}, {"id": "another-model"}]}
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".cache")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        shutil.copyfile(ROOT / "probe_llm.py", self.directory / "probe_llm.py")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("GATEWAY_", "AZURE_", "PYTHON_DOTENV_"))}
        self.env["TIKTOKEN_CACHE_DIR"] = str(ROOT / ".cache" / "tiktoken")
        self.env.update(GATEWAY_URL=self.base + "/gateway/v1", GATEWAY_KEY="gateway-test-key")

    def azure(self):
        self.env.update(AZURE_URL=self.base + "/azure/openai/v1", AZURE_KEY="azure-test-key", AZURE_MODEL="deployment-sol")

    def run_probe(self, *args, code=0):
        result = subprocess.run([sys.executable, str(self.directory / "probe_llm.py"),
                                 "--sizes", "512,1024", "--repeats", "1", "--encoding", "o200k_base", *args],
                                env=self.env, cwd=self.directory, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        report_file = self.directory / "reports" / "probe_results.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else None
        self.assertNotIn("gateway-test-key", result.stdout + result.stderr)
        self.assertNotIn("azure-test-key", result.stdout + result.stderr)
        self.assertNotIn("sensitive-error-body", result.stdout + result.stderr)
        return report, result

    def test_gateway_two_variables_discovery_context_tools_and_stream(self):
        report, _ = self.run_probe()
        self.assertEqual(report["selected_models"], {"gateway": "gpt-5.6-sol"})
        self.assertEqual(len(report["tests"]), 6)
        self.assertTrue(all(r["ok"] for r in report["tests"]))
        self.assertEqual(self.server.seen[0]["path"], "/gateway/v1/models")
        self.assertTrue(all(r["path"] == "/gateway/v1/chat/completions" for r in self.server.seen[1:]))
        stream = report["tests"][-1]
        self.assertEqual(stream["first"]["finish_reason"], "tool_calls")
        self.assertTrue(stream["second"]["usage"])
        for row in report["tests"][:2]:
            self.assertEqual(row["reported_minus_content"], 8)
            self.assertTrue(all(row["checks"].values()))

    def test_both_routes_send_same_bodies_except_model(self):
        self.azure()
        report, _ = self.run_probe("--route", "both")
        self.assertEqual(len(report["tests"]), 12)
        posts = [r for r in self.server.seen if r["method"] == "POST"]
        bodies = {route: [dict(r["body"]) for r in posts if r["route"] == route] for route in ("gateway", "azure")}
        for route in bodies:
            for body in bodies[route]:
                self.assertEqual(body.pop("model"), "gpt-5.6-sol" if route == "gateway" else "deployment-sol")
        self.assertEqual(bodies["gateway"], bodies["azure"])
        self.assertTrue(all(r["api_key"] == "azure-test-key" and not r["auth"] for r in posts if r["route"] == "azure"))

    def test_azure_alone_without_gateway(self):
        self.azure()
        self.env = {k: v for k, v in self.env.items() if not k.startswith("GATEWAY_")}
        report, _ = self.run_probe("--route", "azure")
        self.assertTrue(all(r["ok"] and r["route"] == "azure" for r in report["tests"]))
        self.assertTrue(all(r["method"] == "POST" for r in self.server.seen))

    def test_legacy_azure_full_url_preserves_version(self):
        self.azure()
        self.env["AZURE_URL"] = self.base + "/azure/openai/deployments/deployment-sol/chat/completions?api-version=2025-01-01-preview"
        self.run_probe("--route", "azure", "--mode", "tools")
        self.assertTrue(all(r["path"].endswith("/chat/completions?api-version=2025-01-01-preview") for r in self.server.seen))

    def test_full_gateway_completion_url_and_trailing_slash(self):
        self.env["GATEWAY_URL"] += "/chat/completions/"
        self.run_probe("--mode", "tools")
        self.assertEqual(self.server.seen[0]["path"], "/gateway/v1/models")
        self.assertTrue(all(r["path"] == "/gateway/v1/chat/completions" for r in self.server.seen[1:]))

    def test_root_gateway_url(self):
        self.env["GATEWAY_URL"] = self.base
        self.run_probe("--mode", "tools")
        self.assertEqual(self.server.seen[0]["path"], "/models")
        self.assertTrue(all(r["path"] == "/chat/completions" for r in self.server.seen[1:]))

    def test_ambiguous_and_unidentified_models_require_selection(self):
        for ids in (["gpt-5.6-sol", "gpt-5.6-sol-2026-07-09"], ["opaque-alias"], ["gpt-5.6-solution"], []):
            with self.subTest(ids=ids):
                self.server.models = {"data": [{"id": name} for name in ids]}
                _, result = self.run_probe("--mode", "tools", code=2)
                self.assertIn("--gateway-model", result.stderr)
        report, _ = self.run_probe("--mode", "tools", "--gateway-model", "opaque-alias")
        self.assertEqual(report["selected_models"]["gateway"], "opaque-alias")

    def test_prefixed_sol_discovery(self):
        self.server.models = {"data": [{"id": "azure/gpt-5.6-sol-2026-07-09"}]}
        report, _ = self.run_probe("--mode", "tools")
        self.assertEqual(report["selected_models"]["gateway"], "azure/gpt-5.6-sol-2026-07-09")

    def test_configured_model_avoids_discovery(self):
        self.env["GATEWAY_MODEL"] = "custom-alias"
        self.server.models_status = 403
        self.run_probe("--mode", "tools")
        self.assertTrue(all(r["method"] == "POST" for r in self.server.seen))

    def test_discovery_errors_are_controlled(self):
        for payload in ([], {"data": None}, b"not JSON"):
            with self.subTest(payload=payload):
                self.server.models = payload
                _, result = self.run_probe(code=2)
                self.assertNotIn("Traceback", result.stderr)
        self.server.models_status = 403
        _, result = self.run_probe(code=2)
        self.assertIn("HTTP 403: auth", result.stderr)

    def test_dotenv_loaded_next_to_script_from_other_cwd_and_external_priority(self):
        self.env.pop("GATEWAY_URL")
        self.env.pop("GATEWAY_KEY")
        (self.directory / ".env").write_text(f'GATEWAY_URL="{self.base}/gateway/v1"\nGATEWAY_KEY="gateway-test-key"\n')
        work = self.directory / "elsewhere"
        work.mkdir()
        result = subprocess.run([sys.executable, str(self.directory / "probe_llm.py"), "--mode", "tools", "--encoding", "o200k_base"],
                                cwd=work, env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.env.update(GATEWAY_URL=self.base + "/gateway/v1", GATEWAY_KEY="gateway-test-key")
        (self.directory / ".env").write_text('GATEWAY_URL="http://127.0.0.1:1"\nGATEWAY_KEY="wrong-file-key"\nGATEWAY_MODEL="custom-alias"\n')
        report, _ = self.run_probe("--mode", "tools")
        self.assertEqual(report["selected_models"]["gateway"], "custom-alias")

    def test_missing_route_variables(self):
        for variable in ("GATEWAY_URL", "GATEWAY_KEY"):
            with self.subTest(variable=variable):
                value = self.env.pop(variable)
                _, result = self.run_probe(code=2)
                self.assertIn(variable, result.stderr)
                self.env[variable] = value
        self.azure()
        self.env.pop("AZURE_MODEL")
        _, result = self.run_probe("--route", "azure", code=2)
        self.assertIn("AZURE_MODEL", result.stderr)

    def test_context_limit_and_default_reasoning_diagnostics(self):
        self.azure()
        self.server.behavior["gateway"] = {"context_limit": 800, "reject_default": True}
        report, _ = self.run_probe("--route", "both", code=1)
        self.assertTrue(all(r["ok"] for r in report["tests"] if r["route"] == "azure"))
        self.assertTrue(any("context_limit" in r.get("error", "") for r in report["tests"]))
        self.assertTrue(any("Azure pasa" in d for d in report["diagnostics"]))
        self.assertTrue(any("reasoning_effort=none" in d for d in report["diagnostics"]))

    def test_accepted_context_can_fail_retrieval_or_finish(self):
        for behavior in ({"lose_middle": True}, {"finish": "length"}):
            with self.subTest(behavior=behavior):
                self.server.behavior["gateway"] = behavior
                report, _ = self.run_probe("--mode", "context", code=1)
                self.assertTrue(all(not r["ok"] for r in report["tests"]))
                self.assertGreater(report["context_summary"]["gateway"]["max_accepted_content_tokens"], 0)
                self.assertEqual(report["context_summary"]["gateway"]["max_all_checks_content_tokens"], 0)

    def test_http_classification_halts_only_selected_route(self):
        self.azure()
        for status, kind in ((401, "auth"), (403, "auth"), (404, "endpoint"), (429, "rate_limit"), (500, "server"), (413, "body_limit")):
            with self.subTest(status=status):
                self.server.behavior["gateway"] = {"status": status, "error": "context_length_exceeded"}
                report, _ = self.run_probe("--route", "both", code=1)
                rows = [r for r in report["tests"] if r["route"] == "gateway"]
                self.assertIn(kind, rows[0]["error"])
                self.assertNotIn("context_limit", rows[0]["error"])
                self.assertTrue(all(r["ok"] for r in report["tests"] if r["route"] == "azure"))
                self.assertEqual(sum(bool(r.get("skipped")) for r in rows), 0 if status == 413 else 5)
                self.assertNotIn("sensitive-error-body", json.dumps(report))

    def test_timeout_is_transport_error_not_context_limit(self):
        self.env["GATEWAY_MODEL"] = "gpt-5.6-sol"
        self.server.behavior["gateway"] = {"delay": .2}
        report, _ = self.run_probe("--mode", "context", "--timeout", "0.05", code=1)
        self.assertIn("Timeout", report["tests"][0]["error"])
        self.assertTrue(report["tests"][1]["skipped"])

    def test_tool_validation_and_receipt_failure(self):
        for behavior in ({"call_id": ""}, {"call_type": "wrong"}, {"call_name": "execute"},
                         {"call_args": '{"a":19.0,"b":23}'}, {"call_args": '{"a":19,"b":23,"cmd":"run"}'},
                         {"tool_finish": "stop"}, {"bad_receipt": True}):
            with self.subTest(behavior=behavior):
                self.server.behavior["gateway"] = behavior
                report, _ = self.run_probe("--mode", "tools", code=1)
                self.assertTrue(all(not r["ok"] for r in report["tests"]))
                self.assertTrue(all(r["stage"] == ("tool_result" if behavior.get("bad_receipt") else "tool_call") for r in report["tests"]))

    def test_stream_requires_done_finish_and_correct_mime_and_no_error(self):
        for behavior in ({"no_done": True}, {"no_finish": True}, {"wrong_mime": True}, {"stream_error": True}):
            with self.subTest(behavior=behavior):
                self.server.behavior["gateway"] = behavior
                report, _ = self.run_probe("--mode", "tools", code=1)
                self.assertTrue(all(r["ok"] for r in report["tests"][:-1]))
                self.assertFalse(report["tests"][-1]["ok"])
                self.assertTrue(any("streaming" in d for d in report["diagnostics"]))

    def test_dry_run_without_credentials_writes_report_and_never_calls_server(self):
        self.env = {k: v for k, v in self.env.items() if not k.startswith(("GATEWAY_", "AZURE_"))}
        report, _ = self.run_probe("--dry-run", "--route", "both")
        self.assertTrue(report["dry_run"])
        self.assertEqual(len(report["tests"]), 2)
        self.assertEqual(self.server.seen, [])

    def test_invalid_cli(self):
        for args in (("--sizes", "a"), ("--sizes", ""), ("--sizes", "1"), ("--repeats", "0"),
                     ("--timeout", "nan"), ("--output-tokens", "0"), ("--mode", "tools", "--dry-run")):
            with self.subTest(args=args):
                _, result = self.run_probe(*args, code=2)
                self.assertNotIn("Traceback", result.stderr)

    def test_invalid_url_and_auth_do_not_leak_input(self):
        for url in ("ftp://host", "https://secret-user:secret-password@host", "https://host/#secret-fragment"):
            with self.subTest(url=url):
                self.env["GATEWAY_URL"] = url
                _, result = self.run_probe(code=2)
                self.assertNotIn("secret-password", result.stderr)
                self.assertNotIn("secret-fragment", result.stderr)
        self.env["GATEWAY_URL"] = self.base
        self.env["GATEWAY_AUTH"] = "incorrect"
        _, result = self.run_probe(code=2)
        self.assertIn("AUTH", result.stderr)

    def test_gitignore_excludes_secrets_and_reports_but_tracks_example(self):
        paths = [".env", ".env.local", ".env~", "production.env", ".venv/a", "reports/run.json", "custom-output.json", ".env.example"]
        result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"], input="\n".join(paths),
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(set(result.stdout.splitlines()), set(paths[:-1]))


if __name__ == "__main__":
    unittest.main()
