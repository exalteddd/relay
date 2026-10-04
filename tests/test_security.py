"""Regression checks for public-file access and paid-run authorization."""
import importlib
import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"SESSION_SECRET": "test-secret-not-a-credential"}):
    backend = importlib.import_module("server.app")


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = backend.app.test_client()

    def test_only_frontend_files_are_public(self):
        for path in ("/.env", "/.git/config", "/server/app.py", "/render.yaml",
                     "/medlab/relay_data/events.jsonl", "/assets/../server/app.py"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
        for path in ("/", "/assets/relay-mark.svg", "/assets/fonts/source-serif-4-latin.woff2"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            response.close()

    def test_production_requires_explicit_allowlist(self):
        with patch.object(backend, "IS_DEV", False), patch.object(backend, "ALLOWED", set()):
            with self.client.session_transaction() as session:
                session["user"] = "someone"
            self.assertEqual(self.client.post("/api/run", json={}).status_code, 403)
        with backend.app.test_request_context(), patch.object(backend, "IS_DEV", False), patch.object(backend, "ALLOWED", {"exalteddd"}):
            self.assertTrue(backend.may_run("exalteddd"))
            self.assertFalse(backend.may_run("someone"))

    def test_login_redirect_is_local(self):
        with patch.object(backend, "AUTH_CONFIGURED", True):
            for destination in ("https://evil.example", "//evil.example", "/\\evil.example"):
                self.client.get("/auth/login", query_string={"next": destination})
                with self.client.session_transaction() as session:
                    self.assertEqual(session["post_login"], "/")

    def test_cross_origin_posts_are_rejected(self):
        self.assertEqual(self.client.post("/auth/logout", headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.post("/auth/logout", headers={"Origin": "http://localhost"}).status_code, 200)

    def test_input_and_headers(self):
        with patch.object(backend, "may_run", return_value=True):
            self.assertEqual(self.client.post("/api/run", json={"question": 123}).status_code, 400)
            self.assertEqual(self.client.post("/api/run", json=[1]).status_code, 400)
            self.assertEqual(self.client.post("/api/run", json={"question": "x" * 20000}).status_code, 413)
        response = self.client.get("/api/me")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")


if __name__ == "__main__":
    unittest.main()
