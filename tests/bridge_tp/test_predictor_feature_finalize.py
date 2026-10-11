# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the authenticated, post-generation journal drain path."""

import ast
import asyncio
import hmac
import json
import os
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from tools.bridge_tp.run_phase9_cap0_noop import finalize_predictor_features

ROOT = Path(__file__).resolve().parents[2]


class HttpError(Exception):
    def __init__(self, *, status_code, detail):
        self.status_code = status_code
        super().__init__(detail)


def endpoint():
    """Load the actual route body without importing the GPU server stack."""
    tree = ast.parse((ROOT / "vllm/bridge_tp/takeover_api.py").read_text())
    route = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "finalize_predictor_features"
    )
    route.decorator_list = []
    namespace = {
        "os": os,
        "hmac": hmac,
        "Request": object,
        "Any": Any,
        "HTTPStatus": HTTPStatus,
        "HTTPException": HttpError,
        "_predictor_finalize_lock": asyncio.Lock(),
    }
    exec(compile(ast.Module(body=[route], type_ignores=[]), "route", "exec"), namespace)
    return namespace[route.name]


def invoke(route, request):
    """Run immediate mocked awaits without opening Windows loop sockets."""
    coroutine = route(request)
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("mock route unexpectedly suspended")


class FeatureFinalizeTests(unittest.TestCase):
    def request(self, body, result=None):
        rpc = AsyncMock(return_value=result)
        return SimpleNamespace(
            json=AsyncMock(return_value=body),
            app=SimpleNamespace(
                state=SimpleNamespace(engine_client=SimpleNamespace(collective_rpc=rpc))
            ),
        ), rpc

    def test_disabled_route_does_not_call_worker(self):
        request, rpc = self.request({"token": "secret"})
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(HttpError) as error,
        ):
            invoke(endpoint(), request)
        self.assertEqual(error.exception.status_code, 404)
        rpc.assert_not_awaited()

    def test_wrong_secret_does_not_call_worker(self):
        request, rpc = self.request({"token": "wrong"})
        with (
            patch.dict(
                os.environ,
                {
                    "BRIDGETP_PREDICTOR_FINALIZE_TOKEN": "secret",
                    "BRIDGETP_PREDICTOR_FEATURE_JOURNAL": "features",
                },
            ),
            self.assertRaises(HttpError) as error,
        ):
            invoke(endpoint(), request)
        self.assertEqual(error.exception.status_code, 403)
        rpc.assert_not_awaited()

    def test_named_worker_confirmation(self):
        result = {"status": "FINALIZED", "saved_batches": 1479}
        request, rpc = self.request({"token": "secret"}, [result])
        with patch.dict(
            os.environ,
            {
                "BRIDGETP_PREDICTOR_FINALIZE_TOKEN": "secret",
                "BRIDGETP_PREDICTOR_FEATURE_JOURNAL": "features",
            },
        ):
            self.assertEqual(invoke(endpoint(), request), result)
        rpc.assert_awaited_once_with(
            "bridge_tp_finalize_predictor_features", timeout=60.0
        )

    def test_missing_or_empty_worker_evidence_fails(self):
        for result in (None, [], [{"status": "FINALIZED", "saved_batches": 0}]):
            request, _ = self.request({"token": "secret"}, result)
            with (
                patch.dict(
                    os.environ,
                    {
                        "BRIDGETP_PREDICTOR_FINALIZE_TOKEN": "secret",
                        "BRIDGETP_PREDICTOR_FEATURE_JOURNAL": "features",
                    },
                ),
                self.assertRaises(HttpError) as error,
            ):
                invoke(endpoint(), request)
            self.assertEqual(error.exception.status_code, 500)

    def test_launcher_persists_confirmation_without_secret(self):
        result = {"status": "FINALIZED", "saved_batches": 2}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "finalization.json"
            with patch(
                "tools.bridge_tp.run_atomic_takeover._post_json", return_value=result
            ) as post:
                finalize_predictor_features("http://127.0.0.1:8001", "secret", output)
            self.assertEqual(json.loads(output.read_text()), result)
            post.assert_called_once_with(
                "http://127.0.0.1:8001",
                "/bridge_tp/v1/predictor_features/finalize",
                {"token": "secret"},
                65,
            )

    def test_launcher_rejects_unconfirmed_persistence(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch(
                "tools.bridge_tp.run_atomic_takeover._post_json",
                return_value={"status": "FAILED"},
            ),
            self.assertRaisesRegex(RuntimeError, "not confirmed"),
        ):
            finalize_predictor_features(
                "http://127.0.0.1:8001",
                "secret",
                Path(temporary) / "finalization.json",
            )
