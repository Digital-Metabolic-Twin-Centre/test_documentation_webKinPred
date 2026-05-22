#!/usr/bin/env python3
"""Unit tests for generic subprocess seq_id payload attachment."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.prediction_engines import generic_subprocess as gs


class GenericSubprocessSeqIdTests(unittest.TestCase):
    def test_should_attach_seq_ids_for_shared_cache_methods(self):
        self.assertTrue(gs._should_attach_seq_ids(SimpleNamespace(key="OmniESI")))
        self.assertTrue(gs._should_attach_seq_ids(SimpleNamespace(key="RealKcat")))
        self.assertTrue(gs._should_attach_seq_ids(SimpleNamespace(key="IECata")))
        self.assertFalse(gs._should_attach_seq_ids(SimpleNamespace(key="DLKcat")))

    def test_attach_seq_ids_to_rows_sets_payload_ids(self):
        rows = [{"sequence": "AAAA"}, {"sequence": "BBBB"}]
        with patch.object(gs, "resolve_media_and_tools", return_value=("/tmp/media", "/tmp/tools")):
            with patch.object(gs, "resolve_seq_ids_via_cli", return_value=["sid_a", "sid_b"]):
                gs._attach_seq_ids_to_rows(
                    desc=SimpleNamespace(key="RealKcat"),
                    rows=rows,
                    sequences=["AAAA", "BBBB"],
                    env={},
                )

        self.assertEqual(rows[0]["seq_id"], "sid_a")
        self.assertEqual(rows[1]["seq_id"], "sid_b")


if __name__ == "__main__":
    unittest.main()
