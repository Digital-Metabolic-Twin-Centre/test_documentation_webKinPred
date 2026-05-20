#!/usr/bin/env python3
"""Unit tests for GPU embedding precompute orchestration helpers."""

from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


class _FakeRedis:
    def set(self, *args, **kwargs):
        return None

    def get(self, *args, **kwargs):
        return None

    def delete(self, *args, **kwargs):
        return None


fake_progress_service = types.ModuleType("api.services.progress_service")
fake_progress_service.redis_conn = _FakeRedis()
sys.modules["api.services.progress_service"] = fake_progress_service

from api.services import gpu_embed_service as ges  # noqa: E402


class GpuEmbedServiceTests(unittest.TestCase):
    def test_disabled_by_request_skips_planning_even_in_strict_mode(self):
        with patch.dict("os.environ", {"GPU_EMBED_FAIL_CLOSED": "1"}):
            with patch.object(ges, "build_embedding_plan") as build_plan:
                with patch.object(ges, "_base_url") as base_url:
                    result = ges.run_gpu_precompute_if_available(
                        job_public_id="job_cpu",
                        method_key="OmniESI",
                        target="kcat",
                        valid_sequences=["AAAA"],
                        env={},
                        disabled=True,
                    )

        self.assertFalse(result.attempted)
        self.assertFalse(result.used_gpu)
        self.assertFalse(result.completed)
        self.assertFalse(result.failed)
        self.assertEqual(result.reason, "disabled_by_request")
        build_plan.assert_not_called()
        base_url.assert_not_called()

    def test_gpu_submit_sparse_steps_and_tracker_order(self):
        initial_plan = SimpleNamespace(
            need_computation=2,
            gpu_supported=True,
            gpu_reason=None,
            profile="kinform_full",
            seq_id_to_seq={"sid_1": "AAAA", "sid_2": "BBBB"},
        )
        post_plan = SimpleNamespace(
            need_computation=0,
            gpu_supported=True,
            gpu_reason=None,
            profile="kinform_full",
            seq_id_to_seq={"sid_1": "AAAA", "sid_2": "BBBB"},
        )

        events: list[str] = []

        def _http_side_effect(method, url, payload=None, timeout=5.0):
            events.append(f"http:{method}:{url}")
            if method == "POST":
                self.assertIn("step_work", payload)
                self.assertEqual(
                    payload["step_work"],
                    {
                        "kinform_esm2_layers": ["sid_1"],
                        "kinform_prott5_layers": ["sid_2"],
                    },
                )
                self.assertEqual(payload["seq_id_to_seq"], {"sid_1": "AAAA", "sid_2": "BBBB"})
                return {"job_id": "job_1"}
            return {"status": "done"}

        with patch.object(ges, "build_embedding_plan", side_effect=[initial_plan, post_plan]):
            with patch.object(
                ges,
                "gpu_step_work",
                return_value={"kinform_esm2_layers": ["sid_1"], "kinform_prott5_layers": ["sid_2"]},
            ):
                with patch.object(ges, "_base_url", return_value="http://gpu"):
                    with patch.object(ges, "get_gpu_status", return_value={"online": True}):
                        with patch.object(
                            ges,
                            "start_embedding_tracking",
                            side_effect=lambda **_: events.append("start_tracking"),
                        ):
                            with patch.object(ges, "_http_json", side_effect=_http_side_effect):
                                result = ges.run_gpu_precompute_if_available(
                                    job_public_id="job_x",
                                    method_key="KinForm-H",
                                    target="kcat",
                                    valid_sequences=["AAAA", "BBBB"],
                                    env={},
                                )

        self.assertTrue(result.used_gpu)
        self.assertTrue(result.completed)
        self.assertFalse(result.failed)
        self.assertEqual(events[0], "start_tracking")
        self.assertTrue(events[1].startswith("http:POST:"))

    def test_gpu_failure_is_fail_open(self):
        plan = SimpleNamespace(
            need_computation=1,
            gpu_supported=True,
            gpu_reason=None,
            profile="turnup_esm1b",
            seq_id_to_seq={"sid_1": "AAAA"},
        )

        with patch.object(ges, "build_embedding_plan", return_value=plan):
            with patch.object(ges, "gpu_step_work", return_value={"turnup_esm1b": ["sid_1"]}):
                with patch.object(ges, "_base_url", return_value="http://gpu"):
                    with patch.object(ges, "get_gpu_status", return_value={"online": True}):
                        with patch.object(ges, "start_embedding_tracking", return_value=True):
                            with patch.object(ges, "_http_json", return_value={"job_id": "job_1"}):
                                with patch.object(ges, "_poll_job", return_value={"status": "failed"}):
                                    result = ges.run_gpu_precompute_if_available(
                                        job_public_id="job_y",
                                        method_key="TurNup",
                                        target="kcat",
                                        valid_sequences=["AAAA"],
                                        env={},
                                    )

        self.assertTrue(result.used_gpu)
        self.assertFalse(result.completed)
        self.assertTrue(result.failed)
        self.assertEqual(result.reason, "failed")

    def test_gpu_status_cache_and_unconfigured(self):
        with patch.object(ges, "_base_url", return_value=""):
            status = ges.get_gpu_status(force_refresh=True)
        self.assertFalse(status["configured"])

        mock_http = Mock(return_value={"online": True, "gpu_name": "RTX A4500", "free_vram_gb": 8.0})
        with patch.object(ges, "_base_url", return_value="http://gpu"):
            with patch.object(ges, "_http_json", mock_http):
                status1 = ges.get_gpu_status(force_refresh=True)
                status2 = ges.get_gpu_status(force_refresh=False)

        self.assertEqual(mock_http.call_count, 1)
        self.assertTrue(status1["online"])
        self.assertEqual(status2["gpu_name"], "RTX A4500")

    def test_poll_job_times_out_after_configured_deadline(self):
        def _env_side_effect(name: str, default: int) -> int:
            if name == "GPU_EMBED_JOB_TIMEOUT_SECONDS":
                return 21600
            if name == "GPU_EMBED_POLL_LOG_INTERVAL_SECONDS":
                return 120
            return default

        with patch.object(ges, "_env_int", side_effect=_env_side_effect):
            with patch.object(ges, "_http_json", return_value={"status": "running"}) as mock_http:
                with patch.object(ges.time, "monotonic", side_effect=[0.0, 21601.0]):
                    result = ges._poll_job(
                        "http://gpu",
                        "gpu_1",
                        job_public_id="job_t",
                        method_key="CatPred",
                        target="kcat",
                    )

        self.assertEqual(result["status"], "timeout")
        self.assertEqual(mock_http.call_count, 1)

    def test_fail_closed_raises_when_supported_gpu_is_unavailable(self):
        plan = SimpleNamespace(
            need_computation=1,
            gpu_supported=True,
            gpu_reason=None,
            profile="turnup_esm1b",
            seq_id_to_seq={"sid_1": "AAAA"},
        )

        with patch.dict("os.environ", {"GPU_EMBED_FAIL_CLOSED": "1"}):
            with patch.object(ges, "build_embedding_plan", return_value=plan):
                with patch.object(ges, "gpu_step_work", return_value={"turnup_esm1b": ["sid_1"]}):
                    with patch.object(ges, "_base_url", return_value="http://gpu"):
                        with patch.object(ges, "get_gpu_status", return_value={"online": False, "reason": "offline"}):
                            with self.assertRaises(RuntimeError):
                                ges.run_gpu_precompute_if_available(
                                    job_public_id="job_z",
                                    method_key="TurNup",
                                    target="kcat",
                                    valid_sequences=["AAAA"],
                                    env={},
                                )

    def test_fail_closed_does_not_raise_for_unsupported_method(self):
        plan = SimpleNamespace(
            need_computation=1,
            gpu_supported=False,
            gpu_reason="gpu_offload_phase2",
            profile=None,
            seq_id_to_seq={"sid_1": "AAAA"},
        )

        with patch.dict("os.environ", {"GPU_EMBED_FAIL_CLOSED": "1"}):
            with patch.object(ges, "build_embedding_plan", return_value=plan):
                result = ges.run_gpu_precompute_if_available(
                    job_public_id="job_u",
                    method_key="CatPred",
                    target="kcat",
                    valid_sequences=["AAAA"],
                    env={},
                )

        self.assertFalse(result.completed)
        self.assertEqual(result.reason, "gpu_offload_phase2")

    def test_fail_closed_raises_when_remote_reports_done_but_outputs_missing(self):
        initial_plan = SimpleNamespace(
            need_computation=1,
            gpu_supported=True,
            gpu_reason=None,
            profile="prot_t5_mean",
            seq_id_to_seq={"sid_1": "AAAA"},
        )
        post_plan = SimpleNamespace(
            need_computation=1,
            gpu_supported=True,
            gpu_reason=None,
            profile="prot_t5_mean",
            seq_id_to_seq={"sid_1": "AAAA"},
        )

        with patch.dict("os.environ", {"GPU_EMBED_FAIL_CLOSED": "1"}):
            with patch.object(ges, "build_embedding_plan", side_effect=[initial_plan, post_plan]):
                with patch.object(ges, "gpu_step_work", return_value={"prot_t5_mean": ["sid_1"]}):
                    with patch.object(ges, "_base_url", return_value="http://gpu"):
                        with patch.object(ges, "get_gpu_status", return_value={"online": True}):
                            with patch.object(ges, "start_embedding_tracking", return_value=True):
                                with patch.object(ges, "_http_json", return_value={"job_id": "job_1"}):
                                    with patch.object(ges, "_poll_job", return_value={"status": "done"}):
                                        with self.assertRaises(RuntimeError):
                                            ges.run_gpu_precompute_if_available(
                                                job_public_id="job_fc",
                                                method_key="UniKP",
                                                target="kcat",
                                            valid_sequences=["AAAA"],
                                            env={},
                                        )

    def test_poll_timeout_still_succeeds_when_postcheck_is_complete(self):
        initial_plan = SimpleNamespace(
            need_computation=1,
            gpu_supported=True,
            gpu_reason=None,
            profile="catpred_embed",
            seq_id_to_seq={"sid_1": "AAAA"},
        )
        post_plan = SimpleNamespace(
            need_computation=0,
            gpu_supported=True,
            gpu_reason=None,
            profile="catpred_embed",
            seq_id_to_seq={"sid_1": "AAAA"},
        )

        with patch.object(ges, "build_embedding_plan", side_effect=[initial_plan, post_plan]):
            with patch.object(ges, "gpu_step_work", return_value={"catpred_embed_kcat": ["sid_1"]}):
                with patch.object(ges, "_base_url", return_value="http://gpu"):
                    with patch.object(ges, "get_gpu_status", return_value={"online": True}):
                        with patch.object(ges, "start_embedding_tracking", return_value=True):
                            with patch.object(ges, "_http_json", return_value={"job_id": "job_1"}):
                                with patch.object(ges, "_poll_job", return_value={"status": "timeout"}):
                                    result = ges.run_gpu_precompute_if_available(
                                        job_public_id="job_timeout_complete",
                                        method_key="CatPred",
                                        target="kcat",
                                        valid_sequences=["AAAA"],
                                        env={},
                                    )

        self.assertTrue(result.used_gpu)
        self.assertTrue(result.completed)
        self.assertFalse(result.failed)
        self.assertEqual(result.reason, "done_after_poll_timeout")


if __name__ == "__main__":
    unittest.main()
