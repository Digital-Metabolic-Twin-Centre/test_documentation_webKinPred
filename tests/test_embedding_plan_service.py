#!/usr/bin/env python3
"""Unit tests for shared embedding planning and GPU step partitioning."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.services import embedding_plan_service as eps


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


class EmbeddingPlanServiceTests(unittest.TestCase):
    def test_kinform_mixed_state_sparse_steps_parallel_single_step(self):
        with tempfile.TemporaryDirectory(prefix="emb_plan_kinform_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            seq_ids = ["sid_1", "sid_2", "sid_3"]

            # sid_1 complete; sid_2 missing only esm2; sid_3 missing only prot_t5 + binding-site row
            kin_roots = [
                "esm2_layer_26",
                "esm2_layer_29",
                "esmc_layer_24",
                "esmc_layer_32",
                "prot_t5_layer_19",
                "prot_t5_last",
            ]
            for sid in seq_ids:
                for root in kin_roots:
                    _touch(media / "sequence_info" / root / "mean_vecs" / f"{sid}.npy")
                    _touch(media / "sequence_info" / root / "weighted_vecs" / f"{sid}.npy")

            # remove sid_2 esm2 files only
            for root in ["esm2_layer_26", "esm2_layer_29"]:
                (media / "sequence_info" / root / "mean_vecs" / "sid_2.npy").unlink()
                (media / "sequence_info" / root / "weighted_vecs" / "sid_2.npy").unlink()

            # remove sid_3 prot_t5 files only
            for root in ["prot_t5_layer_19", "prot_t5_last"]:
                (media / "sequence_info" / root / "mean_vecs" / "sid_3.npy").unlink()
                (media / "sequence_info" / root / "weighted_vecs" / "sid_3.npy").unlink()

            # binding site table includes sid_1 and sid_2 only (sid_3 missing)
            bs_path = media / "pseq2sites" / "binding_sites_all.tsv"
            bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs_path.write_text("PDB\tscore\n" "sid_1\t0.9\n" "sid_2\t0.2\n", encoding="utf-8")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=seq_ids):
                    plan = eps.build_embedding_plan(
                        method_key="KinForm-H",
                        target="kcat",
                        sequences=["SEQ1", "SEQ2", "SEQ3"],
                        env={},
                    )

            self.assertEqual(plan.profile, "kinform_full")
            self.assertTrue(plan.gpu_supported)
            # Counts are per-embedding-file, not per-sequence.
            # sid_1: 12/12 present; sid_2: 8/12 present (4 esm2 missing);
            # sid_3: 8/12 present (4 prot_t5 missing).
            self.assertEqual(plan.total, 36)
            self.assertEqual(plan.cached_already, 28)
            self.assertEqual(plan.need_computation, 8)

            self.assertEqual([step.step_key for step in plan.step_plans], ["kinform_t5_full"])
            by_step = {step.step_key: set(step.missing_seq_ids) for step in plan.step_plans}
            # Parallel mode emits a single orchestrated KinForm step.
            self.assertEqual(by_step["kinform_t5_full"], {"sid_2", "sid_3"})

    def test_kinform_legacy_sparse_steps_when_parallel_disabled(self):
        with tempfile.TemporaryDirectory(prefix="emb_plan_kinform_legacy_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            seq_ids = ["sid_1", "sid_2", "sid_3"]

            kin_roots = [
                "esm2_layer_26",
                "esm2_layer_29",
                "esmc_layer_24",
                "esmc_layer_32",
                "prot_t5_layer_19",
                "prot_t5_last",
            ]
            for sid in seq_ids:
                for root in kin_roots:
                    _touch(media / "sequence_info" / root / "mean_vecs" / f"{sid}.npy")
                    _touch(media / "sequence_info" / root / "weighted_vecs" / f"{sid}.npy")

            for root in ["esm2_layer_26", "esm2_layer_29"]:
                (media / "sequence_info" / root / "mean_vecs" / "sid_2.npy").unlink()
                (media / "sequence_info" / root / "weighted_vecs" / "sid_2.npy").unlink()

            for root in ["prot_t5_layer_19", "prot_t5_last"]:
                (media / "sequence_info" / root / "mean_vecs" / "sid_3.npy").unlink()
                (media / "sequence_info" / root / "weighted_vecs" / "sid_3.npy").unlink()

            bs_path = media / "pseq2sites" / "binding_sites_all.tsv"
            bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs_path.write_text("PDB\tscore\n" "sid_1\t0.9\n" "sid_2\t0.2\n", encoding="utf-8")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=seq_ids):
                    plan = eps.build_embedding_plan(
                        method_key="KinForm-H",
                        target="kcat",
                        sequences=["SEQ1", "SEQ2", "SEQ3"],
                        env={"KINFORM_PARALLEL_EMBED_ENABLE": "0"},
                    )

            step_keys = [s.step_key for s in plan.step_plans]
            self.assertIn("kinform_t5_full", step_keys)
            self.assertIn("kinform_esm2_layers", step_keys)
            self.assertIn("kinform_esmc_layers", step_keys)
            by_step = {step.step_key: set(step.missing_seq_ids) for step in plan.step_plans}
            self.assertEqual(by_step["kinform_t5_full"], {"sid_3"})
            self.assertEqual(by_step["kinform_esm2_layers"], {"sid_2"})
            self.assertEqual(by_step["kinform_esmc_layers"], set())

    def test_kinform_binding_site_only_missing(self):
        """seq_id with complete T5 embeddings but absent from binding sites TSV → kinform_t5_full."""
        with tempfile.TemporaryDirectory(prefix="emb_plan_kinform_bs_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            seq_ids = ["sid_a"]

            for root in ["prot_t5_layer_19", "prot_t5_last"]:
                _touch(media / "sequence_info" / root / "mean_vecs" / "sid_a.npy")
                _touch(media / "sequence_info" / root / "weighted_vecs" / "sid_a.npy")
            for root in ["esm2_layer_26", "esm2_layer_29", "esmc_layer_24", "esmc_layer_32"]:
                _touch(media / "sequence_info" / root / "mean_vecs" / "sid_a.npy")
                _touch(media / "sequence_info" / root / "weighted_vecs" / "sid_a.npy")

            # No binding sites TSV at all
            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=seq_ids):
                    plan = eps.build_embedding_plan(
                        method_key="KinForm-H",
                        target="kcat",
                        sequences=["MSEQ"],
                        env={},
                    )

            by_step = {s.step_key: set(s.missing_seq_ids) for s in plan.step_plans}
            self.assertIn("sid_a", by_step["kinform_t5_full"])

    def test_kinform_t5_weighted_only_missing(self):
        """seq_id with mean but not weighted T5 → included in kinform_t5_full."""
        with tempfile.TemporaryDirectory(prefix="emb_plan_kinform_w_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            seq_ids = ["sid_x"]

            # mean exists, weighted missing for T5
            for root in ["prot_t5_layer_19", "prot_t5_last"]:
                _touch(media / "sequence_info" / root / "mean_vecs" / "sid_x.npy")
            for root in ["esm2_layer_26", "esm2_layer_29", "esmc_layer_24", "esmc_layer_32"]:
                _touch(media / "sequence_info" / root / "mean_vecs" / "sid_x.npy")
                _touch(media / "sequence_info" / root / "weighted_vecs" / "sid_x.npy")
            bs_path = media / "pseq2sites" / "binding_sites_all.tsv"
            bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs_path.write_text("PDB\tscore\nsid_x\t0.5\n", encoding="utf-8")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=seq_ids):
                    plan = eps.build_embedding_plan(
                        method_key="KinForm-H",
                        target="kcat",
                        sequences=["MSEQ"],
                        env={},
                    )

            by_step = {s.step_key: set(s.missing_seq_ids) for s in plan.step_plans}
            self.assertIn("sid_x", by_step["kinform_t5_full"])

    def test_kinform_all_complete_no_t5_full(self):
        """All T5 + binding sites present → kinform_t5_full has no seq_ids."""
        with tempfile.TemporaryDirectory(prefix="emb_plan_kinform_full_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            seq_ids = ["sid_z"]

            for root in ["prot_t5_layer_19", "prot_t5_last",
                         "esm2_layer_26", "esm2_layer_29",
                         "esmc_layer_24", "esmc_layer_32"]:
                _touch(media / "sequence_info" / root / "mean_vecs" / "sid_z.npy")
                _touch(media / "sequence_info" / root / "weighted_vecs" / "sid_z.npy")
            bs_path = media / "pseq2sites" / "binding_sites_all.tsv"
            bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs_path.write_text("PDB\tscore\nsid_z\t0.8\n", encoding="utf-8")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=seq_ids):
                    plan = eps.build_embedding_plan(
                        method_key="KinForm-H",
                        target="kcat",
                        sequences=["MSEQ"],
                        env={},
                    )

            self.assertEqual(plan.need_computation, 0)
            self.assertEqual(eps.gpu_step_work(plan), {})

    def test_catapro_reuses_prott5_cache(self):
        with tempfile.TemporaryDirectory(prefix="emb_plan_catapro_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            seq_ids = ["a", "b"]
            for sid in seq_ids:
                _touch(media / "sequence_info" / "prot_t5_last" / "mean_vecs" / f"{sid}.npy")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=seq_ids):
                    plan = eps.build_embedding_plan(
                        method_key="CataPro",
                        target="kcat",
                        sequences=["AA", "BB"],
                        env={},
                    )

            self.assertEqual(plan.profile, "prot_t5_mean")
            self.assertEqual(plan.need_computation, 0)
            self.assertEqual(eps.gpu_step_work(plan), {})

    def test_turnup_partial_cache(self):
        with tempfile.TemporaryDirectory(prefix="emb_plan_turnup_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            _touch(media / "sequence_info" / "esm1b_turnup" / "sid_1.npy")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=["sid_1", "sid_2"]):
                    plan = eps.build_embedding_plan(
                        method_key="TurNup",
                        target="kcat",
                        sequences=["ACD", "EFG"],
                        env={},
                    )

            self.assertEqual(plan.profile, "turnup_esm1b")
            self.assertEqual(plan.need_computation, 1)
            self.assertEqual(eps.gpu_step_work(plan), {"turnup_esm1b": ["sid_2"]})

    def test_eitlem_gpu_supported_with_esm1v_cache(self):
        """EITLEM is GPU-enabled; cache paths are esm1v/ (full residue matrix)."""
        with tempfile.TemporaryDirectory(prefix="emb_plan_eitlem_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"

            # sid_1 already has a cached representation; sid_2 does not.
            _touch(media / "sequence_info" / "esm1v" / "sid_1.npy")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=["sid_1", "sid_2"]):
                    plan = eps.build_embedding_plan(
                        method_key="EITLEM",
                        target="kcat",
                        sequences=["ACD", "EFG"],
                        env={},
                    )

            self.assertTrue(plan.gpu_supported)
            self.assertIsNone(plan.gpu_reason)
            self.assertEqual(plan.profile, "eitlem_esm1v")
            self.assertEqual(plan.total, 2)           # 1 file per seq_id
            self.assertEqual(plan.cached_already, 1)
            self.assertEqual(plan.need_computation, 1)
            self.assertEqual(eps.gpu_step_work(plan), {"eitlem_esm1v": ["sid_2"]})

    def test_omniesi_gpu_supported_with_esm2_cache(self):
        """OmniESI uses its own full-residue ESM2 cache family."""
        with tempfile.TemporaryDirectory(prefix="emb_plan_omniesi_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"

            _touch(media / "sequence_info" / "omniesi_esm2" / "sid_1.pt")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=["sid_1", "sid_2"]):
                    plan = eps.build_embedding_plan(
                        method_key="OmniESI",
                        target="Km",
                        sequences=["ACD", "EFG"],
                        env={},
                    )

            self.assertTrue(plan.gpu_supported)
            self.assertEqual(plan.profile, "omniesi_esm2")
            self.assertEqual(plan.total, 2)
            self.assertEqual(plan.cached_already, 1)
            self.assertEqual(plan.need_computation, 1)
            self.assertEqual(eps.gpu_step_work(plan), {"omniesi_esm2": ["sid_2"]})

    def test_realkcat_gpu_supported_with_mean_esm2_cache(self):
        """RealKcat uses a dedicated persistent ESM2 last-layer mean cache."""
        with tempfile.TemporaryDirectory(prefix="emb_plan_realkcat_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"

            _touch(media / "sequence_info" / "esm2_layer_last_mean" / "sid_1.npy")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=["sid_1", "sid_2"]):
                    plan = eps.build_embedding_plan(
                        method_key="RealKcat",
                        target="kcat",
                        sequences=["ACD", "EFG"],
                        env={},
                    )

            self.assertTrue(plan.gpu_supported)
            self.assertEqual(plan.profile, "realkcat_esm2_last_mean")
            self.assertEqual(plan.total, 2)
            self.assertEqual(plan.cached_already, 1)
            self.assertEqual(plan.need_computation, 1)
            self.assertEqual(eps.gpu_step_work(plan), {"realkcat_esm2_last_mean": ["sid_2"]})

    def test_iecata_gpu_supported_with_residue_cache(self):
        """IECata uses ephemeral full-residue ProtT5 cache family."""
        with tempfile.TemporaryDirectory(prefix="emb_plan_iecata_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"

            _touch(media / "sequence_info" / "iecata_prot_t5_residues" / "sid_1.npy")

            with patch.object(eps, "resolve_media_and_tools", return_value=(media, tools)):
                with patch.object(eps, "resolve_seq_ids_via_cli", return_value=["sid_1", "sid_2"]):
                    plan = eps.build_embedding_plan(
                        method_key="IECata",
                        target="kcat/Km",
                        sequences=["ACD", "EFG"],
                        env={},
                    )

            self.assertTrue(plan.gpu_supported)
            self.assertEqual(plan.profile, "iecata_prot_t5_residues")
            self.assertEqual(plan.total, 2)
            self.assertEqual(plan.cached_already, 1)
            self.assertEqual(plan.need_computation, 1)
            self.assertEqual(eps.gpu_step_work(plan), {"iecata_prot_t5_residues": ["sid_2"]})


if __name__ == "__main__":
    unittest.main()
