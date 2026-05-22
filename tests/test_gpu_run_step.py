#!/usr/bin/env python3
"""Unit tests for GPU embedding step command wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.gpu_embed_service import run_step


class GpuRunStepTests(unittest.TestCase):
    def test_builtin_choices_include_new_steps(self):
        self.assertIn("omniesi_esm2", run_step.STEP_CHOICES)
        self.assertIn("realkcat_esm2_last_mean", run_step.STEP_CHOICES)
        self.assertIn("iecata_prot_t5_residues", run_step.STEP_CHOICES)

    def test_omniesi_step_invokes_worker_with_shared_cache_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="run_step_omniesi_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            captured: list[list[str]] = []

            with patch.object(run_step, "_run", side_effect=lambda cmd, env: captured.append(cmd)):
                run_step.run_step(
                    step="omniesi_esm2",
                    seq_ids=["sid_1"],
                    repo_root=repo_root,
                    media_path=media,
                    tools_path=tools,
                    seq_id_to_seq={"sid_1": "ACDEFG"},
                    job_id="job_x",
                )

            self.assertEqual(len(captured), 1)
            cmd = captured[0]
            self.assertIn("omniesi_esm2_worker.py", " ".join(cmd))
            self.assertIn(str((media / "sequence_info" / "omniesi_esm2").resolve()), cmd)

    def test_realkcat_step_invokes_worker_with_dedicated_cache_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="run_step_realkcat_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            captured: list[list[str]] = []

            with patch.object(run_step, "_run", side_effect=lambda cmd, env: captured.append(cmd)):
                run_step.run_step(
                    step="realkcat_esm2_last_mean",
                    seq_ids=["sid_1"],
                    repo_root=repo_root,
                    media_path=media,
                    tools_path=tools,
                    seq_id_to_seq={"sid_1": "ACDEFG"},
                    job_id="job_x",
                )

            self.assertEqual(len(captured), 1)
            cmd = captured[0]
            self.assertIn("realkcat_esm2_last_mean_worker.py", " ".join(cmd))
            self.assertIn(str((media / "sequence_info" / "esm2_layer_last_mean").resolve()), cmd)

    def test_iecata_step_invokes_worker_with_residue_cache_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="run_step_iecata_") as tmp:
            tmp_path = Path(tmp)
            media = tmp_path / "media"
            tools = tmp_path / "tools"
            captured: list[list[str]] = []

            with patch.object(run_step, "_run", side_effect=lambda cmd, env: captured.append(cmd)):
                run_step.run_step(
                    step="iecata_prot_t5_residues",
                    seq_ids=["sid_1"],
                    repo_root=repo_root,
                    media_path=media,
                    tools_path=tools,
                    seq_id_to_seq={"sid_1": "ACDEFG"},
                    job_id="job_x",
                )

            self.assertEqual(len(captured), 1)
            cmd = captured[0]
            self.assertIn("iecata_prot_t5_residues_worker.py", " ".join(cmd))
            self.assertIn(str((media / "sequence_info" / "iecata_prot_t5_residues").resolve()), cmd)


if __name__ == "__main__":
    unittest.main()
