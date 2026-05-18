import random
import subprocess
import sys
import unittest
from pathlib import Path

from cli_config import build_arg_parser
from scripts.mia_loss_based import (
    parse_saved_config,
    roc_auc_from_scores,
    resolve_forget_client_idx,
    sample_category_matched_nonmembers,
    tpr_at_fpr,
)


class MiaLossBasedHelpersTests(unittest.TestCase):
    def test_script_entrypoint_help_works_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/mia_loss_based.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Evaluate loss-based MIA", result.stdout)

    def test_roc_auc_and_tpr_metrics(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]

        self.assertAlmostEqual(roc_auc_from_scores(labels, scores), 1.0, places=6)
        self.assertAlmostEqual(tpr_at_fpr(labels, scores, 0.5), 1.0, places=6)

    def test_category_matched_sampling_prefers_member_histogram(self):
        member_categories = [0, 0, 1]
        candidate_categories = [0, 0, 0, 1, 1, 2]

        selected = sample_category_matched_nonmembers(
            member_categories=member_categories,
            candidate_categories=candidate_categories,
            sample_size=3,
            rng=random.Random(0),
        )

        self.assertEqual(len(selected), 3)
        selected_categories = [candidate_categories[idx] for idx in selected]
        self.assertCountEqual(selected_categories, [0, 0, 1])

    def test_saved_config_parser_restores_typed_values(self):
        parser = build_arg_parser()
        args = parse_saved_config(
            {
                "use_fedhds_unlearn": "True",
                "unlearn_method": "ga_guarded",
                "unlearn_eta": "0.01",
                "forget_client_idx": "0",
                "amp_dtype": "torch.bfloat16",
            },
            parser,
        )

        self.assertTrue(args.use_fedhds_unlearn)
        self.assertEqual(args.unlearn_method, "ga_guarded")
        self.assertAlmostEqual(args.unlearn_eta, 0.01, places=8)
        self.assertEqual(args.forget_client_idx, 0)
        self.assertEqual(args.amp_dtype, "bf16")

    def test_resolve_forget_client_idx_prefers_recorded_metrics_for_retrain_runs(self):
        parser = build_arg_parser()
        args = parse_saved_config(
            {
                "forget_client_idx": "-1",
                "retrain_exclude_client": "-1",
            },
            parser,
        )

        idx = resolve_forget_client_idx(
            args,
            {
                "experiment_metrics": {
                    "forget_client_idx": 3,
                    "retrain_exclude_client_idx": 2,
                }
            },
        )

        self.assertEqual(idx, 3)


if __name__ == "__main__":
    unittest.main()
