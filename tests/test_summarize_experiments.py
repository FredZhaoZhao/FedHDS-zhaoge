import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_experiments import (
    _annotate_anomalous_rows,
    _should_skip_result_path,
    _summarize_result,
)


class SummarizeExperimentsTests(unittest.TestCase):
    def test_extracts_unlearning_method_and_mia_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "ga_guarded" / "exp" / "adam" / "final_results.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(
                    {
                        "eval_history": [1.8, 1.6],
                        "experiment_metrics": {
                            "global_loss_after_unlearning": 1.55,
                            "mia_loss_auc": 0.61,
                            "mia_loss_tpr_at_fpr001": 0.22,
                            "unlearning_method": "ga_guarded",
                        },
                        "config": {
                            "unlearn_method": "ga_guarded",
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = _summarize_result(root, result_path)

        self.assertEqual(row["group"], "ga_guarded")
        self.assertEqual(row["unlearning_method"], "ga_guarded")
        self.assertEqual(row["mia_loss_auc"], 0.61)
        self.assertEqual(row["mia_loss_tpr_at_fpr001"], 0.22)

    def test_marks_whole_seed_as_anomalous_when_baselines_diverge_early(self):
        rows = [
            {
                "group": "seed46/fl",
                "round2_global_loss": 10.0,
                "final_global_loss": 10.0,
                "mia_member_count": 112,
            },
            {
                "group": "seed46/fedhds",
                "round2_global_loss": 10.0,
                "final_global_loss": 10.0,
                "mia_member_count": 112,
            },
            {
                "group": "seed46/retrain_oracle",
                "round2_global_loss": 10.0,
                "final_global_loss": 10.0,
                "mia_member_count": 112,
            },
            {
                "group": "seed46/ga_guarded",
                "round2_global_loss": 10.0,
                "final_global_loss": 10.0,
                "mia_member_count": 112,
            },
            {
                "group": "seed45/fl",
                "round2_global_loss": 0.2,
                "final_global_loss": 0.16,
                "mia_member_count": 296,
            },
        ]

        annotated = _annotate_anomalous_rows(rows)

        anomalous_rows = [row for row in annotated if row["group"].startswith("seed46/")]
        self.assertTrue(all(str(row["is_anomalous"]).lower() == "true" for row in anomalous_rows))
        self.assertTrue(
            all("baseline_diverged" in row["anomaly_reason"] for row in anomalous_rows)
        )
        self.assertTrue(
            all("forget_client_too_small" in row["anomaly_reason"] for row in anomalous_rows)
        )
        normal_row = next(row for row in annotated if row["group"] == "seed45/fl")
        self.assertEqual(str(normal_row["is_anomalous"]).lower(), "false")
        self.assertEqual(normal_row["anomaly_reason"], "")

    def test_skips_results_under_appendix_directories(self):
        root = Path("/tmp/results")
        appendix_result = root / "appendix_seed46" / "seed46" / "fl" / "exp" / "run" / "final_results.json"
        active_result = root / "seed47" / "fl" / "exp" / "run" / "final_results.json"

        self.assertTrue(_should_skip_result_path(root, appendix_result))
        self.assertFalse(_should_skip_result_path(root, active_result))


if __name__ == "__main__":
    unittest.main()
