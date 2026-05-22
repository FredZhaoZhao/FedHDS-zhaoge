import csv
import tempfile
import unittest
from pathlib import Path

from scripts.make_paper_table import build_table


class BuildPaperTableTests(unittest.TestCase):
    def test_build_table_exposes_mia_and_guard_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "group",
                        "hessian_mode",
                        "forget_client_loss_before_unlearning",
                        "forget_client_loss_after_unlearning",
                        "global_loss_after_unlearning",
                        "actual_param_delta_l2_norm",
                        "unlearn_steps_accepted",
                        "unlearn_num_steps",
                        "unlearning_time_sec",
                        "mia_loss_auc",
                        "mia_loss_tpr_at_fpr001",
                        "unlearn_guard_stop_reason",
                        "unlearning_method",
                        "is_anomalous",
                        "anomaly_reason",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "group": "retained_negative_guard",
                        "hessian_mode": "retained-set",
                        "forget_client_loss_before_unlearning": "2.0",
                        "forget_client_loss_after_unlearning": "2.6",
                        "global_loss_after_unlearning": "1.5",
                        "actual_param_delta_l2_norm": "0.1",
                        "unlearn_steps_accepted": "4",
                        "unlearn_num_steps": "5",
                        "unlearning_time_sec": "12.34",
                        "mia_loss_auc": "0.61",
                        "mia_loss_tpr_at_fpr001": "0.22",
                        "unlearn_guard_stop_reason": "completed",
                        "unlearning_method": "ga_guarded",
                        "is_anomalous": "True",
                        "anomaly_reason": "baseline_diverged,forget_client_too_small",
                    }
                )

            rows = build_table(csv_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["steps"], "4/5")
        self.assertEqual(rows[0]["mia_auc"], "0.6100")
        self.assertEqual(rows[0]["mia_tpr_at_fpr001"], "0.2200")
        self.assertEqual(rows[0]["method"], "ga_guarded")
        self.assertIn("anomalous seed", rows[0]["note"])


if __name__ == "__main__":
    unittest.main()
