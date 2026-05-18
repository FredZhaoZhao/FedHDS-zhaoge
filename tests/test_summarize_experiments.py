import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_experiments import _summarize_result


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


if __name__ == "__main__":
    unittest.main()
