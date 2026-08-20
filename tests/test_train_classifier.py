import unittest

import pandas as pd

from src.train_classifier import chronological_splits, load_config


class ClassifierTests(unittest.TestCase):
    def test_chronological_split(self):
        data = pd.DataFrame({"year": [2017, 2018, 2019, 2020], "drafted": [1, 0, 1, 0]})
        split = {
            "column": "year",
            "train_end": 2017,
            "validation_start": 2018,
            "validation_end": 2019,
            "test_start": 2020,
            "test_end": 2020,
        }
        parts = chronological_splits(data, split)
        self.assertEqual(len(parts["train"]), 1)
        self.assertEqual(len(parts["validation"]), 2)
        self.assertEqual(len(parts["test"]), 1)

    def test_config_rejects_pick_as_feature(self):
        from pathlib import Path
        import json
        import tempfile

        config = {
            "data_path": "data.parquet",
            "target": "drafted",
            "numerical_features": ["overall_pick"],
            "categorical_features": [],
            "split": {},
            "model": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_config_rejects_nba_outcome_as_feature(self):
        from pathlib import Path
        import json
        import tempfile

        config = {
            "data_path": "data.parquet",
            "target": "drafted",
            "numerical_features": ["nba_min_3y"],
            "categorical_features": [],
            "split": {},
            "model": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Outcome or identity"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
