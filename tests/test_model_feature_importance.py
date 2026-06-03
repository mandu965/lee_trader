from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from python.calibrated_classifier import BinaryIsotonicCalibratedClassifier
from python.model_feature_importance_report import build_markdown_report
from python.model_train import export_feature_importance_reports, time_series_folds


class _DummyBooster:
    def __init__(self, gain_values: list[float]) -> None:
        self._gain_values = np.asarray(gain_values, dtype=float)

    def feature_importance(self, importance_type: str = "split") -> np.ndarray:
        if importance_type == "gain":
            return self._gain_values
        raise ValueError(f"unsupported importance_type={importance_type}")


class _DummyModel:
    def __init__(self, split_values: list[float], gain_values: list[float]) -> None:
        self.feature_importances_ = np.asarray(split_values, dtype=float)
        self.booster_ = _DummyBooster(gain_values)


class _DummyCalibratedModel:
    def __init__(self, estimator: object) -> None:
        self.estimator = estimator


class _DummyProbEstimator:
    def predict_proba(self, X):
        rows = len(X)
        positive = np.linspace(0.1, 0.9, rows)
        return np.column_stack([1.0 - positive, positive])


class _DummyCalibrator:
    def predict(self, values):
        return np.clip(np.asarray(values) * 0.5 + 0.25, 0.0, 1.0)


class ModelFeatureImportanceTests(unittest.TestCase):
    def test_binary_isotonic_calibrated_classifier_predict_proba(self) -> None:
        wrapper = BinaryIsotonicCalibratedClassifier(
            estimator=_DummyProbEstimator(),
            calibrator=_DummyCalibrator(),
        )
        probs = wrapper.predict_proba(pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual(probs.shape, (3, 2))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue((probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all())

    def test_time_series_folds_preserve_datetime64_for_isin_masks(self) -> None:
        dates = pd.date_range("2026-01-01", periods=18, freq="B").to_numpy()
        folds = time_series_folds(dates, 3)
        self.assertTrue(folds)
        tr_dates, va_dates = folds[-1]
        tr_mask = np.isin(dates, tr_dates)
        va_mask = np.isin(dates, va_dates)
        self.assertGreater(int(tr_mask.sum()), 0)
        self.assertGreater(int(va_mask.sum()), 0)

    def test_export_feature_importance_reports_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            feature_cols = ["f1", "f2", "f3"]
            reg_models = {
                "target_log_30d": _DummyModel([5, 3, 1], [50, 30, 10]),
            }
            cls_models = {
                "target_60d_top20": _DummyCalibratedModel(_DummyModel([2, 4, 1], [20, 40, 5])),
            }

            written = export_feature_importance_reports(
                reg_models=reg_models,
                cls_models=cls_models,
                feature_cols=feature_cols,
                output_dir=output_dir,
                model_version="test_v1",
                trained_at="2026-05-22T00:00:00Z",
                train_end_date="2026-05-21",
            )

            self.assertTrue(any(path.name == "feature_importance_summary.csv" for path in written))
            summary = pd.read_csv(output_dir / "feature_importance_summary.csv")
            combined = pd.read_csv(output_dir / "feature_importance_all_targets.csv")

            self.assertIn("composite_score", summary.columns)
            self.assertEqual(set(combined["target"]), {"target_log_30d", "target_60d_top20"})
            self.assertEqual(combined.iloc[0]["model_version"], "test_v1")

    def test_build_markdown_report_renders_targets_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "feature": "f2",
                        "target_count": 2,
                        "mean_split_pct": 0.45,
                        "mean_gain_pct": 0.50,
                        "total_split": 9.0,
                        "total_gain": 90.0,
                        "composite_score": 0.475,
                    }
                ]
            ).to_csv(output_dir / "feature_importance_summary.csv", index=False, encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "target": "target_log_30d",
                        "model_group": "regression",
                        "feature": "f2",
                        "importance_split": 5,
                        "importance_gain": 50,
                        "model_version": "test_v1",
                        "trained_at": "2026-05-22T00:00:00Z",
                        "train_end_date": "2026-05-21",
                        "importance_split_pct": 0.5,
                        "importance_gain_pct": 0.5,
                        "rank": 1,
                    }
                ]
            ).to_csv(output_dir / "feature_importance_all_targets.csv", index=False, encoding="utf-8")

            rendered = build_markdown_report(output_dir, top_n=10)

            self.assertIn("# Model Feature Importance Report", rendered)
            self.assertIn("## Overall Top Features", rendered)
            self.assertIn("## target_log_30d", rendered)
            self.assertIn("f2", rendered)


if __name__ == "__main__":
    unittest.main()
