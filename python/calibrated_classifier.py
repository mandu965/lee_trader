from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BinaryIsotonicCalibratedClassifier:
    """Pickle-stable binary classifier wrapper with isotonic calibration."""

    estimator: Any
    calibrator: Any

    def predict_proba(self, X):
        raw = self.estimator.predict_proba(X)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("BinaryIsotonicCalibratedClassifier expects binary predict_proba output.")
        positive = np.asarray(raw[:, 1], dtype=float)
        calibrated = np.asarray(self.calibrator.predict(positive), dtype=float)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
