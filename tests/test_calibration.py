import math

import numpy as np

from conformal_selective_prediction import conformal_quantile


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = np.array([0.4, 1.0, 0.2, 0.8, 0.1, 0.6, 0.9, 0.3, 0.7, 0.5])

    threshold = conformal_quantile(scores, alpha=0.15)

    assert threshold == 1.0


def test_conformal_quantile_returns_infinity_for_unavailable_rank() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    threshold = conformal_quantile(scores, alpha=0.1)

    assert math.isinf(threshold)
