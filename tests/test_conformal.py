import math

import numpy as np

from conformal_selective_prediction import (
    average_set_size,
    conformal_quantile,
    empirical_coverage,
    lac_prediction_sets,
    lac_scores,
)


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = np.array([0.4, 1.0, 0.2, 0.8, 0.1, 0.6, 0.9, 0.3, 0.7, 0.5])

    threshold = conformal_quantile(scores, alpha=0.15)

    assert threshold == 1.0


def test_conformal_quantile_returns_infinity_for_unavailable_rank() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    threshold = conformal_quantile(scores, alpha=0.1)

    assert math.isinf(threshold)


def test_lac_scores_use_the_true_class_probabilities() -> None:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.2, 0.5, 0.3],
        ]
    )
    labels = np.array([0, 2, 1])

    scores = lac_scores(probabilities, labels)

    expected_scores = np.array([0.3, 0.4, 0.5])
    np.testing.assert_allclose(scores, expected_scores)


def test_lac_prediction_sets_include_scores_at_the_threshold() -> None:
    probabilities = np.array(
        [
            [0.75, 0.50, 0.25],
            [0.625, 0.375, 0.00],
        ]
    )

    prediction_sets = lac_prediction_sets(probabilities, threshold=0.5)

    expected_sets = np.array(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    np.testing.assert_array_equal(prediction_sets, expected_sets)


def test_prediction_set_metrics_measure_coverage_and_size() -> None:
    prediction_sets = np.array(
        [
            [True, False, False],
            [False, True, True],
            [False, False, True],
            [True, False, True],
        ]
    )
    labels = np.array([0, 0, 2, 1])

    coverage = empirical_coverage(prediction_sets, labels)
    average_size = average_set_size(prediction_sets)

    assert coverage == 0.5
    assert average_size == 1.5
