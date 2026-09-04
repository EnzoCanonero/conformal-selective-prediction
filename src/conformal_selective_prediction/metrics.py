import numpy as np
from numpy.typing import ArrayLike


# Measure the fraction of samples selected for automation.
def automation_rate(automation_mask: ArrayLike) -> float:
    automated_samples = np.asarray(automation_mask, dtype=np.bool_)

    if automated_samples.ndim != 1:
        raise ValueError("automation_mask must be one-dimensional")

    if automated_samples.size == 0:
        raise ValueError("automation_mask must contain at least one sample")

    rate = np.mean(automated_samples)

    return float(rate)


# Measure classification error among samples selected for automation.
def automated_error_rate(
    predictions: ArrayLike,
    labels: ArrayLike,
    automation_mask: ArrayLike,
) -> float:
    predicted_labels = np.asarray(predictions)
    true_labels = np.asarray(labels)
    automated_samples = np.asarray(automation_mask, dtype=np.bool_)

    if automated_samples.ndim != 1:
        raise ValueError("automation_mask must be one-dimensional")

    if predicted_labels.ndim != 1:
        raise ValueError("predictions must be one-dimensional")

    if true_labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    number_of_samples = automated_samples.size

    if number_of_samples == 0:
        raise ValueError("automation_mask must contain at least one sample")

    if predicted_labels.size != number_of_samples:
        raise ValueError(
            "predictions and automation_mask must contain the same samples"
        )

    if true_labels.size != number_of_samples:
        raise ValueError("labels and automation_mask must contain the same samples")

    if not np.any(automated_samples):
        return float("nan")

    automated_predictions = predicted_labels[automated_samples]
    automated_labels = true_labels[automated_samples]
    incorrect_predictions = automated_predictions != automated_labels
    error_rate = np.mean(incorrect_predictions)

    return float(error_rate)


# Measure the fraction of samples whose prediction set contains the true class.
def empirical_coverage(prediction_sets: ArrayLike, labels: ArrayLike) -> float:
    included_classes = np.asarray(prediction_sets, dtype=np.bool_)
    true_labels = np.asarray(labels)

    if included_classes.ndim != 2:
        raise ValueError("prediction_sets must be two-dimensional")

    number_of_samples = included_classes.shape[0]
    number_of_classes = included_classes.shape[1]

    if number_of_samples == 0:
        raise ValueError("prediction_sets must contain at least one sample")

    if true_labels.size != number_of_samples:
        raise ValueError("prediction_sets and labels must contain the same samples")

    labels_below_range = np.any(true_labels < 0)
    labels_above_range = np.any(true_labels >= number_of_classes)

    if labels_below_range or labels_above_range:
        raise ValueError("labels must refer to existing classes")

    sample_indices = np.arange(number_of_samples)
    covered_samples = included_classes[sample_indices, true_labels]
    coverage = np.mean(covered_samples)

    return float(coverage)


# Measure the average number of classes included per prediction set.
def average_set_size(prediction_sets: ArrayLike) -> float:
    included_classes = np.asarray(prediction_sets, dtype=np.bool_)

    if included_classes.ndim != 2:
        raise ValueError("prediction_sets must be two-dimensional")

    number_of_samples = included_classes.shape[0]

    if number_of_samples == 0:
        raise ValueError("prediction_sets must contain at least one sample")

    set_sizes = np.sum(included_classes, axis=1)
    average_size = np.mean(set_sizes)

    return float(average_size)
