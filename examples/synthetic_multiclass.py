from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from conformal_selective_prediction import (
    automated_error_rate,
    automation_rate,
    average_set_size,
    conformal_quantile,
    empirical_coverage,
    lac_prediction_sets,
    lac_scores,
    singleton_mask,
)


RANDOM_SEED = 42
MIS_COVERAGE_RATE = 0.1


# Run a synthetic IID dataset through the complete LAC workflow.
def run_experiment() -> tuple[float, float, float, float]:
    features, labels = make_classification(
        n_samples=8_000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=0.05,
        random_state=RANDOM_SEED,
    )

    (
        training_features,
        remaining_features,
        training_labels,
        remaining_labels,
    ) = train_test_split(
        features,
        labels,
        test_size=0.5,
        random_state=RANDOM_SEED,
    )

    (
        calibration_features,
        test_features,
        calibration_labels,
        test_labels,
    ) = train_test_split(
        remaining_features,
        remaining_labels,
        test_size=0.5,
        random_state=RANDOM_SEED,
    )

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500),
    )
    classifier.fit(training_features, training_labels)

    calibration_probabilities = classifier.predict_proba(calibration_features)
    calibration_scores = lac_scores(
        calibration_probabilities,
        calibration_labels,
    )
    threshold = conformal_quantile(calibration_scores, MIS_COVERAGE_RATE)

    test_probabilities = classifier.predict_proba(test_features)
    prediction_sets = lac_prediction_sets(test_probabilities, threshold)

    coverage = empirical_coverage(prediction_sets, test_labels)
    mean_set_size = average_set_size(prediction_sets)

    predicted_class_indices = test_probabilities.argmax(axis=1)
    test_predictions = classifier.classes_[predicted_class_indices]
    automation_mask = singleton_mask(prediction_sets)

    automated_fraction = automation_rate(automation_mask)
    automated_error = automated_error_rate(
        test_predictions,
        test_labels,
        automation_mask,
    )

    return coverage, mean_set_size, automated_fraction, automated_error


def main() -> None:
    target_coverage = 1.0 - MIS_COVERAGE_RATE
    (
        coverage,
        mean_set_size,
        automated_fraction,
        automated_error,
    ) = run_experiment()

    print(f"Target coverage: {target_coverage:.3f}")
    print(f"Empirical coverage: {coverage:.3f}")
    print(f"Average set size: {mean_set_size:.3f}")
    print(f"Automation rate: {automated_fraction:.3f}")
    print(f"Automated-case error rate: {automated_error:.3f}")


if __name__ == "__main__":
    main()
