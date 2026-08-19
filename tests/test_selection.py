import numpy as np

from conformal_selective_prediction import singleton_mask


def test_singleton_mask_selects_only_single_class_sets() -> None:
    prediction_sets = np.array(
        [
            [False, True, False],
            [False, False, False],
            [True, False, True],
            [False, False, True],
        ]
    )

    selected_samples = singleton_mask(prediction_sets)

    expected_selection = np.array([True, False, False, True])
    np.testing.assert_array_equal(selected_samples, expected_selection)
