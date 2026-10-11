import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.clustering_core import common_dates, fit_kmeans, standardize_features


class ClusteringTests(unittest.TestCase):
    def test_trailing_window_uses_latest_shared_dates(self):
        dates = common_dates(
            [20240826, 20240822, 20240823, 20240821, 20240822],
            [20240823, 20240826, 20240822, 20240820],
        )
        np.testing.assert_array_equal(dates[-2:], [20240823, 20240826])

    def test_disjoint_dates_do_not_use_an_unshared_day(self):
        self.assertEqual(common_dates([20240820], [20240821]).size, 0)

    def test_scaling_preserves_etf_identity_and_handles_constant_feature(self):
        features = pd.DataFrame(
            {"return": [1., 2., 3.], "constant": [7., 7., 7.]},
            index=["ETF-C", "ETF-A", "ETF-B"],
        )
        original = features.copy(deep=True)
        scaled = standardize_features(features)
        self.assertTrue(scaled.index.equals(features.index))
        self.assertTrue(scaled.columns.equals(features.columns))
        np.testing.assert_allclose(scaled["return"], [-np.sqrt(1.5), 0, np.sqrt(1.5)])
        np.testing.assert_array_equal(scaled["constant"], 0)
        assert_frame_equal(features, original)

    def test_kmeans_recovers_separated_groups_without_relying_on_label_numbers(self):
        features = pd.DataFrame({"x": [-10., -9., -11., 9., 10., 11.],
                                 "y": [-10., -11., -9., 11., 9., 10.]})
        scaled = standardize_features(features)
        model = fit_kmeans(scaled, 2)
        labels = model.labels_
        self.assertTrue(np.all(labels[:3] == labels[0]))
        self.assertTrue(np.all(labels[3:] == labels[3]))
        self.assertNotEqual(labels[0], labels[3])
        np.testing.assert_array_equal(labels, fit_kmeans(scaled, 2).labels_)
        np.testing.assert_array_equal(model.predict(scaled), labels)

    def test_kmeans_rejects_more_clusters_than_samples(self):
        with self.assertRaises(ValueError):
            fit_kmeans([[0., 0.], [1., 1.]], 3)


if __name__ == "__main__":
    unittest.main()
