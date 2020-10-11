import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal

from seqeval.metrics.v1 import unique_labels, precision_recall_fscore_support, classification_report
from seqeval.scheme import IOB2


@pytest.mark.parametrize(
    'y_true, y_pred, expected',
    [
        ([[]], [[]], []),
        ([['B-PER']], [[]], ['PER']),
        ([[]], [['B-PER']], ['PER']),
        ([['B-PER']], [['B-PER']], ['PER']),
        ([['B-PER', 'O']], [[]], ['PER']),
        ([['B-PER', 'I-PER']], [[]], ['PER']),
        ([['B-PER']], [['B-ORG']], ['ORG', 'PER'])
    ]
)
def test_unique_labels(y_true, y_pred, expected):
    labels = unique_labels(y_true, y_pred, IOB2)
    assert labels == expected


class TestPrecisionRecallFscoreSupport:

    def test_bad_beta(self):
        y_true, y_pred = [[]], [[]]
        with pytest.raises(ValueError):
            precision_recall_fscore_support(y_true, y_pred, beta=-0.1, scheme=IOB2)

    def test_bad_average_option(self):
        y_true, y_pred = [[]], [[]]
        with pytest.raises(ValueError):
            precision_recall_fscore_support(y_true, y_pred, average='mega', scheme=IOB2)

    @pytest.mark.parametrize(
        'average', [None, 'macro', 'weighted']
    )
    def test_warning(self, average):
        y_true = [['B-PER']]
        y_pred = [['B-Test']]
        with pytest.warns(UndefinedMetricWarning):
            precision_recall_fscore_support(y_true, y_pred, average=average, beta=1.0, scheme=IOB2)

    def test_fscore_warning(self):
        with pytest.warns(UndefinedMetricWarning):
            precision_recall_fscore_support([[]], [[]], average='micro', scheme=IOB2, warn_for=('f-score', ))

    def test_length(self):
        y_true = [['B-PER']]
        y_pred = [['B-PER', 'O']]
        with pytest.raises(ValueError):
            precision_recall_fscore_support(y_true, y_pred, scheme=IOB2)

    def test_weighted_true_sum_zero(self):
        res = precision_recall_fscore_support([['O']], [['O']], average='weighted', scheme=IOB2)
        assert res == (0.0, 0.0, 0.0, 0)

    def test_scores(self):
        y_true = [['B-A', 'B-B', 'O', 'B-A']]
        y_pred = [['O', 'B-B', 'B-C', 'B-A']]
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, scheme=IOB2)
        assert_array_almost_equal(p, [1.00, 1.00, 0.00], 2)
        assert_array_almost_equal(r, [0.50, 1.00, 0.00], 2)
        assert_array_almost_equal(f, [0.67, 1.00, 0.00], 2)
        assert_array_equal(s, [2, 1, 0])

    @pytest.mark.parametrize(
        'average, expected',
        [
            ('micro', [0.67, 0.67, 0.67, 3]),
            ('macro', [0.67, 0.50, 0.56, 3]),
            ('weighted', [1.00, 0.67, 0.78, 3])
        ]
    )
    def test_average_scores(self, average, expected):
        y_true = [['B-A', 'B-B', 'O', 'B-A']]
        y_pred = [['O', 'B-B', 'B-C', 'B-A']]
        scores = precision_recall_fscore_support(y_true, y_pred, average=average, scheme=IOB2)
        assert_array_almost_equal(scores, expected, 2)

    @pytest.mark.parametrize(
        'average, expected',
        [
            ('micro', [0.67, 0.67, 0.67, 3]),
            ('macro', [0.67, 0.50, 0.50, 3]),
            ('weighted', [1.00, 0.67, 0.67, 3])
        ]
    )
    def test_average_scores_beta_inf(self, average, expected):
        y_true = [['B-A', 'B-B', 'O', 'B-A']]
        y_pred = [['O', 'B-B', 'B-C', 'B-A']]
        scores = precision_recall_fscore_support(y_true, y_pred, average=average, scheme=IOB2, beta=np.inf)
        assert_array_almost_equal(scores, expected, 2)


class TestClassificationReport:

    def test_output_dict(self):
        y_true = [['B-A', 'B-B', 'O', 'B-A']]
        y_pred = [['O', 'B-B', 'B-C', 'B-A']]
        report = classification_report(y_true, y_pred, output_dict=True)
        expected_report = {
            'A': {
                'f1-score': 0.6666666666666666,
                'precision': 1.0,
                'recall': 0.5,
                'support': 2
            },
            'B': {
                'f1-score': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'support': 1
            },
            'C': {
                'f1-score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'support': 0
            },
            'macro avg': {
                'f1-score': 0.5555555555555555,
                'precision': 0.6666666666666666,
                'recall': 0.5,
                'support': 3
            },
            'micro avg': {
                'f1-score': 0.6666666666666666,
                'precision': 0.6666666666666666,
                'recall': 0.6666666666666666,
                'support': 3
            },
            'weighted avg': {
                'f1-score': 0.7777777777777777,
                'precision': 1.0,
                'recall': 0.6666666666666666,
                'support': 3
            }
        }
        assert report == expected_report

    def test_output_string(self):
        y_true = [['B-A', 'B-B', 'O', 'B-A']]
        y_pred = [['O', 'B-B', 'B-C', 'B-A']]
        report = classification_report(y_true, y_pred)
        expected_report = """\
              precision    recall  f1-score   support

           A       1.00      0.50      0.67         2
           B       1.00      1.00      1.00         1
           C       0.00      0.00      0.00         0

   micro avg       0.67      0.67      0.67         3
   macro avg       0.67      0.50      0.56         3
weighted avg       1.00      0.67      0.78         3
"""
        assert report == expected_report
