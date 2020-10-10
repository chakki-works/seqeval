import pytest

from seqeval.metrics.v1 import unique_labels
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
