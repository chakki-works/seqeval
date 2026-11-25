import pytest

from seqeval.metrics import accuracy_score, performance_measure


@pytest.mark.parametrize('metric', [accuracy_score, performance_measure])
@pytest.mark.parametrize('y_true, y_pred', [
    (['O'], ['O', 'B-PER']),
    (['O', 'B-PER'], ['O']),
    ([['O'], []], [['O']]),
    ([['B-PER'], ['O', 'O']], [['B-PER', 'O'], ['O']]),
])
def test_inconsistent_lengths(metric, y_true, y_pred):
    with pytest.raises(ValueError, match='inconsistent numbers of samples'):
        metric(y_true, y_pred)


@pytest.mark.parametrize('y_true, y_pred', [
    (['B-PER', 'O'], ['O', 'O']),
    ([['B-PER'], [], ['O']], [['O'], [], ['O']]),
])
def test_aligned_labels(y_true, y_pred):
    assert accuracy_score(y_true, y_pred) == 0.5
    assert performance_measure(y_true, y_pred) == {'TP': 0, 'FP': 0, 'FN': 1, 'TN': 1}
