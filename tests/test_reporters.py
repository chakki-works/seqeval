import pytest
from seqeval.reporters import DictReporter


@pytest.mark.parametrize(
    'rows, expected',
    [
        ([], {}),
        (
            [['PERSON', 0.82, 0.79, 0.81, 24]],
            {
                'PERSON': {
                    'precision': 0.82,
                    'recall': 0.79,
                    'f1-score': 0.81,
                    'support': 24
                }
            }
        )
    ]
)
def test_dict_reporter_output(rows, expected):
    reporter = DictReporter()
    for row in rows:
        reporter.write(*row)
    assert reporter.report() == expected
