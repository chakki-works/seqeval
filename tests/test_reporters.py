import pytest
from seqeval.reporters import DictReporter, StringReporter


class TestDictReporter:

    def test_write_empty(self):
        reporter = DictReporter()
        reporter.write_blank()
        assert reporter.report_dict == {}

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
    def test_dict_reporter_output(self, rows, expected):
        reporter = DictReporter()
        for row in rows:
            reporter.write(*row)
        assert reporter.report() == expected


class TestStringReporter:

    def test_write_empty(self):
        reporter = StringReporter()
        reporter.write_blank()
        assert reporter.buffer == ['']

    def test_write_header(self):
        reporter = StringReporter()
        report = reporter.write_header()
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
        assert 'support' in report

    def test_write(self):
        reporter = StringReporter()
        reporter.write('XXX', 0, 0, 0, 0)
        assert 'XXX' in reporter.buffer[0]

    def test_report(self):
        reporter = StringReporter()
        reporter.write('XXX', 0, 0, 0, 0)
        report = reporter.report()
        assert 'XXX' in report
        assert 'precision' in report
