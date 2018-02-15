import unittest

from seqeval.metrics import f1_score, accuracy_score, classification_report


class TestMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        cls.y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']

    def test_f1_score(self):
        score = f1_score(self.y_true, self.y_pred)
        self.assertEqual(score, 0.5)

    def test_accuracy_score(self):
        score = accuracy_score(self.y_true, self.y_pred)
        self.assertEqual(score, 0.8)

    def test_classification_report(self):
        pass
