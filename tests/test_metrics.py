import os
import unittest

from seqeval.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities

class TestMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        cls.y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        cls.y_true, cls.y_pred = cls.load_data()

    @classmethod
    def load_data(cls):
        dir = os.path.join(os.path.dirname(__file__), 'data')
        true_file = os.path.join(dir, 'true.txt')
        pred_file = os.path.join(dir, 'pred.txt')

        y_true = [line.rstrip() if line != '\n' else 'O' for line in open(true_file)]
        y_pred = [line.rstrip() if line != '\n' else 'O' for line in open(pred_file)]

        return y_true, y_pred

    def test_get_entities(self):
        y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        print(get_entities(y_true))

    def test_f1_score(self):
        score = f1_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.7435)

    def test_accuracy_score(self):
        score = accuracy_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.8408)

    def test_precision_score(self):
        score = precision_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.6883)

    def test_recall_score(self):
        score = recall_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.8083)

    def test_classification_report(self):
        classification_report(self.y_true, self.y_pred)
