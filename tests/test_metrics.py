"""
Evaluation test is performed for the following dataset.
https://www.clips.uantwerpen.be/conll2000/chunking/output.html
"""
import os
import unittest

from seqeval.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities


class TestMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        cls.y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        dir = os.path.join(os.path.dirname(__file__), 'data')
        true_file = os.path.join(dir, 'true.txt')
        pred_file = os.path.join(dir, 'pred.txt')
        cls.y_true = cls.load_data(true_file)
        cls.y_pred = cls.load_data(pred_file)
        cls.y_true_nested = cls.load_nested_data(true_file)
        cls.y_pred_nested = cls.load_nested_data(pred_file)

    @classmethod
    def load_data(cls, file):
        with open(file) as f:
            y = [line.rstrip() if line != '\n' else 'O' for line in f]

        return y

    @classmethod
    def load_nested_data(cls, file):
        outer = []
        sublist = []
        with open(file) as f:
            for line in f:
                if line == '\n':
                    outer.append(sublist)
                    sublist = []
                else:
                    sublist.append(line.rstrip())
            else:
                outer.append(sublist)

        return outer

    def test_get_entities(self):
        y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        print(get_entities(y_true))

    def test_f1_score(self):
        score = f1_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.7435)
        score = f1_score(self.y_true_nested, self.y_pred_nested)
        self.assertEqual(round(score, 4), 0.7435)

    def test_accuracy_score(self):
        score = accuracy_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.8408)
        score = accuracy_score(self.y_true_nested, self.y_pred_nested)
        self.assertEqual(round(score, 4), 0.8408)

    def test_precision_score(self):
        score = precision_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.6883)
        score = precision_score(self.y_true_nested, self.y_pred_nested)
        self.assertEqual(round(score, 4), 0.6883)

    def test_recall_score(self):
        score = recall_score(self.y_true, self.y_pred)
        self.assertEqual(round(score, 4), 0.8083)
        score = recall_score(self.y_true_nested, self.y_pred_nested)
        self.assertEqual(round(score, 4), 0.8083)

    def test_classification_report(self):
        print(classification_report(self.y_true, self.y_pred))
        print(classification_report(self.y_true_nested, self.y_pred_nested))
