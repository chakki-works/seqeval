"""
Evaluation test is performed for the following dataset.
https://www.clips.uantwerpen.be/conll2000/chunking/output.html
"""
import os
import random
import subprocess
import unittest

from seqeval.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities


class TestMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        cls.y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        cls.dir = os.path.join(os.path.dirname(__file__), 'data')
        true_file = os.path.join(cls.dir, 'true.txt')
        pred_file = os.path.join(cls.dir, 'pred.txt')
        cls.y_true = cls.load_data(true_file)
        cls.y_pred = cls.load_data(pred_file)
        cls.y_true_nested = cls.load_nested_data(true_file)
        cls.y_pred_nested = cls.load_nested_data(pred_file)

    @classmethod
    def load_data(cls, file):
        with open(file) as f:
            # y = [line.rstrip() if line != '\n' else 'O' for line in f]
            y = [line.rstrip() for line in f if line != '\n']

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
        self.assertEqual(get_entities(y_true), [('MISC', 3, 5), ('PER', 7, 8)])

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

    def test_compare_score(self):
        filepath = 'eval_data.txt'
        for i in range(10000):
            print('Iteration: {}'.format(i))
            y_true, y_pred = self.generate_eval_data(filepath)
            with open(filepath) as f:
                output = subprocess.check_output(['perl', 'conlleval.pl'], stdin=f).decode('utf-8')
                acc_true, p_true, r_true, f1_true = self.parse_conlleval_output(output)

                acc_pred = accuracy_score(y_true, y_pred)
                p_pred = precision_score(y_true, y_pred)
                r_pred = recall_score(y_true, y_pred)
                f1_pred = f1_score(y_true, y_pred)

                self.assertLess(abs(acc_pred - acc_true), 1e-4)
                self.assertLess(abs(p_pred - p_true), 1e-4)
                self.assertLess(abs(r_pred - r_true), 1e-4)
                self.assertLess(abs(f1_pred - f1_true), 1e-4)

        os.remove(filepath)

    def parse_conlleval_output(self, text):
        eval_line = text.split('\n')[1]
        items = eval_line.split(' ')
        accuracy, precision, recall = [item[:-2] for item in items if '%' in item]
        f1 = items[-1]
        accuracy = float(accuracy) / 100
        precision = float(precision) / 100
        recall = float(recall) / 100
        f1 = float(f1) / 100

        return accuracy, precision, recall, f1

    def generate_eval_data(self, filepath):
        types = ['PER', 'MISC', 'ORG', 'LOC']
        prefixes = ['B', 'I', 'O']
        report = ''
        raw_fmt = '{} {} {} {}\n'
        y_true, y_pred = [], []
        tmp_true, tmp_pred = [], []
        for i in range(1000):
            type_true = random.choice(types)
            type_pred = random.choice(types)
            prefix_true = random.choice(prefixes)
            prefix_pred = random.choice(prefixes)
            true_out = 'O' if prefix_true == 'O' else '{}-{}'.format(prefix_true, type_true)
            pred_out = 'O' if prefix_pred == 'O' else '{}-{}'.format(prefix_pred, type_pred)
            report += raw_fmt.format('X', 'X', true_out, pred_out)
            tmp_true.append(true_out)
            tmp_pred.append(pred_out)

            # end of sentence
            if random.random() > 0.95:
                report += '\n'
                y_true.append(tmp_true)
                y_pred.append(tmp_pred)
                tmp_true = []
                tmp_pred = []
        else:
            report += '\n'
            y_true.append(tmp_true)
            y_pred.append(tmp_pred)

        with open(filepath, 'w') as f:
            f.write(report)

        return y_true, y_pred

