"""
Evaluation test is performed for the following dataset.
https://www.clips.uantwerpen.be/conll2000/chunking/output.html
"""
import os
import random
import subprocess
import unittest

from seqeval.metrics import (f1_score, accuracy_score, classification_report,
                             precision_score, recall_score,
                             performance_measure)
from seqeval.metrics.sequence_labeling import get_entities


class TestMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_name = os.path.join(os.path.dirname(__file__), 'data/ground_truth.txt')
        cls.y_true, cls.y_pred = cls.load_labels(cls, cls.file_name)
        cls.inv_file_name = os.path.join(os.path.dirname(__file__), 'data/ground_truth_inv.txt')
        cls.y_true_inv, cls.y_pred_inv = cls.load_labels(cls, cls.inv_file_name)

    def test_get_entities(self):
        y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
        self.assertEqual(get_entities(y_true), [('MISC', 3, 5), ('PER', 7, 8)])

    def test_get_entities_with_suffix_style(self):
        y_true = ['O', 'O', 'O', 'MISC-B', 'MISC-I', 'MISC-I', 'O', 'PER-B', 'PER-I']
        self.assertEqual(get_entities(y_true, suffix=True), [('MISC', 3, 5), ('PER', 7, 8)])

    def test_get_entities_with_non_NE_input(self):
        y_true = ['O', 'O', 'O', 'MISC', 'MISC', 'MISC', 'O', 'PER', 'PER']
        with self.assertWarns(UserWarning):
            get_entities(y_true)

        with self.assertWarns(UserWarning):
            get_entities(y_true, suffix=True)

    def test_get_entities_with_only_IOB(self):
        y_true = [['O', 'O', 'O', 'B', 'I', 'I', 'O'], ['B', 'I', 'O']]
        entities = get_entities(y_true)
        self.assertEqual(entities, [('_', 3, 5), ('_', 8, 9)])

    def test_performance_measure(self):
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O', 'B-PER']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O', 'B-MISC']]
        performance_dict = performance_measure(y_true, y_pred)
        self.assertDictEqual(performance_dict, {
                             'FN': 1, 'FP': 3, 'TN': 4, 'TP': 3})

    def test_classification_report(self):
        print(classification_report(self.y_true, self.y_pred))

    def test_inv_classification_report(self):
        print(classification_report(self.y_true_inv, self.y_pred_inv, suffix=True))

    def test_by_ground_truth(self):
        with open(self.file_name) as f:
            output = subprocess.check_output(['perl', 'conlleval.pl'], stdin=f).decode('utf-8')
            acc_true, p_true, r_true, f1_true = self.parse_conlleval_output(output)

            acc_pred = accuracy_score(self.y_true, self.y_pred)
            p_pred = precision_score(self.y_true, self.y_pred)
            r_pred = recall_score(self.y_true, self.y_pred)
            f1_pred = f1_score(self.y_true, self.y_pred)

            self.assertLess(abs(acc_pred - acc_true), 1e-4)
            self.assertLess(abs(p_pred - p_true), 1e-4)
            self.assertLess(abs(r_pred - r_true), 1e-4)
            self.assertLess(abs(f1_pred - f1_true), 1e-4)

    def test_metrics_for_inv_data(self):
        with open(self.file_name) as f:
            acc_pred = accuracy_score(self.y_true, self.y_pred)
            p_pred = precision_score(self.y_true, self.y_pred)
            r_pred = recall_score(self.y_true, self.y_pred)
            f1_pred = f1_score(self.y_true, self.y_pred)

            acc_pred_inv = accuracy_score(self.y_true_inv, self.y_pred_inv)
            p_pred_inv = precision_score(self.y_true_inv, self.y_pred_inv, suffix=True)
            r_pred_inv = recall_score(self.y_true_inv, self.y_pred_inv, suffix=True)
            f1_pred_inv = f1_score(self.y_true_inv, self.y_pred_inv, suffix=True)

            self.assertLess(abs(acc_pred - acc_pred_inv), 1e-4)
            self.assertLess(abs(p_pred - p_pred_inv), 1e-4)
            self.assertLess(abs(r_pred - r_pred_inv), 1e-4)
            self.assertLess(abs(f1_pred - f1_pred_inv), 1e-4)

    def test_statistical_tests(self):
        filepath = 'eval_data.txt'
        for prefix in ['BIO', 'EIO']:
            for i in range(10000):
                print('Iteration: {}'.format(i))
                self.generate_eval_data(filepath, prefix)
                y_true, y_pred = self.load_labels(filepath)
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

    def load_labels(self, filename):
        y_true, y_pred = [], []
        with open(filename) as f:
            tags_true, tags_pred = [], []
            for line in f:
                line = line.rstrip()
                if len(line) == 0:
                    if len(tags_true) != 0:
                        y_true.append(tags_true)
                        y_pred.append(tags_pred)
                        tags_true, tags_pred = [], []
                else:
                    _, _, tag_true, tag_pred = line.split(' ')
                    tags_true.append(tag_true)
                    tags_pred.append(tag_pred)
            else:
                y_true.append(tags_true)
                y_pred.append(tags_pred)
        return y_true, y_pred

    @staticmethod
    def parse_conlleval_output(text):
        eval_line = text.split('\n')[1]
        items = eval_line.split(' ')
        accuracy, precision, recall = [item[:-2] for item in items if '%' in item]
        f1 = items[-1]

        accuracy = float(accuracy) / 100
        precision = float(precision) / 100
        recall = float(recall) / 100
        f1 = float(f1) / 100

        return accuracy, precision, recall, f1

    @staticmethod
    def generate_eval_data(filepath, prefixes='BIO'):
        types = ['PER', 'MISC', 'ORG', 'LOC']
        report = ''
        raw_fmt = '{} {} {} {}\n'
        for i in range(1000):
            type_true = random.choice(types)
            type_pred = random.choice(types)
            prefix_true = random.choice(prefixes)
            prefix_pred = random.choice(prefixes)
            true_out = 'O' if prefix_true == 'O' else '{}-{}'.format(prefix_true, type_true)
            pred_out = 'O' if prefix_pred == 'O' else '{}-{}'.format(prefix_pred, type_pred)
            report += raw_fmt.format('X', 'X', true_out, pred_out)

            # end of sentence
            if random.random() > 0.95:
                report += '\n'

        with open(filepath, 'w') as f:
            f.write(report)
