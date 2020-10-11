# seqeval

seqeval is a Python framework for sequence labeling evaluation.
seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on.

This is well-tested by using the Perl script [conlleval](https://www.clips.uantwerpen.be/conll2002/ner/bin/conlleval.txt),
which can be used for measuring the performance of a system that has processed the CoNLL-2000 shared task data.

## Support features

seqeval supports following formats:
* IOB1
* IOB2
* IOE1
* IOE2
* IOBES

and supports following metrics:

| metrics  | description  |
|---|---|
| accuracy_score(y\_true, y\_pred)  | Compute the accuracy.  |
| precision_score(y\_true, y\_pred)  | Compute the precision.  |
| recall_score(y\_true, y\_pred)  | Compute the recall.  |
| f1_score(y\_true, y\_pred)  | Compute the F1 score, also known as balanced F-score or F-measure.  |
| classification_report(y\_true, y\_pred, digits=2)  | Build a text report showing the main classification metrics. `digits` is number of digits for formatting output floating point values. Default value is `2`. |

## Usage

Behold, the power of seqeval:

```python
>>> from seqeval.metrics import accuracy_score
>>> from seqeval.metrics import classification_report
>>> from seqeval.metrics import f1_score
>>> 
>>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>>
>>> f1_score(y_true, y_pred)
0.50
>>> accuracy_score(y_true, y_pred)
0.80
>>> classification_report(y_true, y_pred)
              precision    recall  f1-score   support

        MISC       0.00      0.00      0.00         1
         PER       1.00      1.00      1.00         1

   micro avg       0.50      0.50      0.50         2
   macro avg       0.50      0.50      0.50         2
weighted avg       0.50      0.50      0.50         2
```

If you want to explicitly specify the evaluation scheme, use `mode='strict'`:

```python
>>> from seqeval.scheme import IOB2
>>> classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
              precision    recall  f1-score   support

        MISC       0.00      0.00      0.00         1
         PER       1.00      1.00      1.00         1

   micro avg       0.50      0.50      0.50         2
   macro avg       0.50      0.50      0.50         2
weighted avg       0.50      0.50      0.50         2
```

Note: The behavior of the strict mode is different from the default one which is designed to simulate conlleval.

## Installation
To install seqeval, simply run:

```
$ pip install seqeval
```
