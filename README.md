# seqeval
seqeval is a testing framework for sequence labeling.
You can evaluate named-entity recognition, part-of-speech tagging, semantic role labeling and so on.

seqeval supports following format:
* IOB
* IOBES

## How to use
Behold, the power of seqeval:

```python
>>> from seqeval.metrics import f1_score, accuracy_score, classification_report
>>> y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'O']
>>> y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER', 'O']
>>> f1_score(y_true, y_pred)
0.50
>>> accuracy_score(y_true, y_pred)
0.80
>>> classification_report(y_true, y_pred)
             precision    recall  f1-score   support

       MISC       0.00      0.00      0.00         1
        PER       1.00      1.00      1.00         1

avg / total       0.50      0.50      0.50         2
```

You can fed a nested list into the functions:

```python
>>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> f1_score(y_true, y_pred)
0.50
```

## Install
To install seqeval, simply run:

```
$ pip install seqeval
```
