# seqeval
seqeval is a testing framework for sequence labeling.

seqeval supports following representations:
* IOB
* IOBES

## How to use
Behold, the power of seqeval:

```python
>>> from seqeval.metrics import f1_score, accuracy_score, classification_report
>>> y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
>>> y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
>>> f1_score(y_true, y_pred, average='micro', format='iob')
0.50
>>> accuracy_score(y_true, y_pred, format='iob')
0.50
>>> classification_report(y_true, y_pred, format='iob')

```


## Install
To install seqeval, simply run:

```
$ pip install seqeval
```
