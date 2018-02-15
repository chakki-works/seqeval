# namaco
***namaco*** is a library for character-based Named Entity Recognition.
namaco will especially focus on Japanese and Chinese named entity recognition.

## How to use

```python
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report

f1_score(y_true, y_pred, average='micro', format='iob')
accuracy_score(y_true, y_pred, format='iob')
classification_report(y_true, y_pred, format='iob')

```


## Install
To install seqeval, simply run:

```
$ pip install seqeval
```
