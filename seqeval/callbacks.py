import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report


class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None, digits=4, outputs=None):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
            digits (int or None): number of digits in printed classification report
              (use None to print only F1 score without a report).
            outputs (list or None): list of output name or layers to score
              (None means all outputs)
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.digits = digits
        self.is_fit = validation_data is None
        self.outputs = outputs

    def convert_idx_to_name(self, y, array_indexes):
        """Convert label index to name.

        Args:
            y (np.ndarray): label index 2d array.
            array_indexes (list): list of valid index arrays for each row.

        Returns:
            y: label name list.
        """
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, X, y):
        """Predict sequences.

        Args:
            X (np.ndarray): input data.
            y (list): tags.

        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred_all = self.model.predict_on_batch(X)
        if not isinstance(y_pred_all, list):
            y_pred_all = [y_pred_all]

        y_true_labels = []
        y_pred_labels = []
        # reduce dimension.
        for i in self.output_indices:
            y_true = np.argmax(y[i], -1)
            y_pred = np.argmax(y_pred_all[i], -1)

            non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0]
                               for y_true_row in y_true]

            y_true_labels.append(self.convert_idx_to_name(y_true, non_pad_indexes))
            y_pred_labels.append(self.convert_idx_to_name(y_pred, non_pad_indexes))

        return y_true_labels, y_pred_labels

    def score(self, y_true, y_pred):
        """Calculate f1 score.

        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.

        Returns:
            score: f1 score.
        """
        scores = {}
        reports = []
        only_one = len(self.model.outputs) == 1
        for (output_index, output_y_true, output_y_pred) in zip(self.output_indices, y_true, y_pred):
            score = f1_score(output_y_true, output_y_pred)
            if only_one:
                title = ""
                name = "f1"
            else:
                title = self.model.output_names[output_index]
                name = "f1_{}".format(title)
            scores[name] = score
            print(' - {}: {:04.2f}'.format(name, score * 100), end='')
            if self.digits:
                reports.append(classification_report(output_y_true, output_y_pred,
                                                     digits=self.digits, title=title))
        print("\n")
        for report in reports:
            print(report)
        return scores

    def on_train_begin(self, logs=None):
        if self.outputs:
            output_indices = []
            for i, output in enumerate(self.model.outputs):
                if output in self.outputs or i in self.outputs:
                    output_indices.append(i)
            self.output_indices = output_indices
        else:
            self.output_indices = range(len(self.model.outputs))

    def on_epoch_end(self, epoch, logs=None):
        if self.is_fit:
            X = self.validation_data[0]
            y = self.validation_data[1:]
            y_true, y_pred = self.predict(X, y)
        else:
            y_true = []
            y_pred = []
            for X, y in self.validation_data:
                y_true_batch, y_pred_batch = self.predict(X, y)
                y_true.extend(y_true_batch)
                y_pred.extend(y_pred_batch)
        scores = self.score(y_true, y_pred)
        logs.update(scores)
