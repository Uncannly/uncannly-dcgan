import numpy as np

from constants import DATA_SIZE, BATCH_SIZE

NUM_EXAMPLES = 500


class DataSet(object):
    def __init__(self, words, labels):
        self._num_examples = words.shape[0]
        self._words = words
        self._labels = labels
        self._index_in_epoch = 0

    def next_batch(self):
        start = self._index_in_epoch
        self._index_in_epoch += BATCH_SIZE

        if self._index_in_epoch > self._num_examples:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)

            self._words = self._words[perm]
            self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = BATCH_SIZE

        end = self._index_in_epoch

        return self._words[start:end]


def read_data_sets():
    words = np.array([[0] * DATA_SIZE] * NUM_EXAMPLES)
    labels = np.array([0] * NUM_EXAMPLES)

    return DataSet(words, labels)
