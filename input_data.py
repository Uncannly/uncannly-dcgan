import numpy as np

from constants import DATA_SIZE

class DataSet(object):
    def __init__(self, words, labels):
        self._num_examples = words.shape[0]
        self._words = words.reshape(words.shape[0], words.shape[1] * words.shape[2])
        self._labels = labels
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)

            self._words = self._words[perm]
            self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch

        return self._words[start:end]

def read_data_sets():
    should_be_arbitrary_data_set_length = 500
    train_words = np.array([[[[0]] * DATA_SIZE]] * should_be_arbitrary_data_set_length)
    train_labels = np.array([0] * should_be_arbitrary_data_set_length)

    return DataSet(train_words, train_labels)

