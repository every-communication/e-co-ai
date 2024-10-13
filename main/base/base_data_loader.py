import numpy as np
import os
import json
import math
import tensorflow as tf
from tensorflow.python.keras.utils.all_utils import Sequence
from tensorflow.python.data import Dataset

class BaseDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0):
        self.dataset = dataset
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        
        self.n_samples = len(dataset)
        self.train_data, self.valid_data = self._split_dataset()

        print("Base Data Loader")
        print(self.dataset)

        print(self.batch_size)
        print(self.shuffle)
        print(self.validation_split)

        print(self.n_samples)
        print(self.train_data)
        print(self.valid_data)
        print()
        print()


    def _split_dataset(self):
        if self.validation_split == 0.0:
            return self.dataset, None
        
        indices = np.arange(self.n_samples)
        np.random.seed(0)
        np.random.shuffle(indices)

        if isinstance(self.validation_split, int):
            assert self.validation_split > 0
            assert self.validation_split < self.n_samples, "Validation set size is larger than the dataset."
            len_valid = self.validation_split
        else:
            len_valid = int(self.n_samples * self.validation_split)

        valid_indices = indices[:len_valid]
        train_indices = indices[len_valid:]

        train_data = tf.data.Dataset.from_tensor_slices((train_indices, self.dataset))
        valid_data = tf.data.Dataset.from_tensor_slices((valid_indices, self.dataset))

        return train_data.batch(self.batch_size), valid_data.batch(self.batch_size) if valid_data else None

    def get_train_data(self):
        if self.shuffle:
            return self.train_data.shuffle(buffer_size=len(self.dataset))
        return self.train_data

    def get_validation_data(self):
        return self.valid_data