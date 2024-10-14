import numpy as np
import os
import json
import math
import tensorflow as tf
from tensorflow.python.keras.utils.all_utils import Sequence
from .keypoints_data_loader import KeyPointsDataLoader

class DataLoaderFactory:
    @staticmethod
    def get_data_loader(data_loader_type, config):
        if data_loader_type == 'KeyPointsDataLoader':
            return KeyPointsDataLoader(config)
        else:
            raise ValueError(f"Unknown model type: {data_loader_type}")

