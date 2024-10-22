import numpy as np
import os
import json
import math
import tensorflow as tf
from .cnn_lstm import CNN_LSTM

class ModelFactory:
    @staticmethod
    def get_model(model_type, config):
        if model_type == 'CNN_LSTM':
            return CNN_LSTM(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

