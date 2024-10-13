import tensorflow as tf
from tensorflow.python.keras import Model, layers, models

class BaseModel:
    def forward(self, *inputs):
        raise NotImplementedError
