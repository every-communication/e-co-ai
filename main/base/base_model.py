import tensorflow as tf
from tensorflow.python.keras import layers, models

class BaseModel(tf.keras.Model):
    """
    Base class for all models
    """
    def call(self, *inputs, **kwargs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary_with_params(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.trainable_variables
        params = sum([tf.reduce_prod(var.shape) for var in model_parameters])
        print(f"Trainable parameters: {params}")
