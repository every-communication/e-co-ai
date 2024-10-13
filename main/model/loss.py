import tensorflow as tf
from tensorflow.python.keras import losses

def cross_entropy_loss(output, target):
    criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
    return criterion(output, target)