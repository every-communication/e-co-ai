import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def accuracy(output, target):
    pred = tf.argmax(output, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.float32))
    return correct.numpy() / len(target)

def top_k_acc(output, target, k=3):
    pred = tf.argsort(output, direction='DESCENDING', axis=1)[:, :k]
    correct = sum([tf.reduce_sum(tf.cast(tf.equal(pred[:, i], target), tf.float32)) for i in range(k)])
    return correct.numpy() / len(target)

def precision(output, target, average='macro'):
    pred = tf.argmax(output, axis=1)
    true_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred, target), tf.equal(target, 1)), tf.float32))
    predicted_positive = tf.reduce_sum(tf.cast(tf.equal(pred, 1), tf.float32))
    precision = true_positive / (predicted_positive + 1e-6)
    return precision.numpy()

def recall(output, target, average='macro'):
    pred = tf.argmax(output, axis=1)
    true_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred, target), tf.equal(target, 1)), tf.float32))
    actual_positive = tf.reduce_sum(tf.cast(tf.equal(target, 1), tf.float32))
    recall = true_positive / (actual_positive + 1e-6)
    return recall.numpy()

def f1_score(output, target, average='macro'):
    prec = precision(output, target, average)
    rec = recall(output, target, average)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
    return f1.numpy()

def confusion_matrix_metric(output, target, num_classes):
    pred = tf.argmax(output, axis=1)
    cm = confusion_matrix(target.numpy(), pred.numpy(), labels=np.arange(num_classes))
    return cm
