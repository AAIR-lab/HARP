import config
import tensorflow as tf
import utils

def rgb(logits):
    softmax = tf.nn.softmax(logits)
    argmax = tf.argmax(softmax, 3)
    n = colors.get_shape().as_list()[0]
    one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    one_hot_matrix = tf.reshape(one_hot, [-1, n])
    rgb_matrix = tf.matmul(one_hot_matrix, colors)
    rgb_tensor = tf.reshape(rgb_matrix, [-1, 224, 224, 3])
    return tf.cast(rgb_tensor, tf.float32)

def visualize_prediction(logits):
    sigmoid = tf.nn.sigmoid(logits)
    predictions = tf.round(sigmoid)
    rgb_matrix = tf.stack([predictions, predictions, predictions], axis=3)
    rgb_matrix = rgb_matrix * 255
    return tf.cast(rgb_matrix, tf.float32)