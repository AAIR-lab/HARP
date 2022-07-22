import tensorflow as tf
import config

def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'tensor/encoded': tf.FixedLenFeature([], tf.string)
        })

    input_ = features['tensor/encoded']
    input_ = tf.cast(tf.decode_raw(input_, tf.float32), tf.float32)
    input_ = tf.reshape(input_, config.input_shape)
    return input_

def read_and_decode_single_example_label(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'tensor/encoded': tf.FixedLenFeature([], tf.string)
        })

    label = features['tensor/encoded']
    label = tf.cast(tf.decode_raw(label, tf.float32), tf.float32)
    label = tf.reshape(label, config.label_shape)
    return label

def read_tfrecords(batch_size, inputs_filename, labels_filename=None, capacity=2000, min_after_deque=1000):
    '''
    reads the tfrecords files and returns a batches in a queue for input pipeline
    '''
    input_ = read_and_decode_single_example(inputs_filename)
    if labels_filename:
        label = read_and_decode_single_example_label(labels_filename)
        inputs_batch, labels_batch = tf.train.shuffle_batch([input_, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_deque, num_threads=4)
        return inputs_batch, labels_batch
    else:
        inputs_batch = tf.train.shuffle_batch([input_], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_deque)
        return inputs_batch, None
