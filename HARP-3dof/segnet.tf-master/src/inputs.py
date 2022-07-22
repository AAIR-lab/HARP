import tensorflow as tf
import config

def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string)
        })

    image = features['image/encoded']
    image = tf.cast(tf.decode_raw(image, tf.float64), tf.float32)
    image = tf.reshape(image, [224, 224, 3 + config.num_dof])
    return image

def read_and_decode_single_example_label(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string)
        })

    image = features['image/encoded']
    image = tf.cast(tf.decode_raw(image, tf.float64), tf.float32)
    image = tf.reshape(image, [224, 224, config.label_channels])
    return image

def read_tfrecords(batch_size, inputs_filename, labels_filename=None, capacity=2000, min_after_deque=1000):
    image = read_and_decode_single_example(inputs_filename)
    if labels_filename:
        label = read_and_decode_single_example_label(labels_filename)
        images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_deque)
        return images_batch, labels_batch
    else:
        images_batch = tf.train.shuffle_batch([image], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_deque)
        return images_batch, None
