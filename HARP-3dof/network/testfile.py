import tensorflow as tf
from inputs import read_tfrecords


X,y = read_tfrecords(1,"./inputs/se2-3dof/train.tfrecords","./inputs/se2-3dof/train_labels.tfrecords",1,0)
a = y * 1

with tf.Session() as sess:
    for i in range(5):
        b = sess.run(a)
        print b.shape