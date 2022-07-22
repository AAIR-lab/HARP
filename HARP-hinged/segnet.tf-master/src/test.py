from __future__ import division
import os, pdb, pickle
from PIL import Image
import numpy as np

from inputs import read_tfrecords
from scalar_ops import accuracy, gan_loss

import classifier
import config
import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import sys

'''
# save random seed
assumes: model trained, test env & label's tfrecords already in correct folder
actions: passes test env through model, records green pixels and their bounds in OR coordinates
'''

model_name = sys.argv[1]
envnum = sys.argv[2]
# model_name = 'unet-v26'
# envnum = '8.0.2'

tf.app.flags.DEFINE_string('ckpt_dir', './ckpts/', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('test_logs', './logs/' + model_name + '-test-' + envnum, 'Log directory')
tf.app.flags.DEFINE_integer('batch', 1, 'Batch size') # was 200

FLAGS = tf.app.flags.FLAGS

def test():

  test_file, test_labels_file = utils.get_dataset(config.working_dataset, 'test', include_labels=False)
  X, y = read_tfrecords(FLAGS.batch, test_file, test_labels_file)

  if y:
    y_gaze = tf.slice(y, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
    y_dof1 = tf.slice(y, [0,0,0,1], [FLAGS.batch, 224, 224, 1])
    y_dof2 = tf.slice(y, [0,0,0,2], [FLAGS.batch, 224, 224, 1])
    tf.summary.image('y_gaze', y_gaze, max_outputs=FLAGS.batch)
    tf.summary.image('y_dof1', y_dof1, max_outputs=FLAGS.batch)
    tf.summary.image('y_dof2', y_dof2, max_outputs=FLAGS.batch)

  model = utils.get_model(config.model, 'train')
  predictions, logits = model.inference(X, 'test', max_outputs=FLAGS.batch)

  saver = tf.train.Saver(tf.global_variables())
  summary = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(FLAGS.test_logs)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  # session_config.gpu_options.per_process_gpu_memory_fraction=config.gpu_memory_fraction

  with tf.Session(config=session_config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

    if not (ckpt and ckpt.model_checkpoint_path):
      print('No checkpoint file found')
      return

    ckpt_path = ckpt.model_checkpoint_path
    saver.restore(sess, ckpt_path)

    summary_str = sess.run(summary) 
    summary_writer.add_summary(summary_str)
    summary_writer.flush()

    coord.request_stop()
    coord.join(threads)
    
    prediction = sess.run(predictions)
    np.save('results/'+envnum+'.npy', prediction)
    logits = sess.run(logits)
    np.save('results/'+envnum+'_logits.npy', logits)
    print("completed")
    return


def main(argv=None):
  test()

if __name__ == '__main__':
  tf.app.run()
