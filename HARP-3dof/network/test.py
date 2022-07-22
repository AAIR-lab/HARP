from __future__ import division
import os, pdb, pickle
from PIL import Image
import numpy as np

from inputs import read_tfrecords
# from scalar_ops import accuracy, gan_loss

# import classifier
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

envnum = sys.argv[1]
# envnum = "10.2.1"
# model_name = sys.argv[2]
# envnum = 'test_env1'
model_name = 'se2-3dof-v1'

tf.app.flags.DEFINE_string('ckpt_dir', './ckpts-v1/', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('test_logs', './logs/' + model_name + '-test-' + envnum, 'Log directory')
tf.app.flags.DEFINE_integer('batch', 1, 'Batch size')

FLAGS = tf.app.flags.FLAGS

def test():

  test_file, test_labels_file = utils.get_dataset(config.working_dataset, 'test', include_labels=False)
  X, y = read_tfrecords(FLAGS.batch, test_file, test_labels_file, 1, 0)

  model = utils.get_model(config.model, 'train')
  predictions, logits = model.inference(X, 'test',1)

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
    
    # prediction = sess.run(predictions)
    prediction = sess.run(predictions)
    np.save('results/'+envnum+'.npy', prediction)
    # logits = sess.run(logits)
    # np.save('results/'+envnum+'_logits.npy', logits)
    print("completed")
    return


def main(argv=None):
  test()

if __name__ == '__main__':
  tf.app.run()
