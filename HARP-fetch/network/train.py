from inputs import read_tfrecords
# from scalar_ops import ae_loss
from tqdm import tqdm
import config
import tensorflow as tf
import utils
import time
import sys
import numpy as np

# model_name = sys.argv[1]
model_name = 'fetch-v2'

tf.app.flags.DEFINE_string('ckpt_dir', './ckpts-v2', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('ckpts_path', './ckpts-v2/model.ckpt', 'Train checkpoint file path')
tf.app.flags.DEFINE_integer('ckpt_step', 1000, 'Train model checkpoint step')
tf.app.flags.DEFINE_string('train_logs', './logs/' + model_name, 'Log directory')

tf.app.flags.DEFINE_integer('summary_step', 10, 'Number of iterations before serializing log data')
tf.app.flags.DEFINE_integer('train_steps', 50000, 'Number of training iterations')
tf.app.flags.DEFINE_integer('batch', 1, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 1e-04, 'learning rate for optimizer')

FLAGS = tf.app.flags.FLAGS

def train():

    train_file, train_labels_file = utils.get_dataset(config.working_dataset, 'train', include_labels=True)
    # val_file, val_labels_file = utils.get_dataset(config.working_dataset, 'test', include_labels=False)

    X, y = read_tfrecords(FLAGS.batch, train_file, train_labels_file, FLAGS.batch, 0)
    # X_val, y_val = read_tfrecords(FLAGS.batch, val_file, val_labels_file, 7, 0)

    model = utils.get_model(config.model, 'train')
    predictions, logits = model.inference(X, 'train', max_outputs=FLAGS.batch)
    loss = model.loss(logits, predictions, y, 'train')

    # predictions_val, logits_val = model.inference(X_val, 'val', max_outputs=6)
    # if y_val:
    #     loss_val = ae_loss(predictions_val, logits_val, y_val, 'val')

    total_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    t_vars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # train_step = optimizer.minimize(loss, var_list=t_vars)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=t_vars)
    gradients_list = []
    for g, v in grads_and_vars:
        # histograms for weights and grads
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + '_grad', g)
    train_step = optimizer.apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.per_process_gpu_memory_fraction=config.gpu_memory_fraction

    with tf.Session(config=session_config) as sess:
        print("Total parameters: {total_params}".format(total_params=total_params))
        # Load checkpoint if exists
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
        if not ckpt:
            print('No checkpoint file found. Initializing...')
            sess.run(init)
        else:
            ckpt_path = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt_path)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_logs, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Run training iterations
        for step in tqdm(range(FLAGS.train_steps + 1)):
            sess.run(train_step)

            # Update tensorboard summary
            if step % FLAGS.summary_step == 0:
                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save checkpoint at step
            if step % FLAGS.ckpt_step == 0:
                saver.save(sess, FLAGS.ckpts_path)

        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    start = time.time()
    train()
    seconds = time.time() - start
    print("Training completed in {hrs}hrs {mins}mins".format(hrs=seconds/3600, mins=seconds/60))

if __name__ == '__main__':
    tf.app.run()