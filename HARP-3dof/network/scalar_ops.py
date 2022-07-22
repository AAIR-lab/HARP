import tensorflow as tf
import config

def accuracy(logits, labels):
    '''
    calculate accuracy for SegNetAutoencoder
    round off output of sigmoid layer and count number of matching pixels with ground truth
    '''
    FLAGS = tf.app.flags.FLAGS
    logits_gaze = tf.slice(logits, [0, 0, 0, 0], [FLAGS.batch, 224, 224, 1])
    sigmoid = tf.nn.sigmoid(logits_gaze)
    predictions = tf.round(sigmoid)
    correct_prediction = tf.equal(predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def ae_loss(predictions, logits, labels, runtype):
    '''
    Calculate loss for unet
    '''
    FLAGS = tf.app.flags.FLAGS

    logits_cr = tf.slice(logits, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
    logits_dof1 = tf.slice(logits, [0,0,0,1], [FLAGS.batch, 224, 224, config.dof1_bins])
    logits_dof2 = tf.slice(logits, [0,0,0,1 + config.dof1_bins], [FLAGS.batch, 224, 224, config.dof2_bins])
    
    labels_cr = tf.slice(labels, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
    labels_dof1 = tf.cast(tf.reshape(tf.slice(labels, [0,0,0,1], [FLAGS.batch, 224, 224, 1]), [FLAGS.batch, 224, 224]), tf.int64)
    labels_dof2 = tf.cast(tf.reshape(tf.slice(labels, [0,0,0,2], [FLAGS.batch, 224, 224, 1]), [FLAGS.batch, 224, 224]), tf.int64)

    loss_cr = tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_cr, labels=labels_cr), [FLAGS.batch, 224, 224])
    tf.summary.scalar(runtype + '_loss_cr', tf.reduce_mean(loss_cr))
    loss_dof1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_dof1, labels=labels_dof1)
    tf.summary.scalar(runtype + '_loss_dof1', tf.reduce_mean(loss_dof1))
    loss_dof2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_dof2, labels=labels_dof2)
    tf.summary.scalar(runtype + '_loss_dof2', tf.reduce_mean(loss_dof2))

    total_loss = tf.reduce_mean(loss_cr + loss_dof1 + loss_dof2)
    tf.summary.scalar(runtype + '_loss', total_loss)

    correct_prediction = tf.equal(predictions, tf.cast(labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar(runtype + '_accuracy', accuracy)
    
    return total_loss

def smooth_ones_labels(logits):
    '''
    smoothing class=1 to [0.7, 1.2]
    REF: https://github.com/soumith/ganhacks
    '''
    return tf.ones_like(logits) - 0.3 + (tf.random.uniform(tf.shape(logits)) * 0.5)

def smooth_zero_labels(logits):
    '''
    smoothing class=0 to [0.0, 0.3]
    '''
	return tf.zeros_like(logits) + tf.random.uniform(tf.shape(logits)) * 0.3

def gan_loss(logits_real, logits_fake, gen_logits, y, ttype):
    '''
    calculate metrics for pix2pix network
    '''
    FLAGS = tf.app.flags.FLAGS
    lambda_1 = 100 # reconstruction loss
    lambda_2 = 10 # classification loss
    
    D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=smooth_zero_labels(logits_fake)))
    tf.summary.scalar('D_fake_loss_' + ttype, D_fake_loss)
    D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=smooth_ones_labels(logits_real)))
    tf.summary.scalar('D_real_loss' + ttype, D_real_loss)
    G_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)))
    tf.summary.scalar('G_fake_loss' + ttype, lambda_2 * G_fake_loss)

    recon_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, tf.nn.tanh(gen_logits)))
    # h = tf.keras.losses.Huber()
    # recon_loss = tf.reduce_mean(h(y, tf.nn.tanh(gen_logits)))
    # recon_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=gen_logits, labels=y, pos_weight=2))
    # recon_loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, gen_image))
    tf.summary.scalar('recon_loss' + ttype, lambda_1 * recon_loss)

    D_loss = D_fake_loss + D_real_loss
    G_loss = lambda_2 * G_fake_loss + lambda_1 * recon_loss
    tf.summary.scalar('D_loss' + ttype, D_loss)
    tf.summary.scalar('G_loss' + ttype, G_loss)

    fake_predictions = tf.round(tf.sigmoid(logits_fake))
    fake_prediction_count = tf.cast(tf.equal(fake_predictions, tf.zeros_like(fake_predictions)), tf.float32)
    real_predictions = tf.round(tf.sigmoid(logits_real))
    real_prediction_count = tf.cast(tf.equal(real_predictions, tf.ones_like(real_predictions)), tf.float32)
    accuracy = tf.reduce_mean(tf.concat([fake_prediction_count, real_prediction_count], axis=0))
    tf.summary.scalar('accuracy' + ttype, accuracy)

    return D_loss, G_loss