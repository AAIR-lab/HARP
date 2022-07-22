import classifier
import convnet as cnn
import tensorflow as tf
import config

class Autoencoder:
    def __init__(self, n, ttype, strided=False, max_images=3):
        self.max_images = max_images
        self.n = n
        self.output_channels = n
        self.strided = strided
        self.ttype = ttype

    def conv(self, x, channels_shape, name):
        return cnn.conv(x, [3, 3], channels_shape, 1, name, self.ttype == 'train')

    def conv2(self, x, channels_shape, name):
        return cnn.conv(x, [3, 3], channels_shape, 2, name, self.ttype == 'train')

    def deconv(self, x, channels_shape, name):
        return cnn.deconv(x, [3, 3], channels_shape, 1, name, self.ttype == 'train')
  
    def deconv2(self, x, channels_shape, name):
        return cnn.deconv(x, [3, 3], channels_shape, 2, name, self.ttype == 'train')

    def pool(self, x):
        return cnn.max_pool(x, 2, 2)

    def unpool(self, x):
        return cnn.unpool(x, 2)

    def resize_conv(self, x, channels_shape, name):
        shape = x.get_shape().as_list()
        height = shape[1] * 2
        width = shape[2] * 2
        resized = tf.image.resize_nearest_neighbor(x, [height, width])
        return cnn.conv(resized, [3, 3], channels_shape, 1, name, repad=True)

    def dropout(self, x):
        return cnn.dropout(x, ttype='train', keep_prob=0.5)

    # def inference(self, images):
    #     if self.strided:
    #         return self.strided_inference(images)
    #     return self.inference_with_pooling(images)

class SegNetAutoencoder(Autoencoder):
    def __init__(self, n, ttype, strided=False, max_images=3):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)

    def inference(self, X, runtype, max_outputs):
        FLAGS = tf.app.flags.FLAGS

        env_image = tf.slice(X, [0,0,0,0], [FLAGS.batch, 224, 224, 3])
        env_image = tf.reshape(env_image, (FLAGS.batch, 224, 224, 3))
        tf.summary.image('input_'+runtype, env_image, max_outputs=max_outputs)

        with tf.variable_scope('pool1', reuse=tf.compat.v1.AUTO_REUSE):
            conv1 = self.conv(X, [7, 64], 'conv1_1')
            conv2 = self.conv(conv1, [64, 64], 'conv1_2')
            pool1 = self.pool(conv2)

        with tf.variable_scope('pool2', reuse=tf.compat.v1.AUTO_REUSE):
            conv3 = self.conv(pool1, [64, 128], 'conv2_1')
            conv4 = self.conv(conv3, [128, 128], 'conv2_2')
            pool2 = self.pool(conv4)

        with tf.variable_scope('pool3', reuse=tf.compat.v1.AUTO_REUSE):
            conv5 = self.conv(pool2, [128, 256], 'conv3_1')
            conv6 = self.conv(conv5, [256, 256], 'conv3_2')
            conv7 = self.conv(conv6, [256, 256], 'conv3_3')
            pool3 = self.pool(conv7)

        with tf.variable_scope('unpool3', reuse=tf.compat.v1.AUTO_REUSE):
            unpool3 = self.unpool(pool3)
            deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
            deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
            deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

        with tf.variable_scope('unpool4', reuse=tf.compat.v1.AUTO_REUSE):
            unpool4 = self.unpool(deconv9)
            deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
            deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

        with tf.variable_scope('unpool5', reuse=tf.compat.v1.AUTO_REUSE):
            unpool5 = self.unpool(deconv11)
            deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
            deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

        logits_cr = tf.slice(deconv13, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
        logits_dof1 = tf.slice(deconv13, [0,0,0,1], [FLAGS.batch, 224, 224, config.dof1_bins])
        logits_dof2 = tf.slice(deconv13, [0,0,0,1 + config.dof1_bins], [FLAGS.batch, 224, 224, config.dof2_bins])

        predictions_cr = tf.round(tf.nn.sigmoid(logits_cr))
        predictions_dof1 = tf.nn.softmax(logits_dof1)
        predictions_dof2 = tf.nn.softmax(logits_dof2)

        tf.summary.image('predicted_cr_'+runtype, predictions_cr, max_outputs=max_outputs)
    
        for i in range(config.dof1_bins):
            channel_slice = tf.slice(predictions_dof1, [0,0,0,i], [FLAGS.batch, 224, 224, 1])
            tf.summary.image('predicted_dof1_channel_'+str(i)+'_'+runtype, channel_slice, max_outputs=max_outputs)

        for i in range(config.dof2_bins):
            channel_slice = tf.slice(predictions_dof2, [0,0,0,i], [FLAGS.batch, 224, 224, 1])
            tf.summary.image('predicted_dof2_channel_'+str(i)+'_'+runtype, channel_slice, max_outputs=max_outputs)

        return deconv13

class CVAE(Autoencoder):
    def __init__(self, n, ttype, strided=False, max_images=3):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2.0) * eps
  
    def inference_with_pooling(self, X):
        FLAGS = tf.app.flags.FLAGS

        channel1_slice = tf.slice(X, [0,0,0,0], [FLAGS.batch, 224, 224, 3])
        channel1_slice = tf.reshape(channel1_slice, (FLAGS.batch, 224, 224, 3))
        tf.summary.image('input', channel1_slice, max_outputs=FLAGS.batch)

        with tf.variable_scope('pool1'):
            conv1 = self.conv(X, [config.input_channels, 64], 'conv1_1')
            conv2 = self.conv(conv1, [64, 64], 'conv1_2')
            pool1 = self.pool(conv2)

        with tf.variable_scope('pool2'):
            conv3 = self.conv(pool1, [64, 128], 'conv2_1')
            conv4 = self.conv(conv3, [128, 128], 'conv2_2')
            pool2 = self.pool(conv4)

        with tf.variable_scope('pool3'):
            conv5 = self.conv(pool2, [128, 256], 'conv3_1')
            conv6 = self.conv(conv5, [256, 256], 'conv3_2')
            conv7 = self.conv(conv6, [256, 256], 'conv3_3')
            pool3 = self.pool(conv7) 
        # pool3 = (batch, 28, 28, 256)

        with tf.variable_scope('pool4'):
            conv8 = self.conv(pool3, [256, 256], 'conv4_1')
            conv9 = self.conv(conv8, [256, 256], 'conv4_2')
            conv10 = self.conv(conv9, [256, 256], 'conv4_3')
            pool4 = self.pool(conv10) 
        # pool4 = (batch, 14, 14, 256)
    
        z_image_size = 14
        z_image_channels = 256
        z_dim = 500

        with tf.variable_scope('FC_1'):

            zmean_W = tf.get_variable('z_mean_W', [z_image_size * z_image_size * z_image_channels, z_dim], initializer=tf.keras.initializers.glorot_normal())
            zmean_b = tf.get_variable('z_mean_b', [z_dim], initializer=tf.constant_initializer(.1))
            
            zsigma_W = tf.get_variable('z_sigma_W', [z_image_size * z_image_size * z_image_channels, z_dim], initializer=tf.keras.initializers.glorot_normal())
            zsigma_b = tf.get_variable('z_sigma_b', [z_dim], initializer=tf.constant_initializer(.1))
            
            x = tf.reshape(pool4, [FLAGS.batch, z_image_size * z_image_size * z_image_channels])
            zmean = tf.matmul(x, zmean_W) + zmean_b
            zsigma = tf.matmul(x, zsigma_W) + zsigma_b
            z = self.sample_z(zmean, zsigma)

        with tf.variable_scope('FC_2'):
            y = tf.slice(X, begin=[0, 0, 0, 3], size=[FLAGS.batch, 1, 1, config.num_dof])
            y = tf.reshape(y, [FLAGS.batch, config.num_dof])
            tf.summary.text('goal', tf.as_string(y))
            z = tf.concat([z, y], axis=1)

            h_W = tf.get_variable('h_W', [z_dim + config.num_dof, z_image_size * z_image_size * z_image_channels], initializer=tf.keras.initializers.glorot_normal())
            h_b = tf.get_variable('h_b', [z_image_size * z_image_size * z_image_channels], initializer=tf.constant_initializer(.1))    

            h = tf.matmul(z, h_W) + h_b
            h = tf.reshape(h, [FLAGS.batch, z_image_size, z_image_size, z_image_channels])

        # h = (batch, 14, 14, 256)
        with tf.variable_scope('unpool2'):
            unpool2 = self.unpool(h)
            deconv4 = self.deconv(unpool2, [256, 256], 'deconv4_3')
            deconv5 = self.deconv(deconv4, [256, 256], 'deconv4_2')
            deconv6 = self.deconv(deconv5, [256, 256], 'deconv4_1')

        with tf.variable_scope('unpool3'):
            unpool3 = self.unpool(deconv6)
            deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
            deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
            deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

        with tf.variable_scope('unpool4'):
            unpool4 = self.unpool(deconv9)
            deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
            deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

        with tf.variable_scope('unpool5'):
            unpool5 = self.unpool(deconv11)
            deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
            deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

        # Write channels as output images
        channel1_slice = tf.slice(deconv13, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
        channel1_slice = tf.reshape(channel1_slice, (FLAGS.batch, 224, 224))
        channel1_image = classifier.visualize_prediction(channel1_slice)
        tf.summary.image('prediction_cr', channel1_image, max_outputs=FLAGS.batch)

        dof_slice = tf.slice(deconv13, [0,0,0,1], [FLAGS.batch, 224, 224, 2])
        # tf.summary.tensor_summary('prediction_dof', dof_slice)

        return deconv13, zmean, zsigma

class cGAN(Autoencoder):
    def __init__(self, n, ttype, strided=False, max_images=3):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)

    def sample_latent(self, shape):
        eps = tf.random_normal(shape=shape)
        return eps
  
    def build_gan(self, inputs, labels):
        FLAGS = tf.app.flags.FLAGS
        
        fake_labels, fake_logits = self.generator(inputs)

        D_logits_real = self.discriminator(labels, inputs, False)
        D_logits_fake = self.discriminator(fake_labels, inputs, True)

        return D_logits_real, D_logits_fake, fake_logits


    def generator(self, X):
        FLAGS = tf.app.flags.FLAGS

        channel1_slice = tf.slice(X, [0,0,0,0], [FLAGS.batch, 224, 224, 3])
        channel1_slice = tf.reshape(channel1_slice, (FLAGS.batch, 224, 224, 3))
        tf.summary.image('input', channel1_slice, max_outputs=FLAGS.batch)

        image_size = 224
        image_channels = 3
        
        cond_fc1_nodes = 28*28*1
        # cond_fc2_nodes = 28*28*128
        cond_input_shape = [FLAGS.batch, 28, 28, 1]

        latent_fc_nodes = 28*28*128
        latent_fc_output_shape = [FLAGS.batch, 28, 28, 128]
        latent_dim = 1000

        decoder_input_channels = cond_input_shape[-1] + latent_fc_output_shape[-1]

        with tf.variable_scope('gen_cond_FC1'):

            cond_W1 = tf.get_variable('gen_cond_W1', [image_size * image_size * image_channels, cond_fc1_nodes], initializer=tf.keras.initializers.glorot_normal())
            cond_b1 = tf.get_variable('gen_cond_b1', [cond_fc1_nodes], initializer=tf.constant_initializer(.1))
            
            X = tf.reshape(X, [FLAGS.batch, image_size * image_size * image_channels])
            cond_fc1_output = tf.matmul(X, cond_W1) + cond_b1
    
        cond_input = tf.reshape(cond_fc1_output, cond_input_shape)
        
        latent_sample = self.sample_latent([FLAGS.batch, latent_dim])
        with tf.variable_scope('gen_latent_FC'):

            latent_W = tf.get_variable('gen_latent_W', [latent_dim, latent_fc_nodes], initializer=tf.keras.initializers.glorot_normal())
            latent_b = tf.get_variable('gen_latent_b', [latent_fc_nodes], initializer=tf.constant_initializer(.1))
            
            latent_input = tf.matmul(latent_sample, latent_W) + latent_b
            latent_input = tf.reshape(latent_input, latent_fc_output_shape)

        gen_input = tf.concat([latent_input, cond_input], axis=3)

        with tf.variable_scope('gen_unpool3'):
            unpool3 = self.unpool(gen_input)
            deconv7 = self.deconv(unpool3, [256, decoder_input_channels], 'gen_deconv3_3')
            deconv8 = self.deconv(deconv7, [256, 256], 'gen_deconv3_2')
            deconv9 = self.deconv(deconv8, [128, 256], 'gen_deconv3_1')

        with tf.variable_scope('gen_unpool4'):
            unpool4 = self.unpool(deconv9)
            deconv10 = self.deconv(unpool4, [128, 128], 'gen_deconv2_2')
            deconv11 = self.deconv(deconv10, [64, 128], 'gen_deconv2_1')

        with tf.variable_scope('gen_unpool5'):
            unpool5 = self.unpool(deconv11)
            deconv12 = self.deconv(unpool5, [64, 64], 'gen_deconv1_2')
            deconv13 = self.deconv(deconv12, [1, 64], 'gen_deconv1_1')

        # Write channels as output images
        channel1_slice = tf.slice(deconv13, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
        channel1_slice = tf.reshape(channel1_slice, (FLAGS.batch, 224, 224))
        channel1_image = classifier.visualize_prediction(channel1_slice)
        tf.summary.image('generated_cr', channel1_image, max_outputs=FLAGS.batch)

        # dof1_slice = tf.slice(deconv13, [0,0,0,1], [FLAGS.batch, 224, 224, 1])
        # tf.summary.image('generated_dof1', tf.nn.sigmoid(dof1_slice), max_outputs=FLAGS.batch)

        # dof2_slice = tf.slice(deconv13, [0,0,0,2], [FLAGS.batch, 224, 224, 1])
        # tf.summary.image('generated_dof2', tf.nn.sigmoid(dof2_slice), max_outputs=FLAGS.batch)

        return tf.nn.sigmoid(deconv13), deconv13

    def discriminator(self, X, y, reuse):
        FLAGS = tf.app.flags.FLAGS

        input_tensor = tf.concat([X, y], axis=3)

        with tf.variable_scope('dis_pool1', reuse=reuse):
            conv1 = self.conv(input_tensor, [4, 64], 'dis_conv1_1')
            conv2 = self.conv(conv1, [64, 64], 'dis_conv1_2')
            pool1 = self.pool(conv2)

        with tf.variable_scope('dis_pool2', reuse=reuse):
            conv3 = self.conv(pool1, [64, 128], 'dis_conv2_1')
            conv4 = self.conv(conv3, [128, 128], 'dis_conv2_2')
            pool2 = self.pool(conv4)

        with tf.variable_scope('dis_pool3', reuse=reuse):
            conv5 = self.conv(pool2, [128, 256], 'dis_conv3_1')
            conv6 = self.conv(conv5, [256, 256], 'dis_conv3_2')
            conv7 = self.conv(conv6, [256, 256], 'dis_conv3_3')
            pool3 = self.pool(conv7) 
        # pool3 = (batch, 28, 28, 256)
    
        flattened_ = tf.reshape(pool3, [FLAGS.batch, -1])
        disc_fc_nodes = 28 * 28 * 256
        
        with tf.variable_scope('dis_FC', reuse=reuse):

            output_W = tf.get_variable('dis_output_W', [disc_fc_nodes, 1], initializer=tf.keras.initializers.glorot_normal())
            output_b = tf.get_variable('dis_output_b', [1], initializer=tf.constant_initializer(.1))
            
            logits = tf.matmul(flattened_, output_W) + output_b
            return logits

class pix2pix(Autoencoder):
    def __init__(self, n, ttype, strided=False, max_images=3):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)
  
    def build_gan(self, inputs, labels, runtype):
        FLAGS = tf.app.flags.FLAGS
        
        gen_image, gen_logits = self.generator(inputs, runtype)

        D_logits_real = self.discriminator(labels, inputs)
        D_logits_fake = self.discriminator(gen_image, inputs)

        return D_logits_real, D_logits_fake, gen_logits, gen_image

    def generator(self, X,runtype):
        FLAGS = tf.app.flags.FLAGS

        channel1_slice = tf.slice(X, [0,0,0,0], [FLAGS.batch, 224, 224, 3])
        channel1_slice = tf.reshape(channel1_slice, (FLAGS.batch, 224, 224, 3))
        tf.summary.image('input_'+runtype, channel1_slice, max_outputs=FLAGS.batch)
        
        # 224, 224, 3
        with tf.variable_scope('gen_pool1', reuse=tf.compat.v1.AUTO_REUSE):
            conv1 = self.conv(X, [7, 64], 'gen_conv1_1')
            conv1_2 = self.conv2(conv1, [64, 64], 'gen_conv1_2')
            # pool1 = self.pool(conv1)
        # 112, 112, 64
        with tf.variable_scope('gen_pool2', reuse=tf.compat.v1.AUTO_REUSE):
            conv2 = self.conv(conv1_2, [64, 128], 'gen_conv2_1')
            conv2_2 = self.conv2(conv2, [128, 128], 'gen_conv2_2')
            # pool2 = self.pool(conv2)
        # 56, 56, 128
        with tf.variable_scope('gen_pool3', reuse=tf.compat.v1.AUTO_REUSE):
            conv3 = self.conv(conv2_2, [128, 256], 'gen_conv3_1')
            conv3_2 = self.conv2(conv3, [256, 256], 'gen_conv3_2')
            # conv3 = self.dropout(conv3)
            # pool3 = self.pool(conv3)
        # 28, 28, 256
        with tf.variable_scope('gen_pool4', reuse=tf.compat.v1.AUTO_REUSE):
            conv4 = self.conv(conv3_2, [256, 512], 'gen_conv4_1')
            conv4_2 = self.conv2(conv4, [512, 512], 'gen_conv4_2')
            # conv4 = self.dropout(conv4)
            # pool4 = self.pool(conv4)
        # 14, 14, 512
        with tf.variable_scope('gen_pool5', reuse=tf.compat.v1.AUTO_REUSE):
            conv5 = self.conv(conv4_2, [512, 1024], 'gen_conv5_1')
            conv5_2 = self.conv2(conv5, [1024, 1024], 'gen_conv5_2')
            # conv5 = self.dropout(conv5)
            # pool5 = self.pool(conv5)
        # 7, 7, 1024
    
        with tf.variable_scope('gen_unpool5', reuse=tf.compat.v1.AUTO_REUSE):
            # unpool5 = self.unpool(pool5)
            deconv5 = self.deconv(conv5_2, [512, 1024], 'gen_deconv5_1')
            deconv5_2 = self.deconv2(deconv5, [512, 512], 'gen_deconv5_2')
            # deconv5_2 = self.dropout(deconv5_2)
            deconv5_3 = tf.concat([deconv5_2, conv4_2], axis=3)
        # 14, 14, 1024
        with tf.variable_scope('gen_unpool4', reuse=tf.compat.v1.AUTO_REUSE):
            # unpool4 = self.unpool(deconv5)
            deconv4 = self.deconv(deconv5_3, [512, 1024], 'gen_deconv4_1')
            deconv4_2 = self.deconv2(deconv4, [256, 512], 'gen_deconv4_2')
            # deconv4 = self.dropout(deconv4)
            deconv4_3 = tf.concat([deconv4_2, conv3_2], axis=3)
        # 28, 28, 512
        with tf.variable_scope('gen_unpool3', reuse=tf.compat.v1.AUTO_REUSE):
            # unpool3 = self.unpool(deconv4)
            deconv3 = self.deconv(deconv4_3, [256, 512], 'gen_deconv3_1')
            deconv3_2 = self.deconv2(deconv3, [128, 256], 'gen_deconv3_2')
            # deconv3 = self.dropout(deconv3)
            deconv3_3 = tf.concat([deconv3_2, conv2_2], axis=3)
        # 56, 56, 256
        with tf.variable_scope('gen_unpool2', reuse=tf.compat.v1.AUTO_REUSE):
            # unpool2 = self.unpool(deconv3)
            deconv2 = self.deconv(deconv3_3, [128, 256], 'gen_deconv2_1')
            deconv2_2 = self.deconv2(deconv2, [64, 128], 'gen_deconv2_2')
            deconv2_3 = tf.concat([deconv2_2, conv1_2], axis=3)
        # 112, 112, 128
        with tf.variable_scope('gen_unpool1', reuse=tf.compat.v1.AUTO_REUSE):
            # unpool1 = self.unpool(deconv2)
            deconv1 = self.deconv(deconv2_3, [64, 128], 'gen_deconv1_1')
            deconv1_3 = self.deconv2(deconv1, [3, 64], 'gen_deconv1_2')
        # 224, 224, 3

        prediction_cr = tf.slice(deconv1_3, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
        prediction_cr = tf.nn.tanh(prediction_cr)

        prediction_dof1 = tf.slice(deconv1_3, [0,0,0,1], [FLAGS.batch, 224, 224, 1])
        prediction_dof1 = tf.nn.tanh(prediction_dof1)
        # prediction_dof1 = tf.math.multiply(prediction_cr, prediction_dof1)

        prediction_dof2 = tf.slice(deconv1_3, [0,0,0,2], [FLAGS.batch, 224, 224, 1])
        prediction_dof2 = tf.nn.tanh(prediction_dof2)
        # prediction_dof2 = tf.math.multiply(prediction_cr, prediction_dof2)

        generated_image = tf.concat([prediction_cr, prediction_dof1, prediction_dof2], axis=3)

        # Write channels as output images
        # channel1_slice = tf.slice(deconv1, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
        # channel1_slice = tf.reshape(channel1_slice, (FLAGS.batch, 224, 224))
        # channel1_image = classifier.visualize_prediction(channel1_slice)
        tf.summary.image('generated_cr_'+runtype, prediction_cr, max_outputs=FLAGS.batch)

        # dof1_slice = tf.slice(deconv1, [0,0,0,1], [FLAGS.batch, 224, 224, 1])
        tf.summary.image('generated_dof1_'+runtype, prediction_dof1, max_outputs=FLAGS.batch)

        # dof2_slice = tf.slice(deconv1, [0,0,0,2], [FLAGS.batch, 224, 224, 1])
        tf.summary.image('generated_dof2_'+runtype, prediction_dof2, max_outputs=FLAGS.batch)

        return generated_image, deconv1_3

    def discriminator(self, X, y):
        FLAGS = tf.app.flags.FLAGS

        input_tensor = tf.concat([X, y], axis=3)

        with tf.variable_scope('dis_pool1', reuse=tf.compat.v1.AUTO_REUSE):
            conv1 = self.conv(input_tensor, [10, 64], 'dis_conv1')
            conv1_2 = self.conv2(conv1, [64, 64], 'dis_conv1_2')
            # pool1 = self.pool(conv1)
        # 112, 112, 64
        with tf.variable_scope('dis_pool2', reuse=tf.compat.v1.AUTO_REUSE):
            conv2 = self.conv(conv1_2, [64, 128], 'dis_conv2')
            conv2_2 = self.conv2(conv2, [128, 128], 'dis_conv2_2')
            # pool2 = self.pool(conv2)
        # 56, 56, 128
        with tf.variable_scope('dis_pool3', reuse=tf.compat.v1.AUTO_REUSE):
            conv3 = self.conv(conv2_2, [128, 256], 'dis_conv3')
            conv3_2 = self.conv2(conv3, [256, 256], 'dis_conv3_2')
            # pool3 = self.pool(conv3)
        # 28, 28, 256
        with tf.variable_scope('dis_pool4', reuse=tf.compat.v1.AUTO_REUSE):
            conv4 = self.conv(conv3_2, [256, 512], 'dis_conv4')
            conv4_2 = self.conv2(conv4, [512, 512], 'dis_conv4_2')
            # pool4 = self.pool(conv4)
        # 14, 14, 512
        with tf.variable_scope('dis_pool5', reuse=tf.compat.v1.AUTO_REUSE):
            conv5 = self.conv(conv4_2, [512, 1], 'dis_conv5')
            # pool4 = self.pool(conv4) 
        # 14, 14, 1
        return conv5

        # flattened_ = tf.reshape(conv4, [FLAGS.batch, -1])
        # disc_fc_nodes = 14 * 14 * 1
        
        # with tf.variable_scope('dis_FC', reuse=reuse):

        #   output_W = tf.get_variable('dis_output_W', [disc_fc_nodes, 1], initializer=tf.keras.initializers.glorot_normal())
        #   output_b = tf.get_variable('dis_output_b', [1], initializer=tf.constant_initializer(.1))
          
        #   logits = tf.matmul(flattened_, output_W) + output_b
        #   return logits

class UNet(Autoencoder):

    def __init__(self, n, ttype, strided=False, max_images=3):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)

    def inference(self, X, runtype, max_outputs):
        FLAGS = tf.app.flags.FLAGS

        channel1_slice = tf.slice(X, [0,0,0,0], [FLAGS.batch, 224, 224, 3])
        channel1_slice = tf.reshape(channel1_slice, (FLAGS.batch, 224, 224, 3))
        tf.summary.image(runtype + 'input', channel1_slice, max_outputs=max_outputs)

        # ENCODER
        # 224, 224, 3
        with tf.variable_scope('enc_block1', reuse=tf.compat.v1.AUTO_REUSE):
            conv1 = self.conv(X, [7, 64], 'enc_conv1_1')
            conv1_2 = self.conv2(conv1, [64, 64], 'enc_conv1_2')
        # 112, 112, 64
        with tf.variable_scope('enc_block2', reuse=tf.compat.v1.AUTO_REUSE):
            conv2 = self.conv(conv1_2, [64, 128], 'enc_conv2_1')
            conv2_2 = self.conv2(conv2, [128, 128], 'enc_conv2_2')
        # 56, 56, 128
        with tf.variable_scope('enc_block3', reuse=tf.compat.v1.AUTO_REUSE):
            conv3 = self.conv(conv2_2, [128, 256], 'enc_conv3_1')
            conv3_2 = self.conv2(conv3, [256, 256], 'enc_conv3_2')
        # 28, 28, 256
        with tf.variable_scope('enc_block4', reuse=tf.compat.v1.AUTO_REUSE):
            conv4 = self.conv(conv3_2, [256, 512], 'enc_conv4_1')
            conv4_2 = self.conv2(conv4, [512, 512], 'enc_conv4_2')
        # 14, 14, 512
        with tf.variable_scope('enc_block5', reuse=tf.compat.v1.AUTO_REUSE):
            conv5 = self.conv(conv4_2, [512, 1024], 'enc_conv5_1')
            conv5_2 = self.conv2(conv5, [1024, 1024], 'enc_conv5_2')
        # 7, 7, 1024

        # DECODER
        # 7, 7, 1024
        with tf.variable_scope('dec_block5', reuse=tf.compat.v1.AUTO_REUSE):
            deconv5 = self.deconv(conv5_2, [512, 1024], 'dec_deconv5_1')
            deconv5_2 = self.deconv2(deconv5, [512, 512], 'dec_deconv5_2')
            deconv5_3 = tf.concat([deconv5_2, conv4_2], axis=3)
        # 14, 14, 1024
        with tf.variable_scope('dec_block4', reuse=tf.compat.v1.AUTO_REUSE):
            deconv4 = self.deconv(deconv5_3, [512, 1024], 'dec_deconv4_1')
            deconv4_2 = self.deconv2(deconv4, [256, 512], 'dec_deconv4_2')
            deconv4_3 = tf.concat([deconv4_2, conv3_2], axis=3)
        # 28, 28, 512
        with tf.variable_scope('dec_block3', reuse=tf.compat.v1.AUTO_REUSE):
            deconv3 = self.deconv(deconv4_3, [256, 512], 'dec_deconv3_1')
            deconv3_2 = self.deconv2(deconv3, [128, 256], 'dec_deconv3_2')
            deconv3_3 = tf.concat([deconv3_2, conv2_2], axis=3)
        # 56, 56, 256
        with tf.variable_scope('dec_block2', reuse=tf.compat.v1.AUTO_REUSE):
            deconv2 = self.deconv(deconv3_3, [128, 256], 'dec_deconv2_1')
            deconv2_2 = self.deconv2(deconv2, [64, 128], 'dec_deconv2_2')
            deconv2_3 = tf.concat([deconv2_2, conv1_2], axis=3)
        # 112, 112, 128
        with tf.variable_scope('dec_block1', reuse=tf.compat.v1.AUTO_REUSE):
            deconv1 = self.deconv(deconv2_3, [64, 128], 'dec_deconv1_1')
            deconv1_2 = self.deconv2(deconv1, [self.output_channels, 64], 'dec_deconv1_2')
        # 224, 224, 3

        logits_cr = tf.slice(deconv1_2, [0,0,0,0], [FLAGS.batch, 224, 224, 1])
        logits_dof1 = tf.slice(deconv1_2, [0,0,0,1], [FLAGS.batch, 224, 224, config.dof1_bins])
        logits_dof2 = tf.slice(deconv1_2, [0,0,0,1 + config.dof1_bins], [FLAGS.batch, 224, 224, config.dof2_bins])

        # Final layer activations
        activations_cr = tf.nn.sigmoid(logits_cr)
        activations_dof1 = tf.nn.softmax(logits_dof1, axis=-1)
        activations_dof2 = tf.nn.softmax(logits_dof2, axis=-1)

        # Get Predictions
        predictions_cr = tf.reshape(tf.cast(tf.round(activations_cr), tf.int64), [FLAGS.batch, 224, 224])
        predictions_dof1 = tf.math.argmax(activations_dof1, axis=-1)
        predictions_dof2 = tf.math.argmax(activations_dof2, axis=-1)
        predictions = tf.stack([predictions_cr, predictions_dof1, predictions_dof2], axis=3)

        tf.summary.image(runtype + '_prediction_cr', tf.reshape(tf.cast(predictions_cr, tf.float64), [FLAGS.batch, 224, 224, 1]), max_outputs=max_outputs)
        tf.summary.image(runtype + '_prediction_dof1', tf.reshape(tf.cast(predictions_dof1, tf.float64), [FLAGS.batch, 224, 224, 1]), max_outputs=max_outputs)
        tf.summary.image(runtype + '_prediction_dof2', tf.reshape(tf.cast(predictions_dof2, tf.float64), [FLAGS.batch, 224, 224, 1]), max_outputs=max_outputs)

        # Write activation images to summary
        tf.summary.image(runtype + '_activation_cr', activations_cr, max_outputs=max_outputs)
        for i in range(config.dof1_bins):
          dof1_channel_slice = tf.slice(activations_dof1, [0,0,0,i], [FLAGS.batch, 224, 224, 1])
          tf.summary.image(runtype + '_activation_dof1_channel'+str(i), dof1_channel_slice, max_outputs=max_outputs)

        for i in range(config.dof2_bins):
          dof2_channel_slice = tf.slice(activations_dof2, [0,0,0,i], [FLAGS.batch, 224, 224, 1])
          tf.summary.image(runtype + '_activation_dof2_channel'+str(i), dof2_channel_slice, max_outputs=max_outputs)

        return predictions, deconv1_2
