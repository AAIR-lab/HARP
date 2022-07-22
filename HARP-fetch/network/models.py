# import classifier
import convnet as cnn
import tensorflow as tf
import config

class Autoencoder:
    '''
    Base Autoencoder class abstracting different layers
    '''
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

    def conv3d(self, x, channels_shape, stride, name):
        return cnn.conv3d(x, [3, 3, 3], channels_shape, stride, name, self.ttype == 'train')

    def deconv(self, x, channels_shape, name):
        return cnn.deconv(x, [3, 3], channels_shape, 1, name, self.ttype == 'train')
  
    def deconv2(self, x, channels_shape, name):
        return cnn.deconv(x, [3, 3], channels_shape, 2, name, self.ttype == 'train')
    
    def deconv3d(self, x, channels_shape, stride, name):
        return cnn.deconv3d(x, [3, 3, 3], channels_shape, stride, name, self.ttype == 'train')

    def pool(self, x):
        return cnn.max_pool(x, 2, 2)

    def pool3d(self, x):
        return cnn.max_pool3d(x, 2, 2)

    def unpool(self, x):
        return cnn.unpool(x, 2)
    
    def unpool3d(self, x):
        return cnn.unpool3d(x, 2)

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

class UNet(Autoencoder):

    def __init__(self, n, ttype, strided=False, max_images=3):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)

    def inference(self, X, runtype, max_outputs):
        FLAGS = tf.app.flags.FLAGS

        channel1_slice = tf.slice(X, [0,0,0,0], [FLAGS.batch] + config.input_shape )
        channel1_slice = tf.reshape(channel1_slice, [FLAGS.batch] + config.input_shape)
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

        logits_cr = tf.slice(deconv1_2, [0,0,0,0], [FLAGS.batch, config.input_size, config.input_size, 1])
        logits_dof1 = tf.slice(deconv1_2, [0,0,0,1], [FLAGS.batch, config.input_size, config.input_size, config.dof1_bins])
        logits_dof2 = tf.slice(deconv1_2, [0,0,0,1 + config.dof1_bins], [FLAGS.batch, config.input_size, config.input_size, config.dof2_bins])

        # Final layer activations
        activations_cr = tf.nn.sigmoid(logits_cr)
        activations_dof1 = tf.nn.softmax(logits_dof1, axis=-1)
        activations_dof2 = tf.nn.softmax(logits_dof2, axis=-1)

        # Get Predictions
        predictions_cr = tf.reshape(tf.cast(tf.round(activations_cr), tf.int64), [FLAGS.batch, config.input_size, config.input_size])
        predictions_dof1 = tf.math.argmax(activations_dof1, axis=-1)
        predictions_dof2 = tf.math.argmax(activations_dof2, axis=-1)
        predictions = tf.stack([predictions_cr, predictions_dof1, predictions_dof2], axis=3)

        tf.summary.image(runtype + '_prediction_cr', tf.reshape(tf.cast(predictions_cr, tf.float64), [FLAGS.batch, config.input_size, config.input_size, 1]), max_outputs=max_outputs)
        tf.summary.image(runtype + '_prediction_dof1', tf.reshape(tf.cast(predictions_dof1, tf.float64), [FLAGS.batch, config.input_size, config.input_size, 1]), max_outputs=max_outputs)
        tf.summary.image(runtype + '_prediction_dof2', tf.reshape(tf.cast(predictions_dof2, tf.float64), [FLAGS.batch, config.input_size, config.input_size, 1]), max_outputs=max_outputs)

        # Write activation images to summary
        tf.summary.image(runtype + '_activation_cr', activations_cr, max_outputs=max_outputs)
        for i in range(config.dof1_bins):
          dof1_channel_slice = tf.slice(activations_dof1, [0,0,0,i], [FLAGS.batch, config.input_size, config.input_size, 1])
          tf.summary.image(runtype + '_activation_dof1_channel'+str(i), dof1_channel_slice, max_outputs=max_outputs)

        for i in range(config.dof2_bins):
          dof2_channel_slice = tf.slice(activations_dof2, [0,0,0,i], [FLAGS.batch, config.input_size, config.input_size, 1])
          tf.summary.image(runtype + '_activation_dof2_channel'+str(i), dof2_channel_slice, max_outputs=max_outputs)

        return predictions, deconv1_2


class UNetV2(Autoencoder):

    def __init__(self, n, ttype, strided=False, max_images=3):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)

    def inference(self, X, runtype, max_outputs):
        FLAGS = tf.app.flags.FLAGS

        # channel1_slice = tf.slice(X, [0,0,0,0,0], [FLAGS.batch,config.input_size,config.input_size,config.input_size,3])
        # channel1_slice = tf.reshape(channel1_slice, [FLAGS.batch,config.input_size,config.input_size,config.input_size,3])
        # tf.summary.image(runtype + 'input', channel1_slice, max_outputs=max_outputs)

        # ENCODER
        # config.input_size, config.input_size, 3
        with tf.variable_scope('enc_block1', reuse=tf.compat.v1.AUTO_REUSE):
            # conv1 = self.conv(X, [config.input_channels, 64], 'enc_conv1_1')
            # conv1_2 = self.conv2(conv1, [64, 64], 'enc_conv1_2')
            
            conv1 = self.conv3d(X, [config.input_channels, 64], 1, 'enc_conv1_1')
            conv1_2 = self.conv3d(conv1, [64, 64], 2, 'enc_conv1_2')
        # 112, 112, 64
        with tf.variable_scope('enc_block2', reuse=tf.compat.v1.AUTO_REUSE):
            # conv2 = self.conv(conv1_2, [64, 128], 'enc_conv2_1')
            # conv2_2 = self.conv2(conv2, [128, 128], 'enc_conv2_2')
            
            conv2 = self.conv3d(conv1_2, [64, 128],1,'enc_conv2_1')
            conv2_2 = self.conv3d(conv2, [128, 128],2, 'enc_conv2_2')
        # 56, 56, 128
        with tf.variable_scope('enc_block3', reuse=tf.compat.v1.AUTO_REUSE):
            # conv3 = self.conv(conv2_2, [128, 256], 'enc_conv3_1')
            # conv3_2 = self.conv2(conv3, [256, 256], 'enc_conv3_2')
            
            conv3 = self.conv3d(conv2_2, [128, 256],1, 'enc_conv3_1')
            conv3_2 = self.conv3d(conv3, [256, 256],2, 'enc_conv3_2')
        # 28, 28, 256
        with tf.variable_scope('enc_block4', reuse=tf.compat.v1.AUTO_REUSE):
            # conv4 = self.conv(conv3_2, [256, 512], 'enc_conv4_1')
            # conv4_2 = self.conv2(conv4, [512, 512], 'enc_conv4_2')
            
            conv4 = self.conv3d(conv3_2, [256, 512],1, 'enc_conv4_1')
            conv4_2 = self.conv3d(conv4, [512, 512],2, 'enc_conv4_2')
        # 14, 14, 512
        # with tf.variable_scope('enc_block5', reuse=tf.compat.v1.AUTO_REUSE):
        #     conv5 = self.conv(conv4_2, [512, 1024], 'enc_conv5_1')
        #     conv5_2 = self.conv2(conv5, [1024, 1024], 'enc_conv5_2')
        # 7, 7, 1024

        # DECODER
        # 7, 7, 1024
        # with tf.variable_scope('dec_block5', reuse=tf.compat.v1.AUTO_REUSE):
        #     deconv5 = self.deconv(conv5_2, [512, 1024], 'dec_deconv5_1')
        #     deconv5_2 = self.deconv2(deconv5, [512, 512], 'dec_deconv5_2')
        #     deconv5_3 = tf.concat([deconv5_2, conv4_2], axis=3)
        # 14, 14, 1024
        with tf.variable_scope('dec_block4', reuse=tf.compat.v1.AUTO_REUSE):
            # deconv4 = self.deconv(conv4_2, [256, 512], 'dec_deconv4_1')
            # deconv4_2 = self.deconv2(deconv4, [256, 256], 'dec_deconv4_2')
            # deconv4_3 = tf.concat([deconv4_2, conv3_2], axis=3)
            
            deconv4 = self.deconv3d(conv4_2, [256, 512],1, 'dec_deconv4_1')
            deconv4_2 = self.deconv3d(deconv4, [256, 256],2, 'dec_deconv4_2')
            deconv4_3 = tf.concat([deconv4_2, conv3_2], axis=4)
        # 28, 28, 512
        with tf.variable_scope('dec_block3', reuse=tf.compat.v1.AUTO_REUSE):
            # deconv3 = self.deconv(deconv4_3, [256, 512], 'dec_deconv3_1')
            # deconv3_2 = self.deconv2(deconv3, [128, 256], 'dec_deconv3_2')
            # deconv3_3 = tf.concat([deconv3_2, conv2_2], axis=3)
            
            deconv3 = self.deconv3d(deconv4_3, [256, 512],1, 'dec_deconv3_1')
            deconv3_2 = self.deconv3d(deconv3, [128, 256],2,'dec_deconv3_2')
            deconv3_3 = tf.concat([deconv3_2, conv2_2], axis=4)
        # 56, 56, 256
        with tf.variable_scope('dec_block2', reuse=tf.compat.v1.AUTO_REUSE):
            # deconv2 = self.deconv(deconv3_3, [128, 256], 'dec_deconv2_1')
            # deconv2_2 = self.deconv2(deconv2, [64, 128], 'dec_deconv2_2')
            # deconv2_3 = tf.concat([deconv2_2, conv1_2], axis=3)
            
            deconv2 = self.deconv3d(deconv3_3,[128, 256],1, 'dec_deconv2_1')
            deconv2_2 = self.deconv3d(deconv2,[64, 128],2, 'dec_deconv2_2')
            deconv2_3 = tf.concat([deconv2_2, conv1_2], axis=4)
        # 112, 112, 128
        with tf.variable_scope('dec_block1', reuse=tf.compat.v1.AUTO_REUSE):
            # deconv1 = self.deconv(deconv2_3, [64, 128], 'dec_deconv1_1')
            # deconv1_2 = self.deconv2(deconv1, [self.output_channels, 64], 'dec_deconv1_2')
            
            deconv1 = self.deconv3d(deconv2_3, [64, 128],1, 'dec_deconv1_1')
            deconv1_2 = self.deconv3d(deconv1, [self.output_channels, 64],2, 'dec_deconv1_2')
            # dof_layer_1_inp = tf.concat([X,deconv1_2],axis = 3)
        # config.input_size, config.input_size, 11

        logits_cr = tf.slice(deconv1_2, [0,0,0,0,0], [FLAGS.batch, config.input_size, config.input_size,config.input_size, 1])
        # logits_dof1 = tf.slice(deconv1_2, [0,0,0,1], [FLAGS.batch, config.input_size,config.input_size,10]) # uncomment this line to get v2.3

        # tf.summary.image("cr_activation",logits_cr,max_outputs=2)
        
        dof_layers = {}
        logits_dir = {}
        a = self.output_channels
        b = self.output_channels - 1
        temp = deconv1_2
        for i in range(8):
            dof_layers[i] = self.conv3d(temp,[a,b],1,'dof_layer_{}'.format(i+1))
            a = b 
            b = b - 10
            logits_dir[i] = tf.slice(dof_layers[i],[0,0,0,0,0],[FLAGS.batch,config.input_size,config.input_size,config.input_size,10])
            temp = dof_layers[i]
        
        
        activations_dir = {} 
        activations_dir["cr"] = tf.cast(tf.nn.sigmoid(logits_cr),tf.float32)
        for i in range(8):
            activations_dir[i] = tf.cast(tf.nn.softmax(logits_dir[i],axis = -1),tf.float32)
        
        # Get Predictions

        activations = activations_dir["cr"]
        logits = logits_cr

        for i in range(8):
            activations = tf.cast(tf.concat([tf.reshape(activations,[FLAGS.batch,config.input_size,config.input_size,config.input_size,i*10+1]),tf.reshape(activations_dir[i],[FLAGS.batch,config.input_size,config.input_size,config.input_size,10]) ],axis = -1),tf.float32)
            logits = tf.concat([tf.reshape(logits,[FLAGS.batch,config.input_size,config.input_size,config.input_size,i*10+1]),tf.reshape(logits_dir[i],[FLAGS.batch,config.input_size,config.input_size,config.input_size,10]) ],axis = -1)

        # Write activation images to summary
        # tf.summary.image(runtype + '_activation_cr', activations_dir["cr"], max_outputs=max_outputs)
        return activations, logits

    def loss(self,logits, predictions, labels, runtype):
        '''
        Calculate loss for unet
        '''
        FLAGS = tf.app.flags.FLAGS

        logits_cr = tf.slice(logits, [0,0,0,0,0], [FLAGS.batch, config.input_size, config.input_size,config.input_size, 1])
        logits_dir = {}
        for i in range(8):
            logits_dir[i] = tf.slice(logits, [0,0,0,0,i*10+1],[FLAGS.batch,config.input_size,config.input_size,config.input_size,10])

        labels_cr = tf.slice(labels, [0,0,0,0,0], [FLAGS.batch, config.input_size, config.input_size,config.input_size, 1])
        labels_dir = {}
        for i in range(8):
            labels_dir[i] = tf.reshape(tf.cast(tf.slice(labels, [0,0,0,0,i+1],[FLAGS.batch,config.input_size,config.input_size,config.input_size,1]),tf.int32),[FLAGS.batch,config.input_size,config.input_size,config.input_size])



        loss_cr = tf.reshape(tf.nn.weighted_cross_entropy_with_logits(labels_cr,logits_cr,2.0), [FLAGS.batch, config.input_size, config.input_size,config.input_size])


        loss_dof = {}
        for i in range(8):
            loss_dof[i] = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_dir[i],labels = labels_dir[i]) 




        # reg_loss = tf.reduce_mean(tf.nn.sigmoid(logits_cr))
        tf.summary.scalar(runtype + '_loss_cr',3 * tf.reduce_mean(loss_cr))

        for i in range(8):
            tf.summary.scalar(runtype + '_loss_dof{}'.format(i), tf.reduce_mean(loss_dof[i]))

        # tf.summary.scalar(runtype + '_reg_loss',0.5 * reg_loss)


        # total_loss = tf.reduce_mean(3 * loss_cr + loss_dof1 + loss_dof2)
        total_loss = 3 * loss_cr
        for i in range(i):
            total_loss += loss_dof[i]
        total_loss = tf.reduce_mean(total_loss)
        # total_loss = tf.reduce_mean(loss_cr)
        tf.summary.scalar(runtype + '_loss', total_loss)

        # correct_prediction = tf.equal(predictions, tf.cast(labels, tf.int64))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.summary.scalar(runtype + '_accuracy', accuracy)
        
        return total_loss

class Encoder2DV3(Autoencoder):
    
    def __init__(self, n, ttype, strided=False, max_images=5):
        Autoencoder.__init__(self, n, ttype, strided=strided, max_images=max_images)

    def inference(self, X, runtype, max_outputs):
        FLAGS = tf.app.flags.FLAGS
        # 64x64x64x17 -> 64x64x64x16 -> 32x32x32x16 -> 64x64x64x16 -> 

        # ENCODER
        # 224, 224, 7
        with tf.variable_scope('enc_block1', reuse=tf.compat.v1.AUTO_REUSE):
            conv1 = self.conv2(X, [config.input_channels, 32], 'conv1_1')
            conv1_2 = self.conv2(conv1, [32, 32],'conv1_2')
        # 32, 32, 32, 32
        with tf.variable_scope('enc_block2', reuse=tf.compat.v1.AUTO_REUSE):
            conv2 = self.conv2(conv1_2, [32, 64],'conv2_1')
            conv2_2 = self.conv2(conv2, [64, 64], 'conv2_2')
        # 16, 16, 16, 64
        with tf.variable_scope('enc_block3', reuse=tf.compat.v1.AUTO_REUSE):
            conv3 = self.conv2(conv2_2, [64, 128], 'conv3_1')
            conv3_2 = self.conv2(conv3, [128, 128], 'conv3_2')
        # 8, 8, 8, 128
        with tf.variable_scope('enc_block4', reuse=tf.compat.v1.AUTO_REUSE):
            conv4 = self.conv2(conv3_2, [128, 256], 'conv4_1')
            conv4_2 = self.conv2(conv4, [256, 256], 'conv4_2')
        # 4, 4, 4, 256
        with tf.variable_scope('enc_block5', reuse=tf.compat.v1.AUTO_REUSE):
            conv5 = self.conv2(conv4_2, [256, 512], 'conv5_1')
            conv5_2 = self.conv2(conv5, [512, 512], 'conv5_2')
        # 2, 2, 2, 512
        with tf.variable_scope('enc_block6', reuse=tf.compat.v1.AUTO_REUSE):
            conv6 = self.conv2(conv5_2, [512, 1024], 'conv6_1')
            conv6_2 = self.conv2(conv6, [1024, 1024], 'conv6_2')
        # 1, 1, 1024
        # with tf.variable_scope('enc_block7', reuse=tf.compat.v1.AUTO_REUSE):
        #     conv7 = self.conv3d(conv6_2, [1024, 2048], 1, 'conv7_1')
        #     conv7_2 = self.conv3d(conv7, [2048, 2048], 2, 'conv7_2')
        
        last_layer_nodes = 1024
        FC_input = tf.reshape(conv6_2, [FLAGS.batch, last_layer_nodes])
        FC_output_shape = [FLAGS.batch] + list(config.label_shape) # [batch, 8, 10]
        with tf.variable_scope('FC_block'):
            
            FC_output_nodes = config.number_of_bins * config.number_of_bins * (config.number_of_dof-1)
            FC_W = tf.get_variable('FC_W', [last_layer_nodes, FC_output_nodes], initializer=tf.keras.initializers.glorot_normal())
            FC_b = tf.get_variable('FC_b', [FC_output_nodes], initializer=tf.constant_initializer(.1))
            
            FC_output = tf.matmul(FC_input, FC_W) + FC_b
            FC_output = tf.reshape(FC_output, FC_output_shape)
        
        predictions = tf.nn.softmax(FC_output, axis=-1)

        return predictions, FC_output

    def inference_pooling(self, X, runtype, max_outputs):
        FLAGS = tf.app.flags.FLAGS

        # ENCODER
        # 64, 64, 64, 17
        with tf.variable_scope('enc_block1', reuse=tf.compat.v1.AUTO_REUSE):
            conv1 = self.conv2(X, [config.input_channels, 32], 'conv1_1')
            # conv1_2 = self.conv3d(conv1, [32, 32], 2, 'conv1_2')
            conv1_2 = self.pool(conv1)
        # 32, 32, 32, 32
        with tf.variable_scope('enc_block2', reuse=tf.compat.v1.AUTO_REUSE):
            conv2 = self.conv2(conv1_2, [32, 64], 'conv2_1')
            # conv2_2 = self.conv3d(conv2, [64, 64], 2, 'conv2_2')
            conv2_2 = self.pool(conv2)
        # 16, 16, 16, 64
        with tf.variable_scope('enc_block3', reuse=tf.compat.v1.AUTO_REUSE):
            conv3 = self.conv2(conv2_2, [64, 128], 'conv3_1')
            # conv3_2 = self.conv3d(conv3, [128, 128], 2, 'conv3_2')
            conv3_2 = self.pool(conv3)
        # 8, 8, 8, 128
        with tf.variable_scope('enc_block4', reuse=tf.compat.v1.AUTO_REUSE):
            conv4 = self.conv2(conv3_2, [128, 256], 'conv4_1')
            # conv4_2 = self.conv3d(conv4, [256, 256], 2, 'conv4_2')
            conv4_2 = self.pool(conv4)
        # 4, 4, 4, 256
        with tf.variable_scope('enc_block5', reuse=tf.compat.v1.AUTO_REUSE):
            conv5 = self.conv2(conv4_2, [256, 512], 'conv5_1')
            # conv5_2 = self.conv3d(conv5, [512, 512], 2, 'conv5_2')
            conv5_2 = self.pool(conv5)
        # 2, 2, 2, 512
        with tf.variable_scope('enc_block6', reuse=tf.compat.v1.AUTO_REUSE):
            conv6 = self.conv2(conv5_2, [512, 1024], 'conv6_1')
            # conv6_2 = self.conv3d(conv6, [1024, 1024], 2, 'conv6_2')
            conv6_2 = self.pool(conv6)
        # 1, 1, 1, 1024
        
        last_layer_nodes = 1024
        FC_input = tf.reshape(conv6_2, [FLAGS.batch, last_layer_nodes])
        FC_output_shape = [FLAGS.batch] + list(config.label_shape) # [batch, 8, 10]
        with tf.variable_scope('FC_block'):
            
            FC_output_nodes = config.number_of_bins * config.number_of_bins * (config.number_of_dof-1)
            FC_W = tf.get_variable('FC_W', [last_layer_nodes, FC_output_nodes], initializer=tf.keras.initializers.glorot_normal())
            FC_b = tf.get_variable('FC_b', [FC_output_nodes], initializer=tf.constant_initializer(.1))
            
            FC_output = tf.matmul(FC_input, FC_W) + FC_b
            FC_output = tf.reshape(FC_output, FC_output_shape)
        
        predictions = tf.nn.softmax(FC_output, axis=-1)

        return predictions, FC_output

    def loss(self, logits, predictions, labels, runtype):
        k = tf.keras.losses.KLDivergence()
        loss = k(labels, predictions)
        # loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(labels, predictions))
        tf.summary.scalar(runtype + '_loss', loss)
        return loss


