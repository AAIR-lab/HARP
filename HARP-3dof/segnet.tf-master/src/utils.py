from models import *
import config
import os
import tensorflow as tf

def get_model(model_name, ttype):
    models = {
        'autoencoder': SegNetAutoencoder,
        'cvae': CVAE,
        'cgan': cGAN,
        'pix2pix': pix2pix,
        'unet': UNet
    }
    return models[model_name](config.output_channels, strided=config.strided, ttype=ttype)

def get_dataset(dataset_name, kind, include_labels):
    path = os.path.join('input', dataset_name)
    data_binary_path = os.path.join(path, '%s.tfrecords' % kind)
    if include_labels:
        labels_binary_path = os.path.join(path, '%s_labels.tfrecords' % kind)
        return data_binary_path, labels_binary_path
    return data_binary_path, False
