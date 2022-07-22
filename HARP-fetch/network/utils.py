from models import *
import config
import os
import tensorflow as tf

def get_model(model_name, ttype):
    '''
    returns an object of the model class
    '''
    models = {
        'unet': UNet,
        'unetv2':UNetV2
    }
    return models[model_name](config.output_channels, strided=config.strided, ttype=ttype)

def get_dataset(dataset_name, kind, include_labels):
    '''
    returns path to tfrecords file for corresponding dataset
    '''
    path = os.path.join('input', dataset_name)
    data_binary_path = os.path.join(path, '%s.tfrecords' % kind)
    if include_labels:
        labels_binary_path = os.path.join(path, '%s_labels.tfrecords' % kind)
        return data_binary_path, labels_binary_path
    return data_binary_path, False
