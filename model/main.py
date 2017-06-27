import os
import time
import argparse
import tensorflow as tf
from network import Unet_3D

"""
This file provides configuration to build U-NET for semantic segmentation.

"""


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 250000, '# of step for training')
    # flags.DEFINE_integer('test_interval', 100000000, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 1000, '# of interval to save a model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save the summary')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    # data
    flags.DEFINE_string('data_dir', '../h5_data_SM/', 'Name of data file(s)')
    flags.DEFINE_boolean('aug_flip', False, 'Training data augmentation: flip. Extra 3 datasets.')
    flags.DEFINE_boolean('aug_rotate', False, 'Training data augmentation: rotate. Extra 9 datasets.')
    flags.DEFINE_integer('validation_id', 10, '1-10, which subject is used for validation')
    flags.DEFINE_integer('patch_size', 32, 'patch size')
    flags.DEFINE_integer('overlap_stepsize', 16, 'overlap stepsize when performing testing')
    flags.DEFINE_string('data_type', '3D', '2D data or 3D data')
    flags.DEFINE_integer('batch', 5, 'training batch size')
    flags.DEFINE_integer('channel', 2, 'channel size')
    flags.DEFINE_integer('depth', 32, 'depth size') # should be equal to patch_size
    flags.DEFINE_integer('height', 32, 'height size') # should be equal to patch_size
    flags.DEFINE_integer('width', 32, 'width size') # should be equal to patch_size
    # Debug
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('savedir', './result', 'Result saving directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 150000, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('network_depth', 4, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 4, 'output class number')
    flags.DEFINE_integer('start_channel_num', 32,
                         'start number of outputs for the first conv layer')
    flags.DEFINE_string(
        'conv_name', 'conv',
        'Use which conv op in decoder: conv or ipixel_cl')
    flags.DEFINE_string(
        'deconv_name', 'deconv',
        'Use which deconv op in decoder: deconv, pixel_dcl, ipixel_dcl')
    # Dense Transformer Networks
    flags.DEFINE_boolean('add_dtn', False, 'add Dense Transformer Networks or not')
    flags.DEFINE_integer('dtn_location', 2, 'The Dense Transformer Networks location')
    flags.DEFINE_string('control_points_ratio', 2,
        'Setup the ratio of control_points comparing with the Dense transformer networks input size')    
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    else:
        model = Unet_3D(tf.Session(), configure())
        getattr(model, args.action)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    tf.app.run()
