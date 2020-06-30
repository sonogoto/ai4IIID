#!/usr/bin/env python3

from vgg16 import vgg16_feat
import tensorflow as tf


def extract_vgg_feat(x, total_layers):
    assert x.shape[1] == total_layers, \
        '''Unmatched count layers, the input tensor has %d layers, 
        which unmatched with `total_layers` with value %d''' % (x.shape[1].value, total_layers)
    # vgg_feat: total_layers * batch_size * 512
    vgg_feat = [vgg16_feat(x[:, layer_idx, :, :, :]) for layer_idx in range(total_layers)]
    # vgg_feat: batch_size * total_layers * 512
    # return tf.expand_dims(
    #     tf.transpose(vgg_feat, perm=(1, 0, 2)),
    #     axis=-1
    # )
    return tf.transpose(vgg_feat, perm=(1, 0, 2, 3))


def top(feat):
    filters_list = [32, 16]
    kernel_size_list = [3, 3]

    def _cnn_layer(feat_):
        for filters, kernel_size in zip(filters_list, kernel_size_list):
            feat_ = tf.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation=tf.nn.relu,
                kernel_initializer='glorot_normal',
                kernel_regularizer='l2'
            )(feat_)
        return feat_

    units_list = [1024, 256]

    def _fc_layer(feat_):
        feat_ = tf.layers.Flatten()(feat_)
        for units in units_list:
            feat_ = tf.layers.Dense(
                units=units,
                kernel_initializer=tf.initializers.glorot_normal,
                kernel_regularizer='l2'
            )(feat_)
        return feat_

    feat_ = _fc_layer(_cnn_layer(feat))
    return tf.layers.Dense(units=1, kernel_initializer=tf.initializers.glorot_normal)(feat_)


def model(x, total_layers=100):
    return top(extract_vgg_feat(x, total_layers))
