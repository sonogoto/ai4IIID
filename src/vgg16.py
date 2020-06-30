#!/usr/bin/env python3

from tensorflow.python.keras import layers
import tensorflow as tf

dropout_rate = .2


class VGG16(object):
    # Block 1
    block1_conv1 = layers.Conv2D(64, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block1_conv1')
    block1_conv2 = layers.Conv2D(64, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block1_conv2')
    block1_dropout = layers.Dropout(rate=dropout_rate)
    block1_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    # Block 2
    block2_conv1 = layers.Conv2D(128, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block2_conv1')
    block2_conv2 = layers.Conv2D(128, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block2_conv2')
    block2_dropout = layers.Dropout(rate=dropout_rate)
    block2_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    # Block 3
    block3_conv1 = layers.Conv2D(256, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block3_conv1')
    block3_conv2 = layers.Conv2D(256, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block3_conv2')
    block3_conv3 = layers.Conv2D(256, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block3_conv3')
    block3_dropout = layers.Dropout(rate=dropout_rate)
    block3_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    # Block 4
    block4_conv1 = layers.Conv2D(512, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block4_conv1')
    block4_conv2 = layers.Conv2D(512, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block4_conv2')
    block4_conv3 = layers.Conv2D(512, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block4_conv3')
    block4_dropout = layers.Dropout(rate=dropout_rate)
    block4_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

    # Block 5
    block5_conv1 = layers.Conv2D(512, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block5_conv1')
    block5_conv2 = layers.Conv2D(512, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block5_conv2')
    block5_conv3 = layers.Conv2D(512, (3, 3),
                                 activation='relu',
                                 padding='same',
                                 name='block5_conv3')
    block5_dropout = layers.Dropout(rate=dropout_rate)
    block5_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

    layers = [
        block1_conv1,
        block1_conv2,
        block1_dropout,
        block1_pool,
        block2_conv1,
        block2_conv2,
        block2_dropout,
        block2_pool,
        block3_conv1,
        block3_conv2,
        block3_conv3,
        block3_dropout,
        block3_pool,
        block4_conv1,
        block4_conv2,
        block4_conv3,
        block4_dropout,
        block4_pool,
        block5_conv1,
        block5_conv2,
        block5_conv3,
        block5_dropout,
        block5_pool,
    ]


def vgg16_feat(x, pooling='max'):
    with tf.device('/device:GPU:0'):
        feat = x
        for layer in VGG16.layers:
            feat = layer(x)
        if pooling == 'max':
            return tf.reduce_max(feat, axis=2)
        elif pooling == 'avg':
            return tf.reduce_mean(feat, axis=2)
        else:
            return feat
