#!/usr/bin/env python3

import sys
import os
from stl2voxel import stl2voxel
from model_vgg16 import model
import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
import pickle as pk
import math
from sklearn.model_selection import train_test_split


print(sys.version)


RESOLUTION = 100
LAYER_AXIS_IDX = 0


def fill_up_layers(voxel, layers=RESOLUTION):
    assert voxel.shape[LAYER_AXIS_IDX] <= layers,\
        'illegal layers count, input has layers count %d,' \
        ' which greater then arg `layers` with value %d'\
        % (voxel.shape[LAYER_AXIS_IDX], layers)
    return np.concatenate(
        (
            voxel,
            np.zeros(
                shape=(layers-voxel.shape[LAYER_AXIS_IDX],
                       RESOLUTION,
                       RESOLUTION),
                dtype=np.float32
            )
         ),
        axis=LAYER_AXIS_IDX
    )


def loss_func(y_out, labels):
    return tf.losses.mean_squared_error(
        labels=labels,
        predictions=y_out
    )


def get_optimize_op(loss):
    # return tf.train.FtrlOptimizer(
    #     learning_rate=.03,
    #     # l1_regularization_strength=.001,
    #     # l2_regularization_strength=.01,
    #     # l2_shrinkage_regularization_strength=.01
    # ).minimize(
    #     loss,
    #     global_step=tf.train.get_global_step()
    # )
    return tf.train.AdamOptimizer().minimize(
        loss,
        global_step=tf.train.get_global_step()
    )


def init_placeholder():
    input_x = tf.placeholder(
        shape=(None, RESOLUTION, RESOLUTION, RESOLUTION, 1),
        dtype=tf.float32
    )
    input_y = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    return input_x, input_y


def get_feeds(stl_file_path, time_cost_path, cached_file=None):
    if cached_file:
        with open(cached_file, 'rb') as fr:
            feed_x, feed_y = pk.load(fr)
    else:
        with open(time_cost_path) as fr:
            time_cost = json.load(fr)
        feed_x = []
        feed_y = []
        for fname in tqdm(os.listdir(stl_file_path), desc='parse stl'):
            if fname not in time_cost:
                # print('no hours cost info for file `%s`' % fname)
                continue
            voxel = stl2voxel(os.path.join(stl_file_path, fname))
            # print(fname + ' with shape z: %d, x: %d, y: %d' % voxel.shape)
            try:
                feed_x.append(np.expand_dims(fill_up_layers(voxel), axis=0))
            except AssertionError as e:
                # print(fname)
                # print(str(e))
                continue

            feed_y.append([math.log(time_cost[fname], 10)])
        with open('/home/duser/ai4IIID/data/feeds_norm.pkl', 'wb') as fw:
            pk.dump((feed_x, feed_y), fw)
    return np.expand_dims(np.concatenate(feed_x, axis=0), axis=-1), np.array(feed_y, dtype=np.float32)


if __name__ == '__main__':
    in_x, in_y = init_placeholder()
    y_out = model(in_x)
    loss = loss_func(y_out, in_y)
    optimize_op = get_optimize_op(loss)
    global_init_op = tf.global_variables_initializer()
    epoch = 0
    print('parsing stl file ......')
    feed_x, feed_y = get_feeds(
        stl_file_path='/home/duser/ai4IIID/data/stl',
        time_cost_path='/home/duser/data/hours.json',
        cached_file='/home/duser/ai4IIID/data/feeds_norm.pkl'
    )
    x_dev, x_valid, y_dev, y_valid = train_test_split(
        feed_x,
        feed_y,
        test_size=.2
    )
    run_options = tf.RunOptions(
        report_tensor_allocations_upon_oom=True,
    )
    with tf.Session() as sess:
        sess.run(global_init_op)
        while epoch < 500:
            loss_value, _ = sess.run(
                (loss, optimize_op),
                feed_dict={in_x: x_dev, in_y: y_dev},
                options=run_options
            )
            print('the %d-th epoch, loss[mse]: %f' % (epoch, loss_value))
            valid_loss = sess.run(
                loss,
                feed_dict={in_x: x_valid, in_y: y_valid},
                options=run_options
            )
            print('the %d-th epoch, valid set loss[mse]: %f' % (epoch, valid_loss))
            epoch += 1
        # print('labels')
        # print(10 ** feed_y)
        print('labels\t\tpredictions')
        # print(10 ** sess.run(y_out, feed_dict={in_x: feed_x}))
        print(10**np.concatenate((y_dev, sess.run(y_out, feed_dict={in_x: x_dev})), axis=1))
        print(10**np.concatenate((y_valid, sess.run(y_out, feed_dict={in_x: x_valid})), axis=1))

