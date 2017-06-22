#!/usr/bin/python
# -*- coding:utf-8 -*-

import random
import time
import os
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})
import tensorflow as tf
import h5py
import DATA
import matplotlib
matplotlib.use('Agg')  # 脱离X window使用
import matplotlib.pyplot as plt


def weight(shape, name=None):
    mlp = tf.get_variable(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    return mlp


def bias(shape, name=None):
    mlp = tf.get_variable(name, shape, initializer=tf.zeros_initializer(shape))
    return mlp


def conv(mlp, c_out, name=None):
    c_in = mlp.get_shape()[4].value
    w = weight([3, 5, 5, c_in, c_out], name=name + 'w')
    mlp = tf.nn.conv3d(mlp, w, [1, 1, 1, 1, 1], 'VALID')
    return mlp


def fc(mlp, c_out, name=None):
    c_in = mlp.get_shape()[1].value
    w = weight([c_in, c_out], name=name + 'w')
    b = weight([c_out], name=name + 'b')
    mlp = tf.nn.xw_plus_b(mlp, w, b)
    return mlp


def build_graph(mode):
    print 'build graph', mode
    assert mode in ['train', 'val']

    x = tf.placeholder(tf.float32, [None, 5, 48, 64, 1])
    y = tf.placeholder(tf.float32, [None, 6])

    mlp = x
    print 'A', mlp.get_shape()

    mlp = conv(mlp, 3, name='conv1')
    mlp = tf.tanh(mlp)
    print 'A', mlp.get_shape()
    mlp = tf.nn.avg_pool3d(mlp, [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], 'SAME')
    print 'A', mlp.get_shape()

    mlp = conv(mlp, 29, name='conv2')
    mlp = tf.tanh(mlp)
    print 'A', mlp.get_shape()
    mlp = tf.nn.avg_pool3d(mlp, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME')
    print 'A', mlp.get_shape()

    mlp = tf.reshape(mlp, [-1, 1*9*13*29])
    print 'A', mlp.get_shape()

    mlp = fc(mlp, 6, 'fc')
    print 'A', mlp.get_shape()

    l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(mlp, y))
    acc = tf.equal(tf.argmax(mlp, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    y_hat = tf.nn.softmax(mlp)

    if mode == 'train':
        with tf.variable_scope('OPT'):
            opt = tf.train.AdamOptimizer()
            train_op = opt.minimize(l)
            with tf.control_dependencies([train_op]):
                train_op = tf.add_check_numerics_ops()
    else:
        train_op = None

    return [x, y, y_hat, l, acc, train_op]


def main():
    # 常量
    OUTPUT_DIR = './100/'
    MB = 26
    EPOCH_MAX = 1000000
    SNAPSHOT_RESUME_FROM = 0
    SNAPSHOT_INTERVAL = 1000000
    ES = 10

    # 加载数据集
    [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, TEST_X, TEST_Y] = DATA.load_cv()

    # fast test
    # TRAIN_X = TRAIN_X[0:MB, :, :, :, :]
    # TRAIN_Y = TRAIN_Y[0:MB, :]
    # VAL_X = VAL_X[0:MB, :, :, :, :]
    # VAL_Y = VAL_Y[0:MB, :]
    # TEST_X = TEST_X[0:MB, :, :, :, :]
    # TEST_Y = TEST_Y[0:MB, :]

    # 计算MEAN STD
    MEAN = np.mean(TRAIN_X, dtype=np.float32)
    STD = np.std(TRAIN_X, dtype=np.float32)
    print 'MEAN: ', MEAN
    print 'STD: ', STD

    # 创建计算图
    with tf.Graph().as_default():
        # 创建网络
        with tf.variable_scope('GRAPH', reuse=None):
            [train_x, train_y, train_y_hat, train_l, train_acc, train_op] = build_graph('train')
        with tf.variable_scope('GRAPH', reuse=True):
            [val_x, val_y, val_y_hat, val_l, val_acc, _] = build_graph('train')

        # 创建会话
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=1000000)

            # 初始化变量或加载快照
            if SNAPSHOT_RESUME_FROM == 0:
                print 'init vars'
                tf.global_variables_initializer().run()
            else:
                print 'load snapshot'
                saver.restore(sess, OUTPUT_DIR + 'snapshot-' + str(SNAPSHOT_RESUME_FROM))

            # 训练循环
            # 1 ~ EPOCH_MAX 或 SNAPSHOT_RESUME_FROM+1 ~ EPOCH_MAX
            es_epoch = 0
            es_train_acc = 0
            es_val_acc = 0
            es_test_acc = 0
            for epoch in xrange(SNAPSHOT_RESUME_FROM + 1, EPOCH_MAX + 1):
                print '---------- epoch %d ----------' % epoch
                t = time.time()

                # 打乱训练集
                idx = np.random.permutation(TRAIN_X.shape[0])
                TRAIN_X = TRAIN_X[idx, :, :, :, :]
                TRAIN_Y = TRAIN_Y[idx, :]

                # 训练
                # 抛弃训练集尾部 担心变化的MB会影响ADAM BATCHNORM等计算
                mean_train_l = 0.0
                mean_train_acc = 0.0
                mean_train_count = 0
                ITER_COUNT = TRAIN_X.shape[0] / MB
                for itr in xrange(ITER_COUNT):
                    # 准备MB
                    train_x_val = TRAIN_X[itr * MB:itr * MB + MB, :, :, :]
                    train_y_val = TRAIN_Y[itr * MB:itr * MB + MB, :]
                    train_x_val = (train_x_val - MEAN) / STD

                    # run
                    [_, train_l_val, train_y_hat_val, train_acc_val] = \
                        sess.run([train_op, train_l, train_y_hat, train_acc], feed_dict={train_x: train_x_val, train_y: train_y_val})
                    mean_train_l += train_l_val * MB
                    mean_train_acc += train_acc_val * MB
                    mean_train_count += MB
                    # print (mean_train_l, mean_train_acc)
                print 'mean_train_l %g, mean_train_acc %g' % \
                      (mean_train_l / mean_train_count, mean_train_acc / mean_train_count)

                # 验证
                # 保留验证集尾部
                mean_val_l = 0.0
                mean_val_acc = 0.0
                mean_val_count = 0
                ITER_COUNT = ((VAL_X.shape[0] - 1) / MB) + 1
                for itr in xrange(ITER_COUNT):
                    # 准备MB
                    mb = min(itr * MB + MB, VAL_X.shape[0]) - itr * MB
                    val_x_val = VAL_X[itr * MB:itr * MB + mb, :, :, :]
                    val_y_val = VAL_Y[itr * MB:itr * MB + mb, :]
                    val_x_val = (val_x_val - MEAN) / STD

                    # run
                    [val_l_val, val_y_hat_val, val_acc_val] = \
                        sess.run([val_l, val_y_hat, val_acc], feed_dict={val_x: val_x_val, val_y: val_y_val})
                    mean_val_l += val_l_val * mb
                    mean_val_acc += val_acc_val * mb
                    mean_val_count += mb
                    # print (mean_val_l, mean_val_acc)
                print 'mean_val_l %g, mean_val_acc %g' % \
                      (mean_val_l / mean_val_count, mean_val_acc / mean_val_count)

                # 测试
                # 保留验证集尾部
                mean_test_l = 0.0
                mean_test_acc = 0.0
                mean_test_count = 0
                ITER_COUNT = ((TEST_X.shape[0] - 1) / MB) + 1
                for itr in xrange(ITER_COUNT):
                    # 准备MB
                    mb = min(itr * MB + MB, TEST_X.shape[0]) - itr * MB
                    test_x_val = TEST_X[itr * MB:itr * MB + mb, :, :, :]
                    test_y_val = TEST_Y[itr * MB:itr * MB + mb, :]
                    test_x_val = (test_x_val - MEAN) / STD

                    # run
                    [test_l_val, test_y_hat_val, test_acc_val] = \
                        sess.run([val_l, val_y_hat, val_acc], feed_dict={val_x: test_x_val, val_y: test_y_val})
                    mean_test_l += test_l_val * mb
                    mean_test_acc += test_acc_val * mb
                    mean_test_count += mb
                    # print (mean_test_l, mean_test_acc)
                print 'mean_test_l %g, mean_test_acc %g' % \
                      (mean_test_l / mean_test_count, mean_test_acc / mean_test_count)

                # 计划的save snapshot
                if (epoch % SNAPSHOT_INTERVAL) == 0:
                    saver.save(sess, OUTPUT_DIR+'snapshot-'+str(epoch), write_meta_graph=False)
                    print 'save snapshot'

                print 't %g' % (time.time() - t)

                # early stopping
                if (mean_val_acc / mean_val_count) > es_val_acc:
                    es_epoch = epoch
                    es_train_acc = (mean_train_acc / mean_train_count)
                    es_val_acc = (mean_val_acc / mean_val_count)
                    es_test_acc = (mean_test_acc / mean_test_count)
                print 'es_epoch %g, es_train_acc %g, es_val_acc %g, es_test_acc %g' % \
                      (es_epoch, es_train_acc, es_val_acc, es_test_acc)
                if epoch >= es_epoch+ES:
                    break


if __name__ == '__main__':
    main()
