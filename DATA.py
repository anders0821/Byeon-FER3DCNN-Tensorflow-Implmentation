# -*- coding:utf-8 -*-

import numpy as np
import h5py


def load_cv():
    # open h5
    FN = './CONVERT.mat'
    h5 = h5py.File(FN, 'r')

    # train
    TRAIN_X = h5['TRAINX'][:]
    N_TRAIN = TRAIN_X.shape[0]
    assert (TRAIN_X.shape == (N_TRAIN, 5, 48, 64, 1))
    TRAIN_X = np.float32(TRAIN_X)
    TRAIN_Y = h5['TRAINY'][:]
    assert (TRAIN_Y.shape == (N_TRAIN, 6))
    TRAIN_Y = np.float32(TRAIN_Y)

    # val
    VAL_X = h5['VALX'][:]
    N_VAL = VAL_X.shape[0]
    assert (VAL_X.shape == (N_VAL, 5, 48, 64, 1))
    VAL_X = np.float32(VAL_X)
    VAL_Y = h5['VALY'][:]
    assert (VAL_Y.shape == (N_VAL, 6))
    VAL_Y = np.float32(VAL_Y)

    # test
    TEST_X = h5['TESTX'][:]
    N_TEST = TEST_X.shape[0]
    assert (TEST_X.shape == (N_TEST, 5, 48, 64, 1))
    TEST_X = np.float32(TEST_X)
    TEST_Y = h5['TESTY'][:]
    assert (TEST_Y.shape == (N_TEST, 6))
    TEST_Y = np.float32(TEST_Y)

    # print shape
    print 'load cv'
    print 'TRAIN_X.shape', TRAIN_X.shape
    print 'TRAIN_Y.shape', TRAIN_Y.shape
    print 'VAL_X.shape', VAL_X.shape
    print 'VAL_Y.shape', VAL_Y.shape
    print 'TEST_X.shape', TEST_X.shape
    print 'TEST_Y.shape', TEST_Y.shape

    return [TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, TEST_X, TEST_Y]
