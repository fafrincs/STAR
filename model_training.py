#!/usr/bin/python3

from __future__ import division
import os
import sys
import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, save_model
import rf_models
import time
import load_slice_IQ
import config
import get_simu_data
import cvnn.layers as complex_layers
from tensorflow.keras.losses import Loss, categorical_crossentropy
from tools import shuffleData
from statistics import mean, stdev
from keras.optimizers import RMSprop
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print(ROOT_DIR)
resDir = os.path.join(ROOT_DIR, 'res_out')
modelDir = os.path.join(resDir, 'modelDir')
os.makedirs(modelDir, exist_ok=True)


class ComplexAverageCrossEntropy(Loss):

    def call(self, y_true, y_pred):
        real_loss = categorical_crossentropy(y_true, tf.math.real(y_pred))
        if y_pred.dtype.is_complex:
            imag_loss = categorical_crossentropy(y_true, tf.math.imag(y_pred))
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2


class LearningController(Callback):
    def __init__(self, num_epoch=0, lr=0., learn_minute=0):
        self.num_epoch = num_epoch
        self.learn_second = learn_minute * 60
        self.lr = lr
        if self.learn_second > 0:
            print("Leraning rate is controled by time.")
        elif self.num_epoch > 0:
            print("Leraning rate is controled by epoch.")

    def on_train_begin(self, logs=None):
        if self.learn_second > 0:
            self.start_time = time.time()
        self.model.optimizer.lr = self.lr

    def on_epoch_end(self, epoch, logs=None):
        if self.learn_second > 0:
            current_time = time.time()
            if current_time - self.start_time > self.learn_second:
                self.model.stop_training = True
                print("Time is up.")
                return

            if current_time - self.start_time > self.learn_second / 2:
                self.model.optimizer.lr = self.lr * 0.1
            if current_time - self.start_time > self.learn_second * 3 / 4:
                self.model.optimizer.lr = self.lr * 0.01

        elif self.num_epoch > 0:
            if epoch >= self.num_epoch / 3:
                self.model.optimizer.lr = self.lr * 0.1
            if epoch >= self.num_epoch * 2 / 3:
                self.model.optimizer.lr = self.lr * 0.01

        print('lr:%.2e' % self.model.optimizer.lr.value())


def main(opts):
    # load data
    same_acc_list = []
    cross_acc_list = []
    target = os.path.basename(opts.trainData)
    outfile = os.path.join(resDir, 'multiruns_cross_day1_{}_res.csv'.format(target))
    f = open(outfile, 'a+')
    resLine = ''

    # setup params
    Batch_Size = 32
    Epoch_Num = 100
    lr = 0.1
    emb_size = 64
    idx_list = [0, 50000, 100000, 150000]

    # idx_list = [0,0,0]

    for idx in idx_list:
        dataOpts = load_slice_IQ.loadDataOpts(opts.input, opts.location, num_slice=opts.num_slice,
                                              slice_len=opts.slice_len,
                                              start_idx=idx, stride=opts.stride, mul_trans=opts.mul_trans,
                                              window=opts.window,
                                              dataType=opts.dataType)

        train_x, train_y, test_x, test_y, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=True)


        saveModelPath = os.path.join(modelDir,
                                     '{}_model_{}_{}_{}_slices_{}_startIdx_{}_stride_{}_len_{}_STFT_{}.h5'.format(
                                         opts.dataType, target, opts.modelType, opts.location, opts.num_slice, idx,
                                         opts.stride, opts.slice_len, opts.window))
        checkpointer = ModelCheckpoint(filepath=saveModelPath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                       mode='max')
        earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
        learning_controller = LearningController(num_epoch=Epoch_Num, lr=lr)
        callBackList = [checkpointer, earlyStopper]

        print('get the model and compile it...')
        inp_shape = (train_x.shape[1], train_x.shape[2])
        print('input shape: {}'.format(inp_shape))
        model = rf_models.create_model(opts.modelType, inp_shape, NUM_CLASS, emb_size, classification=True)
        model.summary()

        #model.compile(loss=ComplexAverageCrossEntropy(), metrics=["accuracy"], optimizer="RMSprop")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        train_y = to_categorical(train_y, NUM_CLASS)
        test_y = to_categorical(test_y, NUM_CLASS)
        train_x, train_y = shuffleData(train_x, train_y)

        print('fit the model with data...')
        start_time = time.time()
        model.fit(x=train_x, y=train_y,
                  batch_size=Batch_Size,
                  epochs=Epoch_Num,
                  verbose=opts.verbose,
                  callbacks=callBackList,
                  validation_split=0.2,
                  shuffle=True)
        end_time = time.time()
        duration = end_time - start_time

        print('test the trained model...')
        # load the model using load_model()

        m = tf.keras.models.load_model(saveModelPath, custom_objects={'ComplexConv1D': complex_layers.ComplexConv1D,
                                                                      'ComplexInput': complex_layers.ComplexInput,
                                                                      'ComplexAvgPooling1D': complex_layers.ComplexAvgPooling1D,
                                                                      'ComplexFlatten': complex_layers.ComplexFlatten,
                                                                      'ComplexDense': complex_layers.ComplexDense,
                                                                      'ComplexDropout': complex_layers.ComplexDropout,
                                                                      'ComplexBatchNormalization': complex_layers.ComplexBatchNormalization,
                                                                      'ComplexAverageCrossEntropy': ComplexAverageCrossEntropy})

        score, acc = m.evaluate(test_x, test_y, batch_size=Batch_Size, verbose=1)
        
        same_acc_list.append(acc)
        tf.keras.models.save_model(m, saveModelPath + '_{:.2f}'.format(acc), save_format='h5')
        print('test acc is: ', acc)
        resLine = resLine + 'dataset: {}, model: {}, location: {}, data size: {}, start_idx: {}, stride: {}\n'.format(
            target, opts.modelType, opts.location, opts.num_slice, idx, opts.stride)
        resLine = resLine + 'same acc is: {:f}, time last: {:f}\n\n'.format(acc, duration)



        print("start testing on cross scenario...")

        dataOpts = load_slice_IQ.loadDataOpts(opts.testData, opts.location, num_slice=opts.num_slice,
                                              slice_len=opts.slice_len,
                                              start_idx=idx, stride=opts.stride, mul_trans=opts.mul_trans,
                                              window=opts.window, dataType=opts.dataType)

        dataOpts.num_slice = int(opts.num_slice * 0.2)
        X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
        y = to_categorical(y, NUM_CLASS)
        X, y = shuffleData(X, y)
        cross_score, cross_acc = m.evaluate(X, y, batch_size=Batch_Size, verbose=1)
        print('cross test acc is: ', cross_acc)
        cross_acc_list.append(cross_acc)
        resLine = resLine + 'cross acc is: {:f}, time last: {:f}\n\n'.format(cross_acc, duration)
    resLine = resLine + 'ave same acc is : {:f}, avg same std is: {:f}, avg cross acc is : {:f}, avg cross std is: {:f}\n'.format(
        mean(same_acc_list), stdev(same_acc_list), mean(cross_acc_list), stdev(cross_acc_list))

    relu = '{}'.format(opts.reluType)
    slicing = '{}'.format(opts.slice_len)
    stride = '{}'.format(opts.stride)
    win = '{}'.format(opts.window)
    model = '{}'.format(opts.modelType)
    sameacc = '{:f}'.format(mean(same_acc_list))
    samestd = '{:f}'.format(stdev(same_acc_list))
    crossacc = '{:f}'.format(mean(cross_acc_list))
    crossstd = '{:f}'.format(stdev(cross_acc_list))

    dict = {'modelType': [model], 'stride': [stride], 'slice_len': [slicing], 'window': [win], 'sameacc': [sameacc],  'samestd': [samestd], 'crossacc': [crossacc], 'crossstd': [crossstd]}

    df = pd.DataFrame(dict, index=[relu])

    print(df, file=f, flush=True)
    print('all test done!')


class testOpts():
    def __init__(self, trainData, testData, location, modelType, reluType, num_slice, slice_len, start_idx, stride,
                 window,
                 dataType):
        self.input = trainData
        self.testData = testData
        self.modelType = modelType
        self.reluType = reluType
        self.location = location
        self.verbose = 1
        self.trainData = trainData
        self.splitType = 'random'
        self.normalize = False
        self.num_slice = num_slice
        self.slice_len = slice_len
        self.start_idx = start_idx
        self.stride = stride
        self.window = window
        self.mul_trans = True
        self.dataType = dataType


if __name__ == "__main__":

    source = ['our_day1']
    target = ['our_day2']
    data = list(zip(source, target))
    for s in [288]:
        for w in [64]:
            for m in ['complex']:
                for j in ['cart']:
                    for p in data:
                       # dataPath = 'C:/Users/fafrin2/Downloads/hackrf/our_day1/' + p[0]
                        #testPath = 'C:/Users/fafrin2/Downloads/hackrf/our_day2/' + p[1]
                        dataPath = '/work/swanlab/fafrin2/hackrf/our_day1/'+p[0]
                        testPath = '/work/swanlab/fafrin2/hackrf/our_day2/'+p[1]
                    opts = testOpts(trainData=dataPath, testData=testPath, location='symbols', modelType=m,
                                    reluType=j, num_slice=1000, slice_len=288, start_idx=0, stride=s, window=w,
                                    dataType='IQ')
                    main(opts)
