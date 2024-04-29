import os

import numpy as np
import pandas as pd
from statistics import mean, stdev
import load_slice_IQ
import tensorflow as tf
import tools as mytools
import cvnn.layers as complex_layers
from tensorflow.keras.losses import Loss, categorical_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from utility import class_ranks
import time

RootDir = os.getcwd()
currentDir = os.path.dirname(__file__)
ResDir = os.path.join(currentDir, 'res_out')
os.makedirs(ResDir, exist_ok=True)


class ComplexAverageCrossEntropy(Loss):

    def call(self, y_true, y_pred):
        real_loss = categorical_crossentropy(y_true, tf.math.real(y_pred))
        if y_pred.dtype.is_complex:
            imag_loss = categorical_crossentropy(y_true, tf.math.imag(y_pred))
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2


class tfOpts():
    def __init__(self, source_path, location, file_key='*.bin', num_slice=1000, start_ix=0, slice_len=288, stride=288):
        self.root_dir = source_path
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.location = location
        self.sample = True


def rank_model_test():
    modelPath = '/lustre/work/swanlab/fafrin2/usrp200/outlab_outdoor/after_fft/crelu/s288/res_out/modelDir/IQ_model_out_lab_complex_after_fft_slices_1000_startIdx_400000_stride_288_len_288_STFT_64.h5_0.64'
    # modelPath = 'C:/Users/fafrin2/Downloads/cart/res_out/modelDir/IQ_model_out_1_complex_after_fft_slices_10000_startIdx_100000_stride_144_len_288_STFT_64.h5'
    m = tf.keras.models.load_model(modelPath, custom_objects={'ComplexConv1D': complex_layers.ComplexConv1D,
                                                              'ComplexInput': complex_layers.ComplexInput,
                                                              'ComplexAvgPooling1D': complex_layers.ComplexAvgPooling1D,
                                                              'ComplexFlatten': complex_layers.ComplexFlatten,
                                                              'ComplexDense': complex_layers.ComplexDense,
                                                              'ComplexDropout': complex_layers.ComplexDropout,
                                                              'ComplexBatchNormalization': complex_layers.ComplexBatchNormalization,
                                                              'ComplexAverageCrossEntropy': ComplexAverageCrossEntropy})

    m.summary()
    dataPath = "/work/swanlab/fafrin2/usrp_b200/out_lab/out_lab/"
    opts = tfOpts(source_path=dataPath, location='after_fft')
    dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location, num_slice=1000, slice_len=288, stride=288,
                                          start_idx=400000, sample=False, dataType='IQ')
    # test_time = 3

    X, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)
    test_y = to_categorical(y, NUM_CLASS)
    test_x, test_y = mytools.shuffleData(X, test_y)

    score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
    print("Acc: {}".format(acc))

    ranks = class_ranks(m, X, y, NUM_CLASS, preds=None)

    total_rank = np.array(ranks)
    # else:
    #    total_rank += np.array(ranks)
    #    print(total_rank.shape)

    df = pd.DataFrame(total_rank, index=None)
    name = os.path.basename(modelPath)
    df.to_csv("sameday_rank_" + name + '.csv', header=False)


if __name__ == '__main__':
    rank_model_test()
