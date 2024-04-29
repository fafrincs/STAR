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


import numpy as np


def model_test():
    test_times = 5
    comb = list()
    for s in [True, False]:
        for m in [True, False]:
            comb.append([s, m])
    outfile = os.path.join(ResDir, 'test_acc.txt')
    f = open(outfile, 'a+')
    print('\n\n#################### My Dataset,  test time is: {} #########################'.format(time.ctime()),
          file=f)
    models = os.listdir('/lustre/work/swanlab/fafrin2/usrp200/hacrf/after_fft/crelu/s288/res_out/modelDir/')
    best_model_name = ""
    best_model_acc = 0.0
    for m in models:
        modelPath = '/lustre/work/swanlab/fafrin2/usrp200/hacrf/after_fft/crelu/s288/res_out/modelDir/' + m
        model_name = os.path.basename(modelPath).split('_')
        sample = model_name[10]
        mul_trans = model_name[13]
        print(modelPath, sample, mul_trans)
        m = tf.keras.models.load_model(modelPath, custom_objects={'ComplexConv1D': complex_layers.ComplexConv1D,
                                                                  'ComplexInput': complex_layers.ComplexInput,
                                                                  'ComplexAvgPooling1D': complex_layers.ComplexAvgPooling1D,
                                                                  'ComplexFlatten': complex_layers.ComplexFlatten,
                                                                  'ComplexDense': complex_layers.ComplexDense,
                                                                  'ComplexDropout': complex_layers.ComplexDropout,
                                                                  'ComplexBatchNormalization': complex_layers.ComplexBatchNormalization,
                                                                  'ComplexAverageCrossEntropy': ComplexAverageCrossEntropy})
        m.summary()
        dataPath = "/work/swanlab/fafrin2/hackrf/our_day2/"
        for c in comb:
            acc_list = list()
            for i in range(test_times):
                opts = tfOpts(source_path=dataPath + "our_day2/", location='symbols')
                dataOpts = load_slice_IQ.loadDataOpts(opts.root_dir, opts.location, num_slice=1000,
                                                      slice_len=opts.slice_len, start_idx=0, sample=c[0],
                                                      mul_trans=c[1])

                x, y, _, _, NUM_CLASS = load_slice_IQ.loadData(dataOpts, split=False)

                print("class:", NUM_CLASS)

                test_y = to_categorical(y, NUM_CLASS)
                print("testy:", test_y)
                test_x, test_y = mytools.shuffleData(x, test_y)
                print("testx:", test_x)

                score, acc = m.evaluate(test_x, test_y, batch_size=64, verbose=1)
                acc_list.append(acc)
                print("Source sample and mul_trans: {},{}, Target sample and mul_trans: {},{}. Acc: {}".format(sample,
                                                                                                               mul_trans,
                                                                                                               c[0],
                                                                                                               c[1],
                                                                                                               acc))
            mean_acc = mean(acc_list)
            if mean_acc > best_model_acc:
                best_model_acc = mean_acc
                best_model_name = modelPath

            print("Source sample and mul_trans: {},{}, Target sample and mul_trans: {},{}. Ave acc: {}, std: {}, model: {}".format(
                sample, mul_trans, c[0], c[1], mean_acc, np.std(acc_list), best_model_name))
            
    print("Source sample and mul_trans: {},{}, Target sample and mul_trans: {},{}. Ave acc: {:.4f}, std: {:.4f}, model: {}".format(
                sample, mul_trans, c[0], c[1], mean_acc, np.std(acc_list), best_model_name), file=f)

    #dict = {"Ave_acc: {:.4f}, std: {:.4f}, model: {}".format(mean_acc, stdev(acc_list), best_model_name), file==f}

    
    #df = pd.DataFrame(dict, index=[mean_acc])

   # print(df, file=f, flush=True)
    print('all test done!')

    f.close()

    print("Best accuracy model: {}".format(best_model_name))


if __name__ == '__main__':
    model_test()
