#! /usr/bin/env python
from __future__ import division
import os
import sys
import argparse
import pdb
import time
from statistics import mean, stdev
import random
import time
import numpy as np
import cvnn.layers as complex_layers
from tensorflow.keras.models import load_model

RootDir = os.getcwd()
toolsDir = os.path.join(RootDir, 'tools')
sys.path.append(toolsDir)
import load_slice_IQ
from utility import create_test_set_Wang_disjoint, kNN_accuracy, getRFdataDict

currentDir = os.path.dirname(__file__)
ResDir = os.path.join(currentDir, 'TF/res_out')
os.makedirs(ResDir, exist_ok=True)


def Wang_Disjoint_Experment(opts, modelPath, n_shot, max_n=20):
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with different distributions.
    The model is trained on AWF777 and tested on Wang100 and the set of
    websites in the training set and the testing set are mutually exclusive.
    '''
    features_model = load_model(modelPath, compile=False, custom_objects={'ComplexConv1D': complex_layers.ComplexConv1D,
                                                                          'ComplexInput': complex_layers.ComplexInput,
                                                                          'ComplexAvgPooling1D': complex_layers.ComplexAvgPooling1D,
                                                                          'ComplexFlatten': complex_layers.ComplexFlatten,
                                                                          'ComplexDense': complex_layers.ComplexDense,
                                                                          'ComplexDropout': complex_layers.ComplexDropout,
                                                                          'ComplexBatchNormalization': complex_layers.ComplexBatchNormalization,
                                                                          })
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper
    m = load_model(modelPath, compile=False, custom_objects={'ComplexConv1D': complex_layers.ComplexConv1D,
                                                             'ComplexInput': complex_layers.ComplexInput,
                                                             'ComplexAvgPooling1D': complex_layers.ComplexAvgPooling1D,
                                                             'ComplexFlatten': complex_layers.ComplexFlatten,
                                                             'ComplexDense': complex_layers.ComplexDense,
                                                             'ComplexDropout': complex_layers.ComplexDropout,
                                                             'ComplexBatchNormalization': complex_layers.ComplexBatchNormalization,
                                                             })
    type_exp = 'N-MEV'

    # KNeighborsClassifier(n_neighbors=n_shot, weights='distance', p=2, metric='cosine', algorithm='brute')
    params = {'k': n_shot,
              'weights': 'distance',
              'p': 2,
              'metric': 'cosine'
              }

    print("N_shot: ", n_shot)
    acc_list_Top1, acc_list_Top5 = [], []
    exp_times = 2
    total_time = 0

    for i in range(exp_times):
        #signature_dict, test_dict, sites = utility.getRFdataDict(opts.input, n_shot, data_dim, train_pool_size=20, test_size=70)
        signature_dict, test_dict, sites = getRFdataDict(opts.testData, opts, n_shot, n_instance=(2000 + max_n), max_n=max_n)
        if i == 0:
            size_of_problem = len(list(test_dict.keys()))
            print("Size of Problem: ", size_of_problem)
        signature_vector_dict, test_vector_dict = create_test_set_Wang_disjoint(signature_dict,
                                                                                test_dict,
                                                                                sites,
                                                                                features_model=features_model,
                                                                                type_exp=type_exp)
        # Measure the performance (accuracy)
        start = time.time()
        acc_knn_top1, acc_knn_top5 = kNN_accuracy(signature_vector_dict, test_vector_dict, params)
        end = time.time()
        total_time += (end - start)
        acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
        acc_list_Top5.append(float("{0:.15f}".format(round(acc_knn_top5, 5))))

    print(str(acc_list_Top1).strip('[]'))
    print(str(acc_list_Top5).strip('[]'))
    rtnLine = 'n_shot: {}\tacc for top 1: {} and std is: {}\n'.format(n_shot, mean(acc_list_Top1), stdev(acc_list_Top1))
    rtnLine = rtnLine + '\nacc for top 5: {} and std is: {}\n'.format(mean(acc_list_Top5), stdev(acc_list_Top5))
    rtnLine = rtnLine + '\nKNN training time : {}\n'.format(total_time / exp_times)
    print(rtnLine)
    return rtnLine


def run(opts):
    source = "out_lab"
    outfile = os.path.join(ResDir, 'ccs19_target{}_results.txt'.format(source))
    f = open(outfile, 'a+')
    print('\n\n#################### test time is: {} #########################'.format(time.ctime()), file=f)
    tsnList = [100, 200, 400, 800, 1600]
    modelPath = "/lustre/work/swanlab/fafrin2/usrp/hackrf/after_fft/crelu/s288/TF/res_out/modelDir/complex_triplet_out_lab_400_False.h5"
    for tsn in tsnList:
        rtnLine = Wang_Disjoint_Experment(opts, modelPath, n_shot=tsn, max_n=max(tsnList))
        print(rtnLine, file=f)
    f.close()


class tfOpts():
    def __init__(self, source_path, test_path, location, file_key='*.bin', num_slice=1000, start_ix=0, slice_len=288,
                 stride=288, window=64, dataType='IQ'):
        self.root_dir = source_path
        self.testData = test_path
        self.semiHard = True
        self.plotModel = False
        self.useGpu = True
        self.mul_trans = True
        self.testType = "tsn"
        self.num_slice = num_slice
        self.start_ix = start_ix
        self.slice_len = slice_len
        self.stride = stride
        self.file_key = file_key
        self.location = location
        self.channel_first = False
        self.sample = True
        self.window = window
        self.dataType = dataType


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--modelPath', help='')
    parser.add_argument('-t', '--exp_type', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    # opts = parseArgs(sys.argv)
    opts = tfOpts(source_path="/work/swanlab/fafrin2/hackrf/our_day1/our_day1/",
                  test_path="/work/swanlab/fafrin2/hackrf/our_day2/our_day2/",
                  location='symbols')
    run(opts)
