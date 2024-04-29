#! /usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import \
    Dense, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, Input, Activation, MaxPooling1D, MaxPooling2D, \
    Conv1D, Conv2D, BatchNormalization, LSTM, Flatten, ELU, AveragePooling1D, Permute
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import cvnn.layers as complex_layers
import cvnn.activations
from tensorflow.keras.losses import Loss, categorical_crossentropy
from keras.optimizers import RMSprop


class ComplexAverageCrossEntropy(Loss):

    def call(self, y_true, y_pred):
        real_loss = categorical_crossentropy(y_true, tf.math.real(y_pred))
        if y_pred.dtype.is_complex:
            imag_loss = categorical_crossentropy(y_true, tf.math.imag(y_pred))
        else:
            imag_loss = real_loss
        return (real_loss + imag_loss) / 2


def createSB_cart(inp_shape, classes_num=10, emb_size=64, weight_decay=1e-4, classification=False):

    tf.device("gpu:1")
    model = tf.keras.models.Sequential()
    model.add(complex_layers.ComplexInput(input_shape=inp_shape))
    model.add(complex_layers.ComplexConv1D(32, 2, activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(2))
    model.add(complex_layers.ComplexConv1D(64, 2, activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(2))
    model.add(complex_layers.ComplexConv1D(128, 2, activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(2))
    model.add(complex_layers.ComplexConv1D(256, 2, activation='crelu', padding='same'))
    model.add(complex_layers.ComplexAvgPooling1D(2))
    model.add(complex_layers.ComplexFlatten())
    #model.add(complex_layers.ComplexDropout(0.5))

    if classification:
        model.add(complex_layers.ComplexDense(classes_num, activation='softmax_real_with_abs'))
    else:
        model.add(complex_layers.ComplexDense(emb_size, activation='linear',
                                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                              bias_regularizer=tf.keras.regularizers.l2(weight_decay)))

    return model


def create_model(reluType, inp_shape, NUM_CLASS, emb_size, classification):
    print("model type: {}".format(reluType))
    if 'complex' == reluType:
        model = createSB_cart(inp_shape, NUM_CLASS, emb_size, classification=classification)

    else:
        raise ValueError('model type {} not support yet'.format(reluType))

    return model


def test_run(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # opt = tf.keras.optimizers.RMSprop(clipnorm=1.0)
    # model.compile(loss=ComplexAverageCrossEntropy(), metrics=["accuracy"], optimizer=opt)


def test():
    modelTypes = ['complex']
    reluType = ['cart']
    NUM_CLASS = 10
    signal = True
    inp_shape = (288, 1)
    emb_size = 64
    for reluType in modelTypes:
        model = create_model(reluType, inp_shape, NUM_CLASS, emb_size, classification=True)
        try:
            test_run(model)
        except Exception as e:
            print(e)
    print('all done!') if signal else print('test failed')


if __name__ == "__main__":
    test()
