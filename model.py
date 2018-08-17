import numpy as np
import keras
from keras import backend as K
from keras.callbacks import Callback

# some aliases
fc = keras.layers.core.Dense


def KLdivergence(P, Y, n_dims=2, batch_size=100):
    alpha = n_dims - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(1e-14)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


def simple_dnn(input_layer):
    model = keras.models.Sequential()
    model.add(fc(500, input_shape=(input_layer.shape[1],), activation='relu'))
    model.add(fc(500, activation='relu'))
    model.add(fc(2000, activation='relu'))
    model.add(fc(2))
    model.compile(loss=KLdivergence, optimizer="adam")
    return model
