import numpy as np
np.random.seed(71)

import matplotlib
matplotlib.use('Agg')

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.datasets import cifar10, mnist

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import multiprocessing as mp


batch_size = 5000
low_dim = 2
nb_epoch = 100
shuffle_interval = nb_epoch + 1
n_jobs = 4
perplexity = 30.0


def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p_job(data):
    i, Di, tol, logU = data
    beta = 1.0
    betamin = -np.inf
    betamax = np.inf
    H, thisP = Hbeta(Di, beta)

    Hdiff = H - logU
    tries = 0
    while np.abs(Hdiff) > tol and tries < 50:
        if Hdiff > 0:
            betamin = beta
            if betamax == -np.inf:
                beta = beta * 2
            else:
                beta = (betamin + betamax) / 2
        else:
            betamax = beta
            if betamin == -np.inf:
                beta = beta / 2
            else:
                beta = (betamin + betamax) / 2

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries += 1

    return i, thisP


def x2p(X):
    tol = 1e-5
    n = X.shape[0]
    logU = np.log(perplexity)

    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))

    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape([n, -1])

    def generator():
        for i in xrange(n):
            yield i, D[i], tol, logU

    pool = mp.Pool(n_jobs)
    result = pool.map(x2p_job, generator())
    P = np.zeros([n, n])
    for i, thisP in result:
        P[i, idx[i]] = thisP

    return P


def calculate_P(X):
    print "Computing pairwise distances..."
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in xrange(0, n, batch_size):
        P_batch = x2p(X[i:i + batch_size])
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12)
        P[i:i + batch_size] = P_batch
    return P


def KLdivergence(P, Y):
    alpha = low_dim - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


print "load data"
# # cifar-10
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# n, channel, row, col = X_train.shape

# # mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
n, row, col = X_train.shape
channel = 1

X_train = X_train.reshape(-1, channel * row * col)
X_test = X_test.reshape(-1, channel * row * col)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print "X_train.shape:", X_train.shape
print "X_test.shape:", X_test.shape

batch_num = int(n // batch_size)
m = batch_num * batch_size


print "build model"
model = Sequential()
model.add(Dense(500, input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(2000))
model.add(Activation('relu'))
model.add(Dense(2))

model.compile(loss=KLdivergence, optimizer="adam")


print "fit"
images = []
fig = plt.figure(figsize=(5, 5))

for epoch in range(nb_epoch):
    # shuffle X_train and calculate P
    if epoch % shuffle_interval == 0:
        X = X_train[np.random.permutation(n)[:m]]
        P = calculate_P(X)

    # train
    loss = 0
    for i in xrange(0, n, batch_size):
        loss += model.train_on_batch(X[i:i+batch_size], P[i:i+batch_size])
    print "Epoch: {}/{}, loss: {}".format(epoch+1, nb_epoch, loss / batch_num)

    # visualize training process
    pred = model.predict(X_test)
    img = plt.scatter(pred[:, 0], pred[:, 1], c=y_test,
                      marker='o', s=3, edgecolor='')
    images.append([img])

ani = ArtistAnimation(fig, images, interval=100, repeat_delay=2000)
ani.save("mlp_process.mp4")

plt.clf()
fig = plt.figure(figsize=(5, 5))
pred = model.predict(X_test)
plt.scatter(pred[:, 0], pred[:, 1], c=y_test, marker='o', s=4, edgecolor='')
fig.tight_layout()

plt.savefig("mlp_result.png")
