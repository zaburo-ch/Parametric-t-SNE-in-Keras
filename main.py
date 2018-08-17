import pathlib
import os
import argparse
import multiprocessing

from tqdm import tqdm
import numpy as np
import keras

import model


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
                beta = beta * 2.
            else:
                beta = (betamin + betamax) / 2.
        else:
            betamax = beta
            if betamin == -np.inf:
                beta = beta / 2.
            else:
                beta = (betamin + betamax) / 2.

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries += 1

    return i, thisP


def x2p(X, perplexity, n_jobs=4):
    tol = 1e-5
    n = X.shape[0]
    logU = np.log(perplexity)

    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))

    idx = np.logical_not(np.identity(n, dtype=np.bool))
    D = D[idx].reshape([n, -1])

    def generator():
        for i in range(n):
            yield i, D[i], tol, logU

    pool = multiprocessing.Pool(n_jobs)
    result = pool.map(x2p_job, generator())
    P = np.zeros([n, n])
    for i, thisP in result:
        P[i, idx[i]] = thisP

    return P


def calculate_P(X, perplexity, batch_size):
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):
        P_batch = x2p(X[i:i + batch_size], perplexity)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12)
        P[i:i + batch_size] = P_batch
    return P


def main(args):
    RESULT_DIR = pathlib.Path('result')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print()

    if args.command == 'train':
        print('Loading dataset.. ', end='')
        dataset = np.load(args.dataset)
        n = dataset.shape[0]
        print('Done', end='\n\n')

        model_f = model.simple_dnn(dataset)
        batch_num = n // args.batch_size
        m = batch_num * args.batch_size

        print('Start training..')
        for epoch in tqdm(range(args.n_epoch)):
            # shuffle dataset and calculate P
            if epoch % (args.n_epoch + 1) == 0:
                X = dataset[np.random.permutation(n)[:m]]
                P = calculate_P(X, args.perplexity, args.batch_size)

            # train
            for i in range(0, n, args.batch_size):
                model_f.train_on_batch(
                    X[i : i + args.batch_size],
                    P[i : i + args.batch_size])
        model_f.save(args.savepath)
        print('Model saved to', args.savepath)
        print('Done', end='\n\n')

        print('Predicting with training dataset.. ', end='')
        pred = model_f.predict(dataset)
        np.save(RESULT_DIR / 'train.npy', pred)
        print('Done', end='\n\n')

    if args.command == 'predict':
        print('Loading dataset.. ', end='')
        dataset = np.load(args.dataset)
        n = dataset.shape[0]
        print('Done')

        model_f = keras.models.load_model(
            args.model, custom_objects={'KLdivergence': model.KLdivergence})

        print('Predicting with given dataset.. ', end='')
        pred = model_f.predict(dataset)
        np.save(RESULT_DIR / 'predict.npy', pred)
        print('Done', end='\n\n')

    keras.backend.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parametric t-SNE.')
    subparsers = parser.add_subparsers(dest='command')

    train_group = subparsers.add_parser(
        'train',
        description='Train a new paramteric t-SNE model.')
    train_group.add_argument(
        '--dataset', type=pathlib.Path,
        default=pathlib.Path('dataset', 'sample1000.npy'),
        help='dataset for training')
    train_group.add_argument(
        '--batch-size', type=int, default=100,
        help='batch size')
    train_group.add_argument(
        '--lower-dim', type=int, default=2,
        help='dimension of embedded space')
    train_group.add_argument(
        '--n-epoch', type=int, default=200,
        help='number of training epochs')
    train_group.add_argument(
        '--perplexity', type=float, default=30.,
        help='perplexity value')
    train_group.add_argument(
        '--savepath', type=pathlib.Path,
        default=pathlib.Path('model', 'model.h5'),
        help='save path for trained model')

    predict_group = subparsers.add_parser(
        'predict',
        description='Predict using pre-trained parametric t-SNE model.')
    predict_group.add_argument(
        '--dataset', type=pathlib.Path,
        default=pathlib.Path('dataset', 'sample100.npy'),
        help='dataset for predicting')
    predict_group.add_argument(
        '--model', type=pathlib.Path,
        default=pathlib.Path('model', 'model.h5'),
        help='pre-trained model')

    main(parser.parse_args())
