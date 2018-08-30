import pathlib
import os
import argparse

from tqdm import tqdm
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense as fc


class ParametricTSNE(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, perplexity=30., n_iter=100, verbose=0):
        """parametric t-SNE

        Keyword Arguments:

            - n_components -- dimension of the embedded space

            - perplexity -- the perplexity is related to the number of nearest
                            neighbors that is used in other manifold learning
                            algorithms

            - n_iter -- maximum number of iterations for the optimizaiton.

            - verbose -- verbosity level
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.verbose = verbose

        self.model = None

    def fit(self, X, y=None, batch_size=100):
        """fit the model with X"""
        n_sample, n_feature = X.shape

        self._log('Building model..', end=' ')
        self._build_model(n_feature, self.n_components)
        self._log('Done')

        self._log('Start training..')
        for epoch in tqdm(range(self.n_iter)):
            new_indices = np.random.permutation(n_sample)
            X = X[new_indices]
            P = self._neighbor_distribution(X, batch_size=batch_size)

            for i in range(0, n_sample, batch_size):
                batch_slice = slice(i, i + batch_size)
                self.model.train_on_batch(X[batch_slice], P[batch_slice])
        self._log('Done')

        return self  # scikit-learn does so..

    def transform(self, X):
        """apply dimensionality reduction to X"""
        # fit should have been called before
        if self.model is None:
            raise sklearn.exceptions.NotFittedError(
                'This ParametricTSNE instance is not fitted yet. Call \'fit\''
                ' with appropriate arguments before using this method.')

        self._log('Predicting embedding points..', end=' ')
        X_new = self.model.predict(X)
        self._log('Done')
        return X_new

    def fit_transform(self, X, y=None):
        """fit the model with X and apply the dimensionality reduction on X."""
        self.fit(X)

        X_new = self.transform(X)
        return X_new

    def _neighbor_distribution(
        self, x, err=1e-5, max_iteration=50, batch_size=100):
        """calculate neighbor distribution from x

        Keyword Arguments:

            - err -- tolerance level for searching sigma numerically

            - max_iteration -- maximum number of iterations for finding sigma

            - batch_size -- batch size for training
        """
        n = x.shape[0]
        log_k = np.log(self.perplexity)

        # calculate squared l2 distance matrix d from x
        d = np.expand_dims(x, axis=0) - np.expand_dims(x, axis=1)
        d = np.square(d)
        d = np.sum(d, axis=-1)

        # find appropriate sigma values with bisection method and
        # multi-threading
        def beta_search(d_i):
            # beta = 1 / (2 * sigma**2)
            beta = 1.0
            beta_min = 0.
            beta_max = np.inf

            for iteration in range(max_iteration):
                # calculate entropy and probability distribution from given
                # beta value
                p_i = np.exp(-d_i * beta)
                s = np.sum(p_i)
                h_i = beta * np.sum(d_i * p_i) / s + np.log(s)

                # normalize p
                p_i /= s

                h_diff = h_i - log_k
                if np.abs(h_diff) < err:
                    break

                if h_diff < 0.:
                    beta_max = beta
                    beta = (beta + beta_min) / 2.
                else:
                    beta_min = beta
                    if beta_max < np.inf:
                        beta = (beta + beta_max) / 2.
                    else:
                        beta *= 2.

            return p_i

        p = np.zeros(shape=(n, n))
        for i in range(n):
            p[i] = beta_search(d[i])

        # remove nan entries
        nan_indices = np.isnan(p)
        p[nan_indices] = 0.

        # make p symmetric and normalize
        p = p + p.T
        p /= np.sum(p)
        p = np.maximum(p, 1e-12)

        p_batches = np.zeros(shape=(n, batch_size), dtype=np.float32)
        for i in range(0, n, batch_size):
            if i + batch_size > n:
                break

            batch_slice = slice(i, i + batch_size)
            p_batches[batch_slice, :] = p[batch_slice, batch_slice]

        return p_batches

    def _kl_divergence(self, P, Y, batch_size=100):
        eps = K.variable(1e-14, dtype='float32')

        # calculate neighbor distribution Q (t-distribution) from Y
        D = K.expand_dims(Y, axis=0) - K.expand_dims(Y, axis=1)
        D = K.square(D)
        D = K.sum(D, axis=-1)

        Q = K.pow(1. + D, -1)

        # eliminate all diagonals
        non_diagonals = 1 - K.eye(batch_size, dtype='float32')
        Q = Q * non_diagonals

        # normalize
        sum_Q = K.sum(Q)
        Q /= sum_Q
        Q = K.maximum(Q, eps)

        divergence = K.sum(P * K.log((P + eps) / (Q + eps)))
        return divergence

    def _build_model(self, n_input, n_output):
        self.model = Sequential()
        self.model.add(fc(500, input_dim=n_input, activation='relu'))
        self.model.add(fc(500, activation='relu'))
        self.model.add(fc(2000, activation='relu'))
        self.model.add(fc(n_output))
        self.model.compile(loss=self._kl_divergence, optimizer='adam')

    def _log(self, *args, **kwargs):
        """logging with given arguments and keyword arguments"""
        if self.verbose >= 1:
            print(*args, **kwargs)


def main(args):
    RESULT_DIR = pathlib.Path('result')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print('Loading dataset.. ', end='')
    dataset = np.load(args.dataset).astype(np.float32)
    print('Done')

    ptsne = ParametricTSNE(
        n_components=args.n_components,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        verbose=1)
    pred = ptsne.fit_transform(dataset)
    np.save(RESULT_DIR / 'tranform.npy', pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parametric t-SNE.')
    subparsers = parser.add_subparsers(dest='command')

    train_group = subparsers.add_parser(
        'train',
        description='Train a new parametric t-SNE model.')
    parser.add_argument(
        '--dataset', type=pathlib.Path,
        default=pathlib.Path('dataset', 'sample1000.npy'),
        help='dataset for training')
    parser.add_argument(
        '--n-components', type=int, default=2,
        help='dimension of embedded space')
    parser.add_argument(
        '--perplexity', type=float, default=30.,
        help='perplexity value')
    parser.add_argument(
        '--n-iter', type=int, default=100,
        help='number of training epochs')

    main(parser.parse_args())
