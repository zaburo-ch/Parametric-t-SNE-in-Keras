import pathlib
import argparse
import sys

import numpy as np


def main(args):
    if args.split_ratio <= 0. or args.split_ratio >= 1.:
        print('Invalid split-ratio')
        sys.exit(1)

    n_first = int(args.n_sample * args.split_ratio)
    n_second = args.n_sample - n_first

    arr = np.random.normal(20., size=(args.n_sample, args.n_feature))
    arr[:n_first] *= -1.

    # save arr
    np.save(args.output, arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample some pretty points.')
    parser.add_argument(
        '--n-sample', type=int, default=1000,
        help='number of samples')
    parser.add_argument(
        '--n-feature', type=int, default=20,
        help='number of features')
    parser.add_argument(
        '--split-ratio', type=float, default=0.2,
        help='splitting ratio')
    parser.add_argument(
        '--output', type=pathlib.Path,
        default=pathlib.Path('sample.npy'),
        help='output file name')

    main(parser.parse_args())
