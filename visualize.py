import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main(args):
    dataset = np.load(args.result)

    sns.scatterplot(x=dataset[:, 0], y=dataset[:, 1])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize what the model predicted.')
    parser.add_argument(
        'result', type=pathlib.Path,
        help='model\'s prediction')

    main(parser.parse_args())
