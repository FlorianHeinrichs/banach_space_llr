#
# nadaraya_watson_estimation.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-03-05
# Author: Florian Heinrichs
#
# Script to estimate the mean through Nadaraya-Watson estimation.

import numpy as np

from tools import get_kernel, convolve

def nadaraya_watson_estimator(X: np.ndarray,
                              bw: int,
                              filter_array: np.ndarray = None) -> np.ndarray:
    """
    Calculate the Nadaraya-Watson estimator. Possibly mask observations by
    providing a filter. First dimension should index time.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator as int.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the NW-estimator(s) given as numpy array.
    """
    spatial_dims = len(X.shape) - 1

    if filter_array is None:
        filter_array = np.ones_like(X, dtype=bool)

    X_filtered = X.copy()
    X_filtered[~filter_array] = 0

    _, kernel = get_kernel(bw)
    kernel = np.expand_dims(kernel, axis=0)
    kernel = kernel.reshape((1,) * spatial_dims + (-1,))

    support = np.ones_like(X_filtered)
    support[~filter_array] = 0

    padding = ((bw, bw),) + spatial_dims * ((0, 0),)
    support = np.pad(support, padding, mode='edge')
    X_filtered = np.pad(X_filtered, padding, mode='edge')

    R = convolve(X_filtered,  kernel)
    S = convolve(support, kernel) + 1e-10

    estimator = (R / S)

    return estimator


if __name__ == '__main__':
    from visualize_data import plot_3d
    from simulate_data import mu1, generate_iid

    mean_operator = mu1(50, 100)
    noise = generate_iid(50, 100, 'BM')
    signal = mean_operator + noise

    estimator = nadaraya_watson_estimator(signal, 10)
    plot_3d(mean_operator, y=estimator)