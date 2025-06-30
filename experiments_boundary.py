#
# experiments_boundary.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-06-04
# Author: Florian Heinrichs
#
# Script containing experiments with various estimators for point estimation at
# the boundary for t = 1.

from time import time
from typing import Callable

import numpy as np

from tools import get_kernel
from experiments import experiment


def single_estimator_boundary(X: np.ndarray,
                              estimator: Callable,
                              name: str,
                              mean: np.ndarray) -> dict:
    """
    Auxiliary function to calculate estimates at the boundary t = 1 and compare
    their MAE and MSE.

    :param X: NumPy array containing functional time series: (n_time, n_space).
    :param estimator: Estimator for mean estimation.
    :param name: Dummy variable for compatibility with experiment().
    :param mean: (Real) Mean operator, used to calculate MAE and MSE at t = 1.
    :return: Results as dictionary with keys 'MAE', 'MSE', 'time'.
    """
    n_time = X.shape[0]
    mean = np.expand_dims(mean[-1], axis=-1)
    min_bw, max_bw = 1, int(np.sqrt(n_time))

    results = {'time': [], 'MAE': [], 'MSE': []}

    for bw in range(min_bw, max_bw + 1):
        start = time()
        estimate = estimator(X[-(bw + 1):])
        end = time()

        mae = np.mean(np.abs(estimate - mean))
        mse = np.mean((estimate - mean) ** 2)

        results['time'].append(1000 * (end - start))
        results['MAE'].append(mae)
        results['MSE'].append(mse)

    return results


def nwe_boundary(X: np.ndarray) -> np.ndarray:
    """
    Calculate the Nadaraya-Watson estimator of the rightmost point t=1. First
    dimension should index time.

    :param X: Time series given as numpy array.
    :return: Returns the NW-estimator(s) given as numpy array.
    """
    spatial_dims = len(X.shape) - 1
    bw = X.shape[0] - 1

    _, kernel = get_kernel(bw, mode='triangular')
    kernel = kernel[:bw + 1]
    kernel = kernel.reshape((-1,) + (1,) * spatial_dims)

    R = np.sum(X * kernel, axis=0)
    S = np.sum(kernel) + 1e-10

    estimator = (R / S)

    return estimator


def lle_boundary(X: np.ndarray) -> np.ndarray:
    """
    Calculate the local linear estimator of the rightmost point t=1. First
    dimension should index time.

    :param X: Time series given as numpy array.
    :return: Returns the local linear estimator(s) given as numpy array.
    """
    spatial_dims = len(X.shape) - 1
    bw = X.shape[0] - 1

    kernel_support, kernel = get_kernel(bw, mode='triangular')
    kernel_support, kernel = kernel_support[:bw + 1], kernel[:bw + 1]
    kernel = kernel.reshape((-1,) + (1,) * spatial_dims)
    kernel_support = kernel_support.reshape((-1,) + (1,) * spatial_dims)

    supp_kern = kernel_support * kernel
    supp2_kern = kernel_support ** 2 * kernel

    S0 = np.sum(kernel)
    S1 = np.sum(supp_kern)
    S2 = np.sum(supp2_kern)

    R0 = np.sum(X * kernel, axis=0)
    R1 = np.sum(X * supp_kern, axis=0)

    denominator = S0 * S2 - S1 ** 2
    mu_hat = (S2 * R0 - S1 * R1) / (denominator + 1e-10)

    return mu_hat


if __name__ == '__main__':
    ESTIMATORS = {'NWE': nwe_boundary, 'LLE': lle_boundary}
    n_space, n_repetitions = 100, 10

    for n_time in [50, 100, 200, 500]:
        start = time()
        results = experiment(n_time, n_space, n_repetitions, estimators=ESTIMATORS,
                             single_est=single_estimator_boundary)
        end = time()
        print(end - start)