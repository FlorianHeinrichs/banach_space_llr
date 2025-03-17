#
# local_linear_estimation.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-03-05
# Author: Florian Heinrichs
#
# Script to estimate the mean through local linear estimation.

import numpy as np

from tools import get_kernel, convolve


def local_linear_estimator(X: np.ndarray,
                           bw: int,
                           filter_array: np.ndarray = None) -> tuple:
    """
    Use local linear regression to estimate mu and its Frechet derivative.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator as int.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the local linear estimators.
    """
    spatial_dims = len(X.shape) - 1
    n_time = X.shape[0]

    if filter_array is None:
        filter_array = np.ones_like(X, dtype=bool)

    X_filtered = X.copy()
    X_filtered[~filter_array] = 0

    X_support = np.ones_like(X_filtered)
    X_support[~filter_array] = 0

    kernel_support, kernel = get_kernel(bw)
    kernel = kernel.reshape((1,) * spatial_dims + (-1,))
    supp_kern = kernel_support * kernel
    supp2_kern = kernel_support ** 2 * kernel

    padding = ((bw, bw),) + spatial_dims * ((0, 0),)
    X_filtered = np.pad(X_filtered, padding, mode='edge')
    X_support = np.pad(X_support, padding, mode='edge')

    S0 = convolve(X_support, kernel)
    S1 = convolve(X_support, supp_kern[..., ::-1])
    S2 = convolve(X_support, supp2_kern[..., ::-1])

    R0 = convolve(X_filtered, kernel)
    R1 = convolve(X_filtered, supp_kern[..., ::-1])

    denominator = S0 * S2 - S1 ** 2
    mu_hat = (S2 * R0 - S1 * R1) / (denominator + 1e-10)
    mu_prime_hat = (S0 * R1 - S1 * R0) / (bw / n_time * denominator + 1e-10)

    return mu_hat, mu_prime_hat


def jackknife_estimator(X: np.ndarray,
                        bw: int,
                        filter_array: np.ndarray = None) -> tuple:
    """
    Function to calculate the Jackknife version of the local linear estimators.

    :param X: Time series given as numpy array.
    :param bw: Bandwidth of the estimator.
    :param filter_array: Filter array to leave out certain observations (used
        for cross validation).
    :return: Returns the Jackknife estimators given as numpy array.
    """
    mu_hat, mu_prime_hat = local_linear_estimator(
        X, bw, filter_array=filter_array
    )

    mu_hat2, mu_prime_hat2 = local_linear_estimator(
        X, int(bw / np.sqrt(2)), filter_array=filter_array
    )

    mu_tilde = 2 * mu_hat2 - mu_hat
    mu_prime_tilde = (np.sqrt(2) / (np.sqrt(2) - 1) * mu_prime_hat2
                      - mu_prime_hat / (np.sqrt(2) - 1))

    return mu_tilde, mu_prime_tilde


if __name__ == '__main__':
    from visualize_data import plot_3d
    from simulate_data import mu1, generate_iid

    mean_operator = mu1(50, 100)
    noise = generate_iid(50, 100, 'BM')
    signal = mean_operator + noise

    estimator = local_linear_estimator(signal, 10)[0]
    plot_3d(mean_operator, y=estimator)