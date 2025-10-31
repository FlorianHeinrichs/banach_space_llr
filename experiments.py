#
# experiments.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-03-05
# Author: Florian Heinrichs
#
# Main script to conduct experiments.

import json
from time import time
from typing import Callable

import numpy as np

from config import CONFIG_MEAN, CONFIG_ERRORS
from nadaraya_watson_estimation import nadaraya_watson_estimator
from local_linear_estimation import local_linear_estimator, jackknife_estimator
from tools import bandwidth_cv, approximate_derivative

ESTIMATORS = {'NWE': nadaraya_watson_estimator,
              'LLE': local_linear_estimator,
              'Jackknife': jackknife_estimator}


def experiment(n_time: int,
               n_space: int,
               n_repetitions: int,
               filepath: str = None,
               estimators: dict = None,
               single_est: Callable = None) -> dict:
    """
    Compare MAE and MSE of the three estimators (NWE, LLE, Jackknife) for mu and
    its derivative.

    :param n_time: Number of time points.
    :param n_space: Number of spatial (grid) points.
    :param n_repetitions: Number of simulated time series per model.
    :param filepath: Path to save the results. Defaults to None, in which case
        no results are saved.
    :param estimators: Dictionary containing estimators to use. If None,
        defaults to the global variable ESTIMATORS.
    :param single_est: Function called for each estimator. Defaults to
        single_estimator.
    :return: Results of experiments as nested dictionary, with levels:
        1. Mean: ['mu_1', 'mu_2']
        2. Errors: ['IID_BM','IID_BB', 'FAR_BM', 'FAR_BB',
                    'Heteroscedastic_Independent', 'Heteroscedastic_Dependent_1',
                    'Heteroscedastic_Dependent_2']
        3. Estimator: ['NWE', 'LLE', 'Jackknife']
        4. Metrics: ['time', 'MAE', 'MSE']
    """
    if estimators is None:
        estimators = ESTIMATORS

    if single_est is None:
        single_est = single_estimator

    results = {}
    for mean_name, mean_func, mean_kwargs in CONFIG_MEAN:
        results[mean_name] = {}
        mean = mean_func(n_time, n_space, **mean_kwargs)

        for error_name, error_func, error_kwargs in CONFIG_ERRORS:
            results[mean_name][error_name] = {}
            data = np.stack([mean + error_func(n_time, n_space, **error_kwargs)
                             for _ in range(n_repetitions)], axis=-1)

            for estimator_name, estimator in estimators.items():
                res = single_est(data, estimator, estimator_name, mean)
                results[mean_name][error_name][estimator_name] = res

                if filepath is not None:
                    with open(filepath, 'w') as json_file:
                        json.dump(results, json_file, indent=4)

    return results

def single_estimator(X: np.ndarray,
                     estimator: Callable,
                     name: str,
                     mean: np.ndarray,
                     tune_drv_bw: bool = True) -> dict:
    """
    Auxiliary function to calculate estimates and compare their MAE and MSE.

    :param X: NumPy array containing functional time series: (n_time, n_space).
    :param estimator: Estimator for mean estimation.
    :param name: Name of estimator in ['NWE', 'LLE', 'Jackknife'].
    :param mean: (Real) Mean operator, used to calculate MAE and MSE.
    :param tune_drv_bw: Specifies whether bandwidth of derivative is tuned
        separately. Defaults to True.
    :return: Results as dictionary with keys 'MAE', 'MSE', 'MAE_drv', 'MSE_drv',
        'time'.
    """
    n_time = X.shape[0]
    min_bw, max_bw = 1, max(n_time // 2 - 1, 1)

    if name == 'NWE':
        base_estimator = estimator
        drv_estimator = lambda *x: approximate_derivative(estimator(*x))
    else:
        base_estimator = lambda *x: estimator(*x)[0]
        drv_estimator = lambda *x: estimator(*x)[1]

    bandwidths = bandwidth_cv(X, min_bw, max_bw, base_estimator, batch_axis=2)

    if tune_drv_bw:
        X_drv = approximate_derivative(X)
        bandwidths_drv = bandwidth_cv(X_drv, min_bw, max_bw, drv_estimator,
                                      batch_axis=2)
    else:
        bandwidths_drv = bandwidths

    mean_drv = approximate_derivative(mean)

    results = {'time': [], 'MAE': [], 'MSE': [], 'MAE_drv': [], 'MSE_drv': []}

    X_transposed = np.moveaxis(X, -1, 0)

    for x, bw, bw_drv in zip(X_transposed, bandwidths, bandwidths_drv):

        start = time()

        if tune_drv_bw:
            if name == 'NWE':
                estimate = estimator(x, bw)
                estimate_drv = approximate_derivative(estimator(x, bw_drv))
            else:
                estimate = estimator(x, bw)[0]
                estimate_drv = estimator(x, bw_drv)[1]
        else:
            estimate = estimator(x, bw)

            if name == 'NWE':
                estimate_drv = approximate_derivative(estimate)
            else:
                estimate, estimate_drv = estimate

        end = time()

        mae = np.mean(np.abs(estimate - mean))
        mse = np.mean((estimate - mean) ** 2)

        mae_drv = np.mean(np.abs(estimate_drv - mean_drv))
        mse_drv = np.mean((estimate_drv - mean_drv) ** 2)

        results['time'].append(end - start)
        results['MAE'].append(mae)
        results['MSE'].append(mse)
        results['MAE_drv'].append(mae_drv)
        results['MSE_drv'].append(mse_drv)

    return results


if __name__ == '__main__':
    n_time, n_space, n_repetitions = 50, 100, 2

    results = experiment(n_time, n_space, n_repetitions)

    print(results)