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
               filepath: str = None) -> dict:
    """
    Compare MAE and MSE of the three estimators (NWE, LLE, Jackknife) for mu and
    its derivative.

    :param n_time: Number of time points.
    :param n_space: Number of spatial (grid) points.
    :param n_repetitions: Number of simulated time series per model.
    :param filepath: Path to save the results. Defaults to None, in which case
        no results are saved.
    :return: Results of experiments as nested dictionary, with levels:
        1. Mean: ['mu_1', 'mu_2']
        2. Errors: ['IID_BM','IID_BB', 'FAR_BM', 'FAR_BB',
                    'Heteroscedastic_Independent', 'Heteroscedastic_Dependent_1',
                    'Heteroscedastic_Dependent_2']
        3. Estimator: ['NWE', 'LLE', 'Jackknife']
        4. Metrics: ['time', 'MAE', 'MSE']
    """
    results = {}
    for mean_name, mean_func, mean_kwargs in CONFIG_MEAN:
        results[mean_name] = {}
        mean = mean_func(n_time, n_space, **mean_kwargs)

        for error_name, error_func, error_kwargs in CONFIG_ERRORS:
            results[mean_name][error_name] = {}
            data = np.stack([mean + error_func(n_time, n_space, **error_kwargs)
                             for _ in range(n_repetitions)], axis=-1)

            for estimator_name, estimator in ESTIMATORS.items():
                res = single_estimator(data, estimator, estimator_name, mean)
                results[mean_name][error_name][estimator_name] = res

                if filepath is not None:
                    with open(filepath, 'w') as json_file:
                        json.dump(results, json_file, indent=4)

    return results

def single_estimator(X: np.ndarray,
                     estimator: Callable,
                     name: str,
                     mean: np.ndarray) -> dict:
    """
    Auxiliary function to calculate estimates and compare their MAE and MSE.

    :param X: NumPy array containing functional time series: (n_time, n_space).
    :param estimator: Estimator for mean estimation.
    :param name: Name of estimator in ['NWE', 'LLE', 'Jackknife'].
    :param mean: (Real) Mean operator, used to calculate MAE and MSE.
    :return: Results as dictionary with keys 'MAE', 'MSE', 'MAE_drv', 'MSE_drv',
        'time'.
    """
    n_time = X.shape[0]
    min_bw, max_bw = 1, int(np.sqrt(n_time))
    base_estimator = estimator if name == 'NWE' else lambda *x: estimator(*x)[0]
    bandwidths = bandwidth_cv(X, min_bw, max_bw, base_estimator, batch_axis=2)

    mean_drv = approximate_derivative(mean)

    results = {'time': [], 'MAE': [], 'MSE': [], 'MAE_drv': [], 'MSE_drv': []}

    X_transposed = np.moveaxis(X, -1, 0)

    for x, bw in zip(X_transposed, bandwidths):
        start = time()
        estimate = estimator(x, bw)
        end = time()

        if name == 'NWE':
            estimate_drv = approximate_derivative(estimate)
        else:
            estimate, estimate_drv = estimate

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
    n_time, n_space, n_repetitions = 50, 30, 3

    results = experiment(n_time, n_space, n_repetitions)

    print(results)