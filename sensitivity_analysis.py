#
# sensitivity_analysis.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-06-05
# Author: Florian Heinrichs
#
# Script containing experiments to study the sensitivity of local linear
# regression to the choice of bandwidth in real data scenarios.

import json
from typing import Callable

import numpy as np

from experiments_eeg import experiment
from experiments_video import load_data
from local_linear_estimation import local_linear_estimator
from tools import bandwidth_cv


def single_estimator_eeg(X: np.ndarray,
                         y: np.ndarray,
                         estimator: Callable,
                         name: str,
                         norm: str = 'l2') -> dict:
    """
    Auxiliary function to calculate estimates, and use smooth version of the
    data to predict the target y.

    :param X: NumPy array containing EEG as functional time series:
        (n_time, n_space, 4).
    :param y: Dummy variable for compatibility with experiment().
    :param estimator: Estimator for mean estimation.
    :param name: Dummy variable for compatibility with experiment().
    :param norm: Dummy variable for compatibility with experiment().
    :return: Results as tuple: 'time', 'correlation'.
    """
    min_bw, max_bw, step_size = 5, 250, 5

    base_estimator = lambda *x: estimator(*x)[0]
    bandwidths, mses = bandwidth_cv(X, min_bw, max_bw, base_estimator,
                                    return_mses=True, step_size=step_size)

    result = {bw: mse.tolist()
              for bw, mse in zip(range(min_bw, max_bw + 1, step_size), mses)}

    return result


def experiment_video(videos: list, filepath: str = None):
    """
    Calculate MSE of local linear regressor for different bandwidths using
    cross validation.

    :param videos: List of videos to use for experiments. Every video is given
        as a tuple with entries (name, folder), e.g.
        ('WalkByShop1front', 'path/to/video').
    :param filepath: Path to save the results. Defaults to None, in which case
        no results are saved.
    """

    min_bw, max_bw, step_size = 5, 250, 5

    base_estimator = lambda *x: local_linear_estimator(*x)[0]
    results = {}

    for src, folder in videos:
        video = load_data(folder, src=src)

        bandwidths, mses = bandwidth_cv(video, min_bw, max_bw, base_estimator,
                                        return_mses=True, step_size=step_size)
        result = {bw: mse.tolist()
                  for bw, mse in zip(range(min_bw, max_bw + 1, step_size), mses)}
        results[src] = result

        if filepath is not None:
            with open(filepath, 'w') as json_file:
                json.dump(results, json_file, indent=4)


if __name__ == '__main__':
    fp = "../results/mse_eeg.json"
    estimators = {'LLE': local_linear_estimator}
    folder = "../../BCI_ET_Benchmark/data/csv_preprocessed"

    experiment(folder=folder, filepath=fp, estimators=estimators,
               single_est=single_estimator_eeg)

    videos = [('WalkByShop1front', '../data/WalkByShop1front'),
              ('LeftBag_AtChair_jpg', '../data/LeftBag_AtChair_jpg')]
    fp = "../results/mse_video.json"
    experiment_video(videos, filepath=fp)