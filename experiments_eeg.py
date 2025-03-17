#
# experiments_eeg.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-03-06
# Author: Florian Heinrichs
#
# Script to load EEG eye tracking data. See:
# Vasconcelos Afonso, T., & Heinrichs, F. (2025). Consumer-Grade EEG and
# Eye-Tracking Dataset (1.0.1) [Data set]. Zenodo.
# https://doi.org/10.5281/zenodo.14860668
# Python package for data handling:
#   https://github.com/FlorianHeinrichs/eeg_eye_tracking

import json
from time import time
from typing import Callable

from eeg_et_benchmark.load_data import load_dataset
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import ccf

from nadaraya_watson_estimation import nadaraya_watson_estimator
from local_linear_estimation import local_linear_estimator, jackknife_estimator


def frequency_filter(X: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Frequency filter as benchmark.

    :param X: Time series given as numpy array.
    :return: Returns the filtered time series.
    """
    nyq = 0.5 * 256 / 5
    f0, f1, low, high = 50 / nyq, 60 / nyq, 0.5 / nyq, 40 / nyq

    f0 = f0 - int(f0)
    f1 = f1 - int(f1)
    low = low - int(low)
    high = high - int(high)

    filters = [
        (2, [f0 - 1 / 30, f0 + 1 / 30], 'bandstop'),
        (2, [f1 - 1 / 30, f1 + 1 / 30], 'bandstop'),
        (5, [low, high], 'bandpass')
    ]

    X_filtered = X.copy()

    for order, freqs, filter_type in filters:
        sos = butter(order, freqs, btype=filter_type, output="sos")
        X_filtered = sosfiltfilt(sos, X_filtered, axis=0)

    return X_filtered


ESTIMATORS = {'NWE': nadaraya_watson_estimator,
              'LLE': local_linear_estimator,
              'Jackknife': jackknife_estimator,
              'Filter': frequency_filter}


def experiment(folder: str = None, filepath: str = None) -> dict:
    """
    Compare MAE and MSE of the three estimators (NWE, LLE, Jackknife) with that
    of a frequency filter.

    :param folder: Folder to EEG data (if locally available).
    :param filepath: Path to save the results. Defaults to None, in which case
        no results are saved.
    :return: Results of experiments as nested dictionary, with levels:
        1. Recording with keys 'Pxxx_yy'
        2. Estimator: ['NWE', 'LLE', 'Jackknife', 'Filter']
        3. Metrics: ['time', 'MAE', 'MSE']
    """
    task = "level-2-smooth"
    eeg_columns = ['EEG_TP9', 'EEG_AF7', 'EEG_AF8', 'EEG_TP10']
    exclude = ["P002_01", "P004_01"] + [
        f"P0{k}_01" for k in list(range(16, 21)) + list(range(62, 68)) + [79]]

    recordings = load_dataset(task=task, exclude=exclude, folder=folder)
    recordings = recordings[0] + recordings[1]

    results = {}

    for i, rec in enumerate(recordings):
        results[i] = {}

        eeg = rec[eeg_columns].to_numpy()
        X = np.lib.stride_tricks.sliding_window_view(
            eeg, window_shape=(50, 1))[::5]
        X = np.squeeze(X, axis=-1)
        y = rec[['Stimulus_x', 'Stimulus_y']].to_numpy()[:-50:5]

        for estimator_name, estimator in ESTIMATORS.items():
            res = single_estimator(X, y, estimator, estimator_name)
            results[i][estimator_name] = res

            if filepath is not None:
                with open(filepath, 'w') as json_file:
                    json.dump(results, json_file, indent=4)

    return results


def single_estimator(X: np.ndarray,
                     y: np.ndarray,
                     estimator: Callable,
                     name: str,
                     norm: str = 'l2') -> tuple:
    """
    Auxiliary function to calculate estimates, and use smooth version of the
    data to predict the target y.

    :param X: NumPy array containing EEG as functional time series:
        (n_time, n_space, 4).
    :param y: NumPy array containing the target on screen: (n_time, 2)
    :param estimator: Estimator for mean estimation.
    :param name: Name of estimator in ['NWE', 'LLE', 'Jackknife'].
    :param norm: Norm (either 'l2' or 'sup') used to project functional time
        series onto multivariate time series.
    :return: Results as tuple: 'time', 'correlation'.
    """
    bw = 50

    start = time()
    estimate = estimator(X, bw)
    end = time()

    if name in ['LLE', 'Jackknife']:
        estimate = estimate[0]

    if norm == 'l2':
        estimate_norm = np.sqrt(np.mean(estimate ** 2, axis=-1))
    elif norm == 'sup':
        estimate_norm = np.max(np.abs(estimate), axis=-1)
    else:
        raise ValueError(f"Norm {norm} is not supported.")

    lags = []
    for i in range(2):
        for j in range(4):
            cross_corr = ccf(estimate_norm[:, j], y[:, i], adjusted=False, nlags=150)
            lags.append(np.argmax(cross_corr))
    lag = int(np.round(np.mean(lags)))

    model = LinearRegression()
    estimate_lag = estimate_norm[lag:]
    model.fit(estimate_lag, y[:len(estimate_lag)])
    y_pred = model.predict(estimate_lag)

    corr = [np.corrcoef(y[:len(estimate_lag), i], y_pred[:, i])[0, 1] for i in range(2)]

    return end - start, corr


if __name__ == '__main__':
    fp = "results.json"
    experiment(filepath=fp)
