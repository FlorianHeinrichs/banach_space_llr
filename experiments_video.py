#
# experiments_video.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-03-10
# Author: Florian Heinrichs
#
# Script for experiments related to the dataset (videos) from the EC Funded
# CAVIAR project/IST 2001 37540. See: https://homepages.inf.ed.ac.uk/rbf/CAVIAR/

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

from local_linear_estimation import local_linear_estimator, jackknife_estimator
from nadaraya_watson_estimation import nadaraya_watson_estimator


ESTIMATORS = {'NWE': nadaraya_watson_estimator,
              'LLE': local_linear_estimator,
              'Jackknife': jackknife_estimator}


def load_data(folder: str,
              start: int = None,
              end: int = None,
              src: str = 'WalkByShop1front') -> np.ndarray:
    """
    Function to load the video from a folder containing images as jpg, as
    obtained from the project's website.

    :param folder: Path to folder
    :param start: Start of video extract.
    :param end: End of video extract.
    :param src: Source of video (either 'WalkByShop1front' or 'LeftBag_AtChair').
    :return: Video as Numpy array.
    """

    if src == 'WalkByShop1front' and None in [start, end]:
        start, end = 0, 2360
    elif src == 'LeftBag_AtChair':
        start, end = 254, 744
    else:
        raise ValueError(f"{src=} unknown.")

    image_files = [f"{src}{k:04d}.jpg" for k in range(start, end + 1)]

    images_array = []

    for image_file in image_files:
        img = Image.open(os.path.join(folder, image_file))
        img_array = np.array(img)
        images_array.append(img_array)

    images_array = np.array(images_array)

    return images_array


def store_video(x: np.ndarray, fp: str):
    """
    Auxiliary function to store video.

    :param x: Numpy array with image data.
    :param fp: Filepath to video. Should end with ".avi".
    """
    video_data = x.astype(np.uint8)
    height, width = x.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(fp, fourcc, 25.0, (width, height))

    for frame in video_data:
        output_video.write(frame)

    output_video.release()


def standardize_video(x: np.ndarray) -> np.ndarray:
    """
    Auxiliary function to standardize video with values between 0 and 255.

    :param x: Numpy array with image data.
    :return: Standardized video.
    """
    x_min = np.min(x, axis=(1, 2, 3), keepdims=True)
    x_max = np.max(x, axis=(1, 2, 3), keepdims=True)
    x = 256 * (x - x_min) / (x_max - x_min)

    return x


def display_results(results: dict, y_label: str = ''):
    """
    Auxiliary function to visualize results obtained from calculate_estimates().

    :param results: Results as dictionary - output of calculate_estimates().
    :param y_label: Label string to use for y-axis.
    """
    estimator_labels = {'NWE': '$\\hat{\\mu}_{NW}$',
                        'LLE': '$\\hat{\\mu}_{h_n}$',
                        'Jackknife': '$\\tilde{\\mu}_n$'}

    for est_name, res in results.items():
        plt.plot(res[0][1:-1], label=estimator_labels[est_name])
    plt.legend()
    plt.xlabel("Frame Number", labelpad=-30)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


def store_smooth_video(video: np.ndarray, results: dict, fp: str):
    """
    Auxiliary function to store original video, concatenated with smoothed
    versions of the video:
    - Top left: Original
    - Top right: Nadaraya-Watson estimate
    - Bottom left: Local linear estimate
    - Bottom right: Jackknife estimate

    :param results: Results as dictionary - output of calculate_estimates().
    """
    video = standardize_video(video)
    video = np.concatenate([
        np.concatenate([video, results['NWE'][1]],  axis=2),
        np.concatenate([results['LLE'][1], results['Jackknife'][1]], axis=2)
        ], axis=1)
    store_video(video, fp)


def calculate_estimates(video: np.ndarray,
                        bw: int,
                        cusum: bool = False,
                        norm: str = 'l2') -> dict:
    """
    Function to smooth the video, calculate norm of the residuals (or the CUSUM
    process of the latter).

    :param video: Original video.
    :param bw: Bandwidth of kernel estimators.
    :param cusum: Boolean indicating whether to calculate the CUSUM process (if
        True), or the norm of the residuals (if False).
    :param norm: Norm to use (either 'l2' or 'sup').
    :return: Results as dictionary with estimator names ('NWE', 'LLE',
        'Jackknife') as keys and (norm_of_residuals, smooth_video) as values.
    """
    results = {}

    for est_name, estimator in ESTIMATORS.items():

        mean_local = estimator(video, bw)

        if est_name != 'NWE':
            mean_local = mean_local[0]

        residual = video - mean_local

        if norm == 'l2':
            residual_norm = np.sqrt(np.mean(residual ** 2, axis=(1, 2, 3)))
        elif norm == 'sup':
            residual_norm = np.max(np.abs(residual), axis=(1, 2, 3))
        else:
            raise ValueError(f"{norm=} not supported.")

        if cusum:
            n = len(residual_norm)
            cumsum = np.cumsum(residual_norm)
            cusum_process = cumsum / n - np.arange(1, n + 1) / n * cumsum[-1] / n
            result = cusum_process
        else:
            result = residual_norm

        results[est_name] = result, standardize_video(mean_local)

    return results


def detect_outliers(folder: str, video_path: str = None):
    """
    Main function to apply smoothing to "WalkByShop1front" for outlier
    detection. A video containing the original video and the smooth versions
    is stored optionally.

    :param folder: Folder containing the video as images.
    :param video_path: Path to video. If None (default), video is not stored.
    """
    start, end, bw = 0, 2359, 50
    src = 'WalkByShop1front'

    video = load_data(folder, start=start, end=end, src=src)
    results = calculate_estimates(video, bw, cusum=False, norm='l2')

    display_results(results, y_label="$L^2$-norm")

    if video_path is not None:
        store_smooth_video(video, results, video_path)


def detect_change(folder: str, video_path: str = None):
    """
    Main function to apply smoothing to "LeftBag_AtChair_jpg" for change point
    detection. A video containing the original video and the smooth versions
    is stored optionally.

    :param folder: Folder containing the video as images.
    :param video_path: Path to video. If None (default), video is not stored.
    """
    start, end, bw = 254, 744, 100
    src = 'LeftBag_AtChair_jpg'

    video = load_data(folder, start=start, end=end, src=src)
    results = calculate_estimates(video, bw, cusum=True, norm='sup')

    display_results(results, y_label='CUSUM')

    if video_path is not None:
        store_smooth_video(video, results, video_path)


def make_video(folder: str, video_path: str):
    """
    Auxiliary function to create a video consisting of the original video,
    a smooth version (using the Nadaraya-Watson estimator), and the calulated
    residuals.
    """
    start, end, bw = 254, 744, 100
    src = 'left_bag_chair'

    video = load_data(folder, start=start, end=end, src=src)
    mean_local = nadaraya_watson_estimator(video, bw)
    residual = video - mean_local

    final_video = np.concatenate(
        [standardize_video(x) for x in [video, mean_local, residual]], axis=2)
    store_video(final_video, video_path)


if __name__ == '__main__':
    folder = "path/to/data"
    detect_change(folder)
    # detect_outliers(folder)
    # make_video(folder, video_path)
