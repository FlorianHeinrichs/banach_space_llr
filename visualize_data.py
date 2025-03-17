#
# visualize_data.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-03-05
# Author: Florian Heinrichs
#
# Script to visualize data.

import numpy as np
import matplotlib.pyplot as plt


def plot_3d(x: np.ndarray,
            y: np.ndarray = None,
            elev: int = 30,
            azim: int = -60):
    n_time, n_space = x.shape[0], x.shape[1]
    X, Y = np.meshgrid(np.arange(n_space) / n_space,
                       np.arange(n_time) / n_time)

    ax_idx = 111 if y is None else 121

    fig = plt.figure()
    ax = fig.add_subplot(ax_idx, projection='3d')
    ax.plot_surface(X, Y, x, cmap='viridis', edgecolor='none')

    ax.set_xlabel('Space')
    ax.set_ylabel('Time')
    ax.set_zlabel('Function Value')
    ax.view_init(elev=elev, azim=azim)

    if y is not None:
        n_time, n_space = y.shape[0], y.shape[1]
        X, Y = np.meshgrid(np.arange(n_space) / n_space,
                           np.arange(n_time) / n_time)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X, Y, y, cmap='viridis', edgecolor='none')
        ax2.view_init(elev=elev, azim=azim)
        ax2.set_xlabel('Space')
        ax2.set_ylabel('Time')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from simulate_data import mu1, mu2

    mean_operator = mu1(50, 100)
    mean_operator2 = mu2(50, 100)
    plot_3d(mean_operator, y=mean_operator2, azim=-40)
