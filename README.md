# Local Linear Regression for Banach Space-Valued Time Series

This repository contains the code for the methods and experiments presented in the paper titled:

Title: A Note on Local Linear Regression for Time Series in Banach Spaces

Authors: Florian Heinrichs

### Overview

This repository includes code for the local linear and Jackknife estimators proposed in the paper. The goal of this work is to generalize local linear estimation to Banach space-valued (non-stationary) time series. The estimators are compared with the Nadaraya-Watson estimator.

### Requirements

To use the proposed methods, only NumPy and SciPy are required. Additional Python packages are required for the real data applications.

### Usage

The high-level functions implemting the proposed estimators are defined in `local_linear_estimation.py` (and the Nadaraya-Watson estimator in `nadaraya_watson_estimation.py`).

### Datasets

The datasets used for evaluation in the paper are available for download here:
- [Consumer-Grade EEG and Eye-Tracking Dataset](https://zenodo.org/records/14860668) ([Python Code](https://github.com/FlorianHeinrichs/eeg_eye_tracking)
- [EC Funded CAVIAR project/IST 2001 37540](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/)

- **Important**: The videos by the CAVIAR project are licensed under **Creative Commons BY-SA**. In particular, the processed videos in the "videos" folder are licensed under the same license. 

### Citation

If you use this code in your own work, please cite the following pre-print (or the peer reviewed paper, once available):

Heinrichs, F. (2025). A Note on Local Linear Regression for Time Series in Banach Spaces. *arXiv preprint arXiv:2503.15039*.

    @article{heinrichs2025locallinearregression,
      title={A Note on Local Linear Regression for Time Series in Banach Spaces},
      author={Heinrichs, Florian},
      journal={arXiv preprint arXiv:2503.15039},
      year={2025}
    }

    

### License

This project (**except of the processed CAVIAR videos**) is licensed under the MIT License - see the LICENSE file for details.
