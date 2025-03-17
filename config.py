#
# config.py
#
# Project: Local Linear Regression for Time Series in Banach Spaces
# Date: 2025-03-05
# Author: Florian Heinrichs
#
# Configuration for experiments.

import numpy as np

from simulate_data import (mu1, mu2, generate_iid, generate_far,
                           generate_hetero_indep, generate_hetero_dep)

# CONFIG_MEAN entries: (name, function, kwargs)
CONFIG_MEAN = [
    ('mu_1', mu1, {'spatial_freq': 1, 'c': 1}),
    ('mu_2', mu2, {'coefficients': np.array([-8, 16, -11, 3, 1])})
]

# CONFIG_ERRORS entries: (name, function, kwargs)
CONFIG_ERRORS = [
    ('IID_BM', generate_iid, {'innovation_type': 'BM'}),
    ('IID_BB', generate_iid, {'innovation_type': 'BB'}),
    ('FAR_BM', generate_far, {'innovation_type': 'BM'}),
    ('FAR_BB', generate_far, {'innovation_type': 'BB'}),
    ('Heteroscedastic_Independent', generate_hetero_indep, {}),
    ('Heteroscedastic_Dependent_1', generate_hetero_dep, {'model': 1}),
    ('Heteroscedastic_Dependent_2', generate_hetero_dep, {'model': 2}),
]
