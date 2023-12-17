import random

import numpy as np

from arms import StrategicArm


class Config:
    N = 60
    N_range = [50, 60, 70, 80, 90, 100]

    K = 20
    K_range = [10, 20, 30, 40, 50]

    B = 5e5
    B_range = [i * 10 for i in range(1, 11)]
    B_range = np.array(B_range) * 1e4

    line_styles = {
        'AUCB': {'color': '#060506', 'marker': 's', 'label': 'AUCB'},
        'optimal': {'color': '#ed1e25', 'marker': 'o', 'label': 'optimal'},
        'separated': {'color': '#3753a4', 'marker': '^', 'label': 'separated'},
        '0.1-first': {'color': '#097f80', 'marker': 'v', 'label': '0.1-first'},
        '0.5-first': {'color': '#ba529e', 'marker': '<', 'label': '0.5-first'},
    }
