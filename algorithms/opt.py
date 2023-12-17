"""
Author: DURUII
Date: 2023/12/17

Ref:
1. Experimental Methodology in the paper
"""

import numpy as np

from algorithms.base import BaseAlgorithm
from arms import StrategicArm


class Opt(BaseAlgorithm):
    def __init__(self, arms: list[StrategicArm], n_arms: int, n_selected: int, budget: float):
        super().__init__(arms, n_arms, n_selected, budget)

    def initialize(self):
        """ sort N arms in descending order of p.p.r """
        self.arms.sort(key=lambda arm: arm.mu / arm.b, reverse=True)

    def run(self):
        """ selects the top K arms according to r/b every round """
        b = np.array([arm.b for arm in self.arms])

        while True:
            omni = self.omniscience()

            # select Phi
            mask = np.zeros(self.N)
            mask[:self.K] = 1

            # payment p
            p = b * mask

            # update t, R, B
            self.t += 1

            if np.sum(p) >= self.B:
                break

            self.R += np.sum(omni * mask)
            self.B -= np.sum(p)

        return self.R, self.t
