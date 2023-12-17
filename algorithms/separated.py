"""
Author: DURUII
Date: 2023/12/17

Ref:
1. Experimental Methodology in the paper
2. aucb.py
"""
import math

import numpy as np

from algorithms.base import BaseAlgorithm
from arms import StrategicArm


class Separated(BaseAlgorithm):
    def __init__(self, arms: list[StrategicArm], n_arms: int, n_selected: int, budget: float):
        super().__init__(arms, n_arms, n_selected, budget)


        self.budget_exploration = 0.0
        self.budget_exploitation = 0.0

        # average sampling/empirical rewards
        self.beta, self.r_bar = np.zeros(n_arms), np.zeros(n_arms)

    def initialize(self) -> None:
        # B_1 in the paper
        self.budget_exploration = (StrategicArm.c_max * self.N * math.log(self.N * self.B)) ** (1 / 3) * self.B ** (
                2 / 3) / 2 ** (1 / 3)

        self.budget_exploitation = self.B - self.budget_exploration

    def run(self):
        # exploration
        lo, hi = 0, self.K - 1
        while True:
            omni = self.omniscience()

            # select Phi
            mask = np.zeros(self.N)
            if lo < hi:
                mask[lo:hi + 1] = 1
            else:
                mask[:hi + 1] = 1
                mask[lo:] = 1

            lo = (lo + self.K) % self.N
            hi = (hi + self.K) % self.N

            # payment p
            p = StrategicArm.c_max * mask

            # update t, R, B, r_bar, beta
            self.t += 1
            if self.B - sum(p) <= self.budget_exploitation:
                break

            self.r_bar = ((self.r_bar * self.beta + omni) / (self.beta + np.ones(self.N))) * mask + self.r_bar * (
                    1 - mask)
            self.beta += mask

            self.R += np.sum(omni * mask)
            self.B -= np.sum(p)

        # exploitation
        b = np.array([arm.b for arm in self.arms])
        while True:
            omni = self.omniscience()

            # select Phi
            u = np.sqrt(self.N * StrategicArm.c_max * np.log(self.N * self.B) / 2 * self.budget_exploration)
            r_tilde = self.r_bar + u

            criteria, mask = r_tilde / b, np.zeros(self.N)
            arrange = np.argsort(criteria)[::-1]
            mask[arrange[:self.K]] = 1

            # payment
            deno = r_tilde[arrange[self.K]]
            p = np.minimum(r_tilde / deno * b[arrange[self.K]], StrategicArm.c_max) * mask

            # update t, R, B, regardless of beta and r_bar
            self.t += 1
            if np.sum(p) >= self.B:
                break

            self.R += np.sum(omni * mask)
            self.B -= np.sum(p)

        return self.R, self.t
