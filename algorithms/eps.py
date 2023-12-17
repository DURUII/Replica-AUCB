"""
Author: DURUII
Date: 2023/12/17

Ref:
1. Experimental Methodology in the paper
2. separated.py
"""

import numpy as np

from algorithms.base import BaseAlgorithm
from arms import StrategicArm
import random


class EpsilonFirst(BaseAlgorithm):
    def __init__(self, arms: list[StrategicArm], n_arms: int, n_selected: int, budget: float, epsilon: float):
        super().__init__(arms, n_arms, n_selected, budget)
        self.eps = epsilon

        # average sampling/empirical rewards
        self.beta, self.r_bar = np.zeros(n_arms), np.zeros(n_arms)

        # placeholder
        self.budget_exploration = None
        self.budget_exploitation = None

    def initialize(self) -> None:
        self.budget_exploration = self.B * self.eps
        self.budget_exploitation = self.B - self.budget_exploration

    def run(self):
        # exploration
        while True:
            omni = self.omniscience()

            # select Phi randomly
            mask = np.zeros(self.N)
            mask[np.random.choice(np.arange(self.N), self.K)] = 1

            # payment p
            # FIXME p is not mentioned in this paper
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
            criteria, mask = self.r_bar / b, np.zeros(self.N)
            arrange = np.argsort(criteria)[::-1]
            mask[arrange[:self.K]] = 1

            # payment
            deno = self.r_bar[arrange[self.K]]
            p = np.minimum(self.r_bar / deno * b[arrange[self.K]], StrategicArm.c_max) * mask

            # update
            self.t += 1
            if np.sum(p) >= self.B:
                break

            self.r_bar = ((self.r_bar * self.beta + omni) / (self.beta + np.ones(self.N))) * mask + self.r_bar * (
                    1 - mask)
            self.beta += mask

            self.R += np.sum(omni * mask)
            self.B -= np.sum(p)

        return self.R, self.t
